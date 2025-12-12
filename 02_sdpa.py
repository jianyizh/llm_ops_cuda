
import torch
from common.utils import run_benchmark, common_cuda_flags, common_sycl_flags
from torch.utils.cpp_extension import load
from functools import partial
import os
os.environ["TORCH_XPU_ARCH_LIST"] = "pvc,bmg"
device = "xpu" if torch.xpu.is_available() else "cuda"
if device == "xpu":
    # host compiler by default is c++, which will cause compile error
    os.environ["CXX"] = "icpx"
SM = '89'  # 4090, Ada
if device == "cuda" and "A100" in torch.cuda.get_device_name():
    SM = '80'  # A100, Ampere


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    CUTLASS_REPO_PATH = os.environ.get(
        "CUTLASS_REPO_PATH", os.path.expanduser(
            "./third_party/cutlass") if device == "cuda" else "./third_party/cutlass-sycl"
    )

    macros = [
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        f"-DTORCH_CURRENT_DEVICE=cutlass::arch::Sm{SM}",
        f"-gencode=arch=compute_{SM},code=sm_{SM}",
        # "-g -lineinfo"
    ] if device == "cuda" else ["-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET"]

    # Load the CUDA kernel as a python module
    lib = None
    if device == "xpu":
        lib = load(
            name="flash_attention_lib",
            sources=["fmha/xpu/cute.sycl"],
            extra_sycl_cflags=common_sycl_flags,
            extra_cflags=["-std=c++17"] + macros,
            extra_include_paths=[os.path.join(
                CUTLASS_REPO_PATH, "include"), "./third_party/cutlass-sycl/tools/util/include", "./third_party/cutlass-sycl/tools/util/include", "./fmha/xpu"],
            # extra_ldflags=["-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier"],
            # I have to add sycl_dlink_post_cflags += ['-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier'] and remove spir64 from -fsycl-targets
            verbose=True,
        )
    print("-" * 80)
    batch = 1
    num_heads = 16
    seq_len = 1024
    head_dim = 128

    if device == "cuda":
        with torch.no_grad():
            q = torch.randn((batch, seq_len, num_heads, head_dim)
                            ).cuda().half().contiguous().transpose(1, 2)
            k = torch.randn((batch, seq_len, num_heads, head_dim)
                            ).cuda().half().contiguous().transpose(1, 2)
            v = torch.randn((batch, seq_len, num_heads, head_dim)
                            ).cuda().half().contiguous().transpose(1, 2)
            output = torch.zeros(
                (batch, num_heads, seq_len, head_dim)).cuda().half().contiguous()
            output, _ = run_benchmark(
                torch.nn.functional.scaled_dot_product_attention, q, k, v, tag="f16_torch")
            from flash_attn import flash_attn_func

            output, _ = run_benchmark(
                flash_attn_func, q, k, v, tag="f16_flash_attn")
    else:
        with torch.no_grad():
            q = torch.randn((batch, num_heads, seq_len, head_dim)
                            ).xpu().half().contiguous()
            k = torch.randn((batch, num_heads, seq_len, head_dim)
                            ).xpu().half().contiguous()
            v = torch.randn((batch, num_heads, seq_len, head_dim)
                            ).xpu().half().contiguous()
            output = torch.zeros(
                (batch, num_heads, seq_len, head_dim)).xpu().half().contiguous()
            output, _ = run_benchmark(
                torch.nn.functional.scaled_dot_product_attention, q, k, v, tag="f16_torch")
            output_a = output.clone()
            output, _ = run_benchmark(
                lib.cute_example, q, k, v, out=output, tag="f16_cutlass_xpu")
            print((output - output_a).abs().max())

    print("-" * 80)
