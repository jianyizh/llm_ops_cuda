
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

    CUTLASS_REPO_PATH = os.environ.get(
        "CUTLASS_REPO_PATH", os.path.expanduser(
            "./third_party/cutlass") if device == "cuda" else "./third_party/cutlass-sycl"
    )

    macros = [
        f"-DTORCH_CURRENT_DEVICE=cutlass::arch::Sm{SM}",
        f"-gencode=arch=compute_{SM},code=sm_{SM}",
    ] if device == "cuda" else ["-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET"]

    # Load the CUDA kernel as a python module
    lib = None
    if device == "xpu":
        lib = load(
            name="basic_gemm_lib",
            sources=["basic_gemm/basic_gemm.sycl", "basic_gemm/cute_0.sycl"],
            extra_sycl_cflags=common_sycl_flags,
            extra_cflags=["-std=c++17"] + macros,
            extra_include_paths=[os.path.join(
                CUTLASS_REPO_PATH, "include"), "./third_party/cutlass-sycl/tools/util/include"],
            #extra_ldflags=["-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier"],
            # I have add sycl_dlink_post_cflags += ['-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier']
            verbose=True,
        )
    else:
        lib = load(
            name="basic_gemm_lib",
            sources=["basic_gemm/basic_gemm.cu",
                     "basic_gemm/cute_0.cu"],
            extra_cuda_cflags=common_cuda_flags + macros,
            extra_cflags=["-std=c++17"],
            extra_include_paths=[os.path.join(CUTLASS_REPO_PATH, "include")],
            verbose=True,
        )

    print("-" * 80)
    M, N, K = 8192, 4096, 2048
    if device == "cuda":
        a = torch.randn((M, K)).cuda().half().contiguous()
        b = torch.randn((N, K)).cuda().half().contiguous().transpose(0, 1)
        c = torch.zeros((M, N)).cuda().half().contiguous()
        run_benchmark(partial(torch.matmul, out=c), a, b, tag="f16_torch")
        c_torch = c.cpu()
        run_benchmark(lib.basic_gemm, a, b, out=c, tag="cutlass_basic_gemm")
        run_benchmark(lib.cute_example, a, b, out=c, tag="cute_example_gemm")
        c_cute = c.cpu()
    if device == "xpu":
        a = torch.randn((M, K)).xpu().half().contiguous()
        b = torch.randn((N, K)).xpu().half().contiguous().transpose(0, 1)
        c = torch.zeros((M, N)).xpu().half().contiguous()
        run_benchmark(partial(torch.matmul, out=c), a, b, tag="f16_torch")
        c_torch = c.cpu()
        run_benchmark(lib.basic_gemm, a, b, out=c, tag="cutlass_basic_gemm")
        run_benchmark(lib.cute_example, a, b, out=c, tag="cute_example_gemm")
        c_cute = c.cpu()
    # print(c_torch, c_cute)
    print("-" * 80)
