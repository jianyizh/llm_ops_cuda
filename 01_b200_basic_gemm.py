# ncu --config-file off --export ./cute_b200_gemm_1 --force-overwrite --kernel-name gemm_device_0 --set full --import-source yes python 01_b200_basic_gemm.py
import torch
from common.utils import run_benchmark, common_cuda_flags, common_sycl_flags
from torch.utils.cpp_extension import load
from functools import partial
import os
device = "xpu" if torch.xpu.is_available() else "cuda"
SM = '89'  # 4090, Ada
if device == "cuda" and "B200" in torch.cuda.get_device_name():
    SM = '100a'  # B200, Blackwell
else:
    quit()


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    CUTLASS_REPO_PATH = os.environ.get(
        "CUTLASS_REPO_PATH", os.path.expanduser(
            "./third_party/cutlass") if device == "cuda" else "./third_party/cutlass-sycl"
    )

    macros = [
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
        f"-DTORCH_CURRENT_DEVICE=cutlass::arch::Sm{SM}",
        f"-gencode=arch=compute_{SM},code=sm_{SM}",
        # "-g -lineinfo"
    ] if device == "cuda" else ["-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET"]

    # Load the CUDA kernel as a python module
    lib = None
    if device == "xpu":
        lib = load(
            name="b200_basic_gemm_lib",
            sources=[],  # ["b200_basic_gemm/basic_gemm.sycl", "b200_basic_gemm/cute_0.sycl"],
            extra_sycl_cflags=common_sycl_flags,
            extra_cflags=["-std=c++17"] + macros,
            extra_include_paths=[os.path.join(
                CUTLASS_REPO_PATH, "include"), "./third_party/cutlass-sycl/tools/util/include"],
            # extra_ldflags=["-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier"],
            # I have to add sycl_dlink_post_cflags += ['-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier'] and remove spir64 from -fsycl-targets
            verbose=True,
        )
    else:
        lib = load(
            name="b200_basic_gemm_lib",
            sources=["b200_basic_gemm/basic_gemm.cu",
                     "b200_basic_gemm/cute_0.cu",
                     ],
            extra_cuda_cflags=common_cuda_flags + macros,
            extra_cflags=["-std=c++17"],
            extra_include_paths=[os.path.join(
                CUTLASS_REPO_PATH, "include"), "./third_party/cutlass/tools/util/include"],
            verbose=True,
        )

    print("-" * 80)
    M, N, K = 16384, 8192, 8192
    if device == "cuda":
        # a_tile = torch.arange(128*64).reshape(128, 64).half().cuda()/100.
        # a = a_tile.repeat(M//128, K//64)
        a = torch.randn((M, K)).cuda().half().contiguous()
        b = torch.randn((N, K)).cuda().half().contiguous().transpose(0, 1)
        c = torch.zeros((M, N)).cuda().half().contiguous()
        run_benchmark(partial(torch.matmul, out=c), a, b, tag="f16_torch")
        c_torch = c.cpu()
        run_benchmark(lib.basic_gemm, a, b, out=c, tag="cutlass_basic_gemm")
        run_benchmark(lib.cute_example, a, b, out=c, tag="cute_example_gemm")
        c_cute = c.cpu()
        print((c_cute == c_torch).all())
    if device == "xpu":
        a = torch.randn((M, K)).xpu().half().contiguous()
        b = torch.randn((N, K)).xpu().half().contiguous().transpose(0, 1)
        c = torch.zeros((M, N)).xpu().half().contiguous()
        run_benchmark(partial(torch.matmul, out=c), a, b, tag="f16_torch")
        c_torch = c.cpu()
        run_benchmark(lib.basic_gemm, a, b, out=c, tag="cutlass_basic_gemm")
        run_benchmark(lib.cute_example, a, b, out=c, tag="cute_example_gemm")
        c_cute = c.cpu()
        print((c_cute == c_torch).all())
    print("-" * 80)
