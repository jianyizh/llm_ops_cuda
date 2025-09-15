
import os
from functools import partial
from torch.utils.cpp_extension import load
from common.utils import run_benchmark, common_cuda_flags
import torch
SM = '89'  # 4090, Ada
if "A100" in torch.cuda.get_device_name():
    SM = '80'  # A100, Ampere


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    CUTLASS_REPO_PATH = os.environ.get(
        "CUTLASS_REPO_PATH", os.path.expanduser("./third_party/cutlass")
    )

    macros = [
        f"-DTORCH_CURRENT_DEVICE=cutlass::arch::Sm{SM}",
        f"-gencode=arch=compute_{SM},code=sm_{SM}",
    ]

    # Load the CUDA kernel as a python module
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
    a = torch.randn((M, K)).cuda().half().contiguous()
    b = torch.randn((N, K)).cuda().half().contiguous().transpose(0, 1)
    c = torch.zeros((M, N)).cuda().half().contiguous()
    run_benchmark(partial(torch.matmul, out=c), a, b, tag="f16_torch")
    c_torch = c.cpu()
    run_benchmark(lib.basic_gemm, a, b, out=c, tag="cutlass_basic_gemm")
    run_benchmark(lib.cute_example, a, b, out=c, tag="cute_example_gemm")
    c_cute = c.cpu()
    # print(c_torch, c_cute)
    print("-" * 80)
