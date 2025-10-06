import time
import torch
from typing import Optional, Any


common_cuda_flags = ["-O3",
                     "--expt-relaxed-constexpr",
                     "-ftemplate-backtrace-limit=0",
                     "-Xcompiler=-Wconversion",
                     "-Xcompiler=-fno-strict-aliasing",
                     "--expt-extended-lambda",
                     "--use_fast_math",]
common_sycl_flags = ["-O3"]


def flush_cache():
    torch.sum(torch.randn(1024, 1024, 256,
              device=torch.accelerator.current_accelerator()))


def run_benchmark(
    perf_func: callable,
    *args: Any,
    out: Optional[torch.Tensor] = None,
    tag: str = "",
    warmup: int = 10,
    iters: int = 10,
    show_all: bool = False,
):
    if out is not None:
        for i in range(warmup):
            out.fill_(0)
            flush_cache()
            perf_func(*args, out)
    else:
        for i in range(warmup):
            flush_cache()
            _ = perf_func(*args)

    total_time = 0
    # iters
    if out is not None:
        for i in range(iters):
            out.fill_(0)
            flush_cache()
            torch.accelerator.synchronize()
            start = time.time()
            perf_func(*args, out)
            torch.accelerator.synchronize()
            end = time.time()
            total_time += (end - start)
    else:
        for i in range(iters):
            flush_cache()
            torch.accelerator.synchronize()
            start = time.time()
            out = perf_func(*args)
            torch.accelerator.synchronize()
            end = time.time()
            total_time += (end - start)
    total_time *= 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>13}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time
