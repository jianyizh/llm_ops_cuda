import time
import torch
from typing import Optional, Any


common_cuda_flags = ["-O3",
                     "-forward-unknown-to-host-compiler",
                     "--expt-relaxed-constexpr",
                     "-ftemplate-backtrace-limit=0",
                     "-Xcompiler=-Wconversion",
                     "-Xcompiler=-fno-strict-aliasing",
                     "--expt-extended-lambda",
                     "--use_fast_math",]
common_sycl_flags = ["-O3"]


def flush_cache():
    shape = (1024, 1024, 256)
    if torch.accelerator.current_accelerator().type == 'cuda':
        if "B200" in torch.cuda.get_device_name():
            shape = (1024, 1024, 1024)
    torch.sum(torch.randn(shape,
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

def get_mha_tflops(
    batch: int, num_heads_q: int, seq_len_qo: int, head_size_qk: int, num_heads_kv: int, seq_len_kv: int, head_size_vo: int, secs: float = 1.0):
    flops_qk = 2.0 * batch * num_heads_q * seq_len_qo * seq_len_kv * head_size_qk
    flops_pv = 2.0 * batch * num_heads_q * seq_len_qo * head_size_vo * seq_len_kv
    tflops = ((flops_qk + flops_pv) * 1e-12) / secs
    element_size_bytes = 2
    gbps_qk = batch * (
        element_size_bytes * num_heads_q * seq_len_qo * head_size_qk +
        element_size_bytes * num_heads_kv * seq_len_kv * head_size_qk
    )
    gbps_pv = (
        element_size_bytes * batch * num_heads_kv * seq_len_kv * head_size_vo +
        element_size_bytes * batch * num_heads_q *seq_len_qo * head_size_vo
    )
    gbps = ((gbps_qk + gbps_pv) * 1e-9) / secs


    return tflops, gbps
