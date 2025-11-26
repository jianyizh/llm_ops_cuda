#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.h>

#include <cutlass/gemm/dispatch_policy.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>

#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#ifndef TORCH_CURRENT_DEVICE
#define TORCH_CURRENT_DEVICE cutlass::arch::Sm80
#endif

// https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
/*
A flash attention v2 forward pass example for NVIDIA Ampere SM80 architecture using CUTE DSL.

- Matrix Q is BxSqxNxH, B is batch dimension, Sq is query sequence length, N is number of heads, H is head dimension
- Matrix K is BxSkxNxH, B is batch dimension, Sk is key sequence length, N is number of heads, H is head dimension
- Matrix V is BxSkxNxH, B is batch dimension, Sk is key sequence length, N is number of heads, H is head dimension
- Matrix O is BxSqxNxH, B is batch dimension, Sq is query sequence length, N is number of heads, H is head dimension

This kernel supports the following features:
    - Utilizes CpAsync for efficient memory operations
    - Utilizes Ampere's tensor core for matrix multiply-accumulate (MMA) operations
    - Utilizes register pipeline to overlap shared memory-to-register transfers with computations.
    - Leverages DSL to implement an integrated online softmax fusion pattern.

This kernel works as follows:
1. Load Q and K matrices from global memory (GMEM) to shared memory (SMEM) using CpAsync operations.
2. Perform matrix multiply-accumulate (MMA) operations using tensor core instructions to compute intermediate result S.
3. Apply padding mask or causal mask to S during initial iterations.
4. Apply online softmax to S and rescale O using results from previous iteration.
5. Load V matrices and perform matrix multiply-accumulate (MMA) operations to compute final result O.
6. Normalize O after all iterations complete and store result back to global memory (GMEM).
*/

template <class Dtype,
          class sQ_layout,
          class sKV_layout>
struct SharedStorage
{
  alignas(1024) cute::ArrayEngine<Dtype, cute::cosize_v<sQ_layout>> sQ;
  alignas(1024) cute::ArrayEngine<Dtype, cute::cosize_v<sKV_layout>> sK;
  alignas(1024) cute::ArrayEngine<Dtype, cute::cosize_v<sKV_layout>> sV;
};

template <class ProblemShape,  class T, class QStride, class KStride, class VStride, class OStride, class QSmemLayout, class KVSmemLayout, class OSmemLayout, class TiledCopyQKV, class TiledCopyO, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void cute_kernel_device(ProblemShape prob_shape, T const *Q, QStride dQ, T const *K, KStride dK, T const *V, VStride dV, T *out, OStride dO, float softmax_scale_log2, QSmemLayout sQ_layout, KVSmemLayout sKV_layout, OSmemLayout sO_layout, TiledCopyQKV gmem_tiled_copy_QKV,TiledCopyO gmem_tiled_copy_O, TiledMma mma) {
  using namespace cute;
  Tensor mQ = make_tensor(make_gmem_ptr(Q), select<0, 1, 3, 4>(prob_shape), dQ);
  Tensor mK = make_tensor(make_gmem_ptr(K), select<0, 2, 3, 4>(prob_shape), dK);
  Tensor mV = make_tensor(make_gmem_ptr(V), select<0, 2, 3, 4>(prob_shape), dV);
  Tensor mO = make_tensor(make_gmem_ptr(out), select<0, 1, 3, 4>(prob_shape), dO);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, blockIdx.y); // m_block, batch_size, num_head

}

cudaError_t cute_example_mha(
    cute::half_t const *q,
    cute::half_t const *k,
    cute::half_t const *v,
    cute::half_t *out,
    const int batch_size,
    const int seqlen_q,
    const int num_head,
    const int head_dim,
    const int seqlen_k,
    torch::Device device)
{
  using namespace cute;
  using T = cute::half_t;
  using T_acc = float;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto prob_shape = make_shape(batch_size, seqlen_q, seqlen_k, num_head, head_dim);
  auto dQ = make_stride(seqlen_q * num_head * head_dim, num_head * head_dim, head_dim, _1{});
  auto dK = make_stride(seqlen_k * num_head * head_dim, num_head * head_dim, head_dim, _1{});
  auto dV = make_stride(seqlen_k * num_head * head_dim, num_head * head_dim, head_dim, _1{});
  auto dO = make_stride(seqlen_q * num_head * head_dim, num_head * head_dim, head_dim, _1{});
  auto m_block_size = _128{};
  auto n_block_size = _64{};
  auto cta_tiler = make_shape(m_block_size, n_block_size);
  auto num_threads = _128{};

  // TODO: make head_dim_padded and smem_k_block_size as template parameter.
  // const int head_dim_padded = (head_dim + 31) / 32 * 32; // padding head_dim to a multiple of 32 as k_block_size
  // auto smem_k_block_size = head_dim_padded % 64 == 0 ? _64{} : _32{};
  auto head_dim_padded = _128{};
  auto smem_k_block_size = _64{};
  const int swizzle_bits = 3; // smem_k_block_size == _64{} ? 3 : 2;

  /*
  const int smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 2) * 2; // Q/K/V double buffering
  int maxSharedMem = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device.index());
  cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  if (smem_usage > maxSharedMem) {
    std::cerr << "CUTLASS mha kernel requires " << smem_usage <<" shared memory." << std::endl;
  }
  if (m_block_size * 2) % num_threads != 0 {
    std::cerr << "(m_block_size * 2) % num_threads != 0" << std::endl;
  }*/

  auto sQ_layout_atom = composition(
      Swizzle<swizzle_bits, 3, 3>{},
      make_layout(make_shape(_8{}, smem_k_block_size), make_stride(smem_k_block_size, _1{})));
  auto sQ_layout = tile_to_shape(sQ_layout_atom, make_shape(m_block_size, head_dim_padded));
  auto sKV_layout_atom = sQ_layout_atom;
  auto sKV_layout = tile_to_shape(sKV_layout_atom, make_shape(n_block_size, head_dim_padded));
  auto sO_layout = sQ_layout;
  using SMEMStorage = SharedStorage<half_t, decltype(sQ_layout), decltype(sKV_layout)>;

  auto atom_async_copy = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{}; // TODO: test cache global mode
  auto atom_universal_copy = Copy_Atom<UniversalCopy<uint128_t>, cute::half_t>{};
  // tQKV_layout: thread layout for QKV load
  auto tQKV_shape_dim_1 = _8{};                           // sQ_layout_atom.shape[1] / 8
  auto tQKV_layout = make_layout(make_shape(_16{}, _8{}), // (self._num_threads / tQKV_shape_dim_1, tQKV_shape_dim_1)
                                 make_stride(_8{}, _1{}));
  // tO_layout: thread layout for O store
  auto tO_layout = tQKV_layout;
  // Value layouts for copies
  auto vQKV_layout = make_layout(make_shape(_1{}, _8{}));
  auto vO_layout = vQKV_layout;
  // gmem_tiled_copy_QKV: tiled copy for QKV load
  TiledCopy gmem_tiled_copy_QKV = make_tiled_copy(
      atom_async_copy, tQKV_layout, vQKV_layout);
  // gmem_tiled_copy_O: tiled copy for O store
  TiledCopy gmem_tiled_copy_O = make_tiled_copy(
      atom_universal_copy, tO_layout, vO_layout);
  ///////////////////////////////////////////////////////////////////////////////
  // Tiled mma
  ///////////////////////////////////////////////////////////////////////////////
  TiledMMA tiled_mma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<_4, _1, _1>>{},
      Tile<_64, _16, _16>{});
  // grid_dim: (m_block, batch_size, num_head)
  dim3 grid_dim = (size(ceil_div(seqlen_q, m_block_size)), batch_size, num_head);
  dim3 dimBlock(size(tiled_mma));
  auto kernel_fptr = cute_kernel_device<>;
  cudaFuncSetAttribute(
      kernel_fptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);  
  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>();
  return cudaSuccess;
}

void cute_example(torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &out)
{
  const int batch_size = q.size(0);
  const int seqlen_q = q.size(2);
  const int num_head = q.size(1);
  const int head_dim = q.size(3);
  const int seqlen_k = k.size(2);

  // TODO: check dtype is half or bfloat16
  // TODO: Check if head dimension is a multiple of 8

  auto result = cute_example_mha(reinterpret_cast<cute::half_t *>(q.data_ptr()),
                                 reinterpret_cast<cute::half_t *>(k.data_ptr()),
                                 reinterpret_cast<cute::half_t *>(v.data_ptr()),
                                 reinterpret_cast<cute::half_t *>(out.data_ptr()),
                                 batch_size,
                                 seqlen_q,
                                 num_head,
                                 head_dim,
                                 seqlen_k,
                                 q.device());
  if (result != cudaSuccess)
  {
    std::cerr << "CUTLASS mha kernel failed: "
              << cudaGetErrorString(result) << std::endl;
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  TORCH_BINDING_COMMON_EXTENSION(cute_example)
}