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


template <class Op>
CUTLASS_DEVICE static float threadquad_reduce(float val, Op op) {
  float other = __shfl_xor_sync(0xffffffffu, val, 2, 32);
  val = op(val, other);
  other = __shfl_xor_sync(0xffffffffu, val, 1, 32);
  val = op(val, other);
  return val;
}

struct ThreadQuadSum {
  CUTLASS_DEVICE float operator()(float x, float y) const {
    return x + y;
  }
};

CUTLASS_DEVICE static float threadquad_reduce_sum(float val) {
  return threadquad_reduce(val, ThreadQuadSum{});
}

struct ThreadQuadMax {
  CUTLASS_DEVICE float operator()(float x, float y) const {
    return fmaxf(x, y);
  }
};

CUTLASS_DEVICE static float threadquad_reduce_max(float val) {
  return threadquad_reduce(val, ThreadQuadMax{});
}

template <class TensorType>
CUTLASS_DEVICE static auto make_acc_tensor_mn_view(TensorType& acc) {
  using namespace cute;
  auto acc_layout_col_major = make_layout(shape(acc));
  auto acc_shape_col_major = shape(acc_layout_col_major);
  auto acc_stride_col_major = stride(acc_layout_col_major);

  auto acc_layout_mn = make_layout(
      make_shape(
          make_shape(get<1>(get<0>(acc_shape_col_major)), get<1>(acc_shape_col_major)),
          make_shape(get<0>(get<0>(acc_shape_col_major)), get<2>(acc_shape_col_major))),
      make_stride(
          make_coord(get<1>(get<0>(acc_stride_col_major)), get<1>(acc_stride_col_major)),
          make_coord(get<0>(get<0>(acc_stride_col_major)), get<2>(acc_stride_col_major))));

  acc_layout_mn = composition(acc.layout(), acc_layout_mn);
  return make_tensor(acc.data(), acc_layout_mn);
}

template <class TensorAcc, class TensorRowSum>
CUTLASS_DEVICE static void normalize_softmax(TensorAcc& acc_O, TensorRowSum& row_sum) {
  using namespace cute;

  Tensor acc_O_mn = make_acc_tensor_mn_view(acc_O);
  for (int r = 0; r < size(row_sum); ++r) {
    row_sum(r) = threadquad_reduce_sum(row_sum(r));
    float row_val = row_sum(r);
    bool acc_O_mn_row_is_zero_or_nan = (row_val == 0.0f) || isnan(row_val);
    float scale = acc_O_mn_row_is_zero_or_nan ? 1.0f : __frcp_rn(row_val);
    for (int n = 0; n < size<1>(acc_O_mn); ++n) {
      acc_O_mn(r, n) = acc_O_mn(r, n) * scale;
    }
  }
}

template <
    bool IsFirstNBlock,
    bool InMaskSteps,
    class TensorMQ,
    class TensorMK,
    class ThrMMA,
    class TensorAccS,
    class TensorAccO,
    class TensorRowMax,
    class TensorRowSum,
    class CtaShape>
CUTLASS_DEVICE static void softmax_rescale_O(
    bool is_causal,
    TensorMQ const& mQ,
    TensorMK const& mK,
    int batch_idx,
    int head_idx,
    int m_block,
    int n_block,
    CtaShape const& cta_tiler,
    ThrMMA const& thr_mma,
    TensorAccS& acc_S,
    TensorAccO& acc_O,
    TensorRowMax& row_max,
    TensorRowSum& row_sum,
    float softmax_scale_log2) {
  using namespace cute;
  auto acc_S_mn = make_acc_tensor_mn_view(acc_S);
  auto acc_O_mn = make_acc_tensor_mn_view(acc_O);
  int const num_rows = int(size(row_max));
  int const num_cols = int(size(get<1>(shape(acc_S_mn))));

  auto process_row = [&](int r) {
    float const row_sum_prev = row_sum(r);
    float row_max_cur = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < num_cols; ++c) {
      row_max_cur = fmaxf(row_max_cur, acc_S_mn(r, c));
    }
    row_max_cur = threadquad_reduce_max(row_max_cur);

    float prev_minus_cur_exp = 0.0f;
    if constexpr (!IsFirstNBlock) {
      float const row_max_prev = row_max(r);
      row_max_cur = fmaxf(row_max_prev, row_max_cur);
      prev_minus_cur_exp = exp2f((row_max_prev - row_max_cur) * softmax_scale_log2);
      for (int c = 0; c < num_cols; ++c) {
        acc_O_mn(r, c) *= prev_minus_cur_exp;
      }
    }

    if (is_causal && row_max_cur == -std::numeric_limits<float>::infinity()) {
      row_max_cur = 0.0f;
    }

    float acc_S_row_sum = 0.0f;
    for (int c = 0; c < num_cols; ++c) {
      float const exp_val = exp2f((acc_S_mn(r, c) - row_max_cur) * softmax_scale_log2);
      acc_S_mn(r, c) = exp_val;
      acc_S_row_sum += exp_val;
    }

    if constexpr (!IsFirstNBlock) {
      acc_S_row_sum += row_sum_prev * prev_minus_cur_exp;
    }

    row_max(r) = row_max_cur;
    row_sum(r) = acc_S_row_sum;
  };

  if constexpr (InMaskSteps) {
    auto mcS = make_identity_tensor(
        make_shape(size<0>(mQ), size<1>(mQ), size<2>(mQ), size<1>(mK)));
    auto cS = local_tile(mcS(batch_idx, _, head_idx, _),
                         make_shape(get<0>(cta_tiler), get<1>(cta_tiler)),
                         make_coord(m_block, n_block));
    auto tScS = thr_mma.partition_C(cS);
    auto tScS_mn = make_acc_tensor_mn_view(tScS);
    int const seqlen_k = int(get<1>(shape(mK)));

    for (int r = 0; r < num_rows; ++r) {
      if (!is_causal) {
        for (int c = 0; c < num_cols; ++c) {
          int const key_idx = int(get<3>(tScS_mn(make_coord(0, c)))) + 1;
          if (seqlen_k < key_idx) {
            acc_S_mn(r, c) = -std::numeric_limits<float>::infinity();
          }
        }
      } else {
        int const col_idx_limit =
            std::min(int(get<1>(tScS_mn(make_coord(r, 0)))) + 1, seqlen_k);
        for (int c = 0; c < num_cols; ++c) {
          int const key_idx = int(get<3>(tScS_mn(make_coord(0, c)))) + 1;
          if (col_idx_limit < key_idx) {
            acc_S_mn(r, c) = -std::numeric_limits<float>::infinity();
          }
        }
      }
      process_row(r);
    }
  } else {
    for (int r = 0; r < num_rows; ++r) {
      process_row(r);
    }
  }
}

template <
    bool IsFirstNBlock,
    bool InMaskSteps,
    class T,
    class TensorMQ,
    class TensorMK,
    class CtaShape,
    class ThrMma,
    class TiledMma,
    class TensorAccO,
    class TensorRowMax,
    class TensorRowSum,
    class TiledCopyQKV,
    class TensortKgK,
    class TensortKsK,
    class TensortKVcKV,
    class TensortKVpKV,
    class TensortVgV,
    class TensortVsV,
    class TiledCopyQ,
    class TensortSsQ,
    class TensortSrQCopy,
    class TensortSrQ,
    class TiledCopyK,
    class TensortSsK,
    class TensortSrKCopy,
    class TensortSrK,
    class TiledCopyV,
    class TensortOsVt,
    class TensortOrVtCopy,
    class TensortOrVt>
CUTLASS_DEVICE static void compute_one_n_block(
    bool is_causal,
    TensorMQ const& mQ,
    TensorMK const& mK,
    int batch_idx,
    int head_idx,
    int m_block_idx,
    int n_block_idx,
    CtaShape const& cta_tiler,
    ThrMma& thr_mma,
    TiledMma const& tiled_mma,
    TensorAccO& acc_O,
    TensorRowMax& row_max,
    TensorRowSum& row_sum,
    float softmax_scale_log2,
    TiledCopyQKV const& gmem_tiled_copy_QKV,
    TensortKgK const& tKgK,
    TensortKsK& tKsK,
    TensortKVcKV const& tKVcKV,
    TensortKVpKV const& tKVpKV,
    TensortVgV const& tVgV,
    TensortVsV& tVsV,
    TiledCopyQ const& smem_tiled_copy_Q,
    TensortSsQ const& tSsQ,
    TensortSrQCopy& tSrQ_copy_view,
    TensortSrQ& tSrQ,
    TiledCopyK const& smem_tiled_copy_K,
    TensortSsK const& tSsK,
    TensortSrKCopy& tSrK_copy_view,
    TensortSrK& tSrK,
    TiledCopyV const& smem_tiled_copy_V,
    TensortOsVt const& tOsVt,
    TensortOrVtCopy& tOrVt_copy_view,
    TensortOrVt& tOrVt) {
  using namespace cute;
  using T_acc = float;

  auto acc_shape_S = partition_shape_C(tiled_mma, make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
  auto acc_S = make_tensor<T_acc>(acc_shape_S);
  clear(acc_S);

  cp_async_wait<0>();
  __syncthreads();

  if constexpr (IsFirstNBlock) {
    for (int n = 0; n < size<1>(tVsV); ++n) {
      if (elem_less(get<1>(tKVcKV(Int<0>{}, n, Int<0>{})), size<1>(mK))) {
        copy_if(gmem_tiled_copy_QKV,
                tKVpKV(_, n, _),
                tVgV(_, n, _, n_block_idx),
                tVsV(_, n, _));
      } else {
        clear(tVsV(_, n, _));
      }
    }
  } else {
    copy_if(gmem_tiled_copy_QKV,
            tKVpKV,
            tVgV(_, _, _, n_block_idx),
            tVsV);
  }

  cp_async_fence();

  copy(smem_tiled_copy_Q,
       tSsQ(_, _, 0),
       tSrQ_copy_view(_, _, 0));
  copy(smem_tiled_copy_K,
       tSsK(_, _, 0),
       tSrK_copy_view(_, _, 0));

  int const num_k_tiles = size<2>(tSsQ);
  for (int k = 0; k < num_k_tiles; ++k) {
    int const k_next = (k + 1) % num_k_tiles;
    copy(smem_tiled_copy_Q,
         tSsQ(_, _, k_next),
         tSrQ_copy_view(_, _, k_next));
    copy(smem_tiled_copy_K,
         tSsK(_, _, k_next),
         tSrK_copy_view(_, _, k_next));
    gemm(tiled_mma,
         acc_S,
         tSrQ(_, _, k),
         tSrK(_, _, k),
         acc_S);
  }

  cp_async_wait<0>();
  __syncthreads();

  if (n_block_idx > 0) {
    copy_if(gmem_tiled_copy_QKV,
            tKVpKV,
            tKgK(_, _, _, n_block_idx - 1),
            tKsK);
    cp_async_fence();
  }

  softmax_rescale_O<IsFirstNBlock, InMaskSteps>(
      is_causal,
      mQ,
      mK,
      batch_idx,
      head_idx,
      m_block_idx,
      n_block_idx,
      cta_tiler,
      thr_mma,
      acc_S,
      acc_O,
      row_max,
      row_sum,
      softmax_scale_log2);

  auto rP = make_fragment_like<T>(acc_S);
  copy(acc_S, rP);
  auto tOrS = thr_mma.retile_A(rP);

  copy(smem_tiled_copy_V,
       tOsVt(_, _, 0),
       tOrVt_copy_view(_, _, 0));

  int const num_v_tiles = size<2>(tOrS);
  for (int k = 0; k < num_v_tiles; ++k) {
    int const k_next = (k + 1) % num_v_tiles;
    copy(smem_tiled_copy_V,
         tOsVt(_, _, k_next),
         tOrVt_copy_view(_, _, k_next));
    gemm(tiled_mma,
         acc_O,
         tOrS(_, _, k),
         tOrVt(_, _, k),
         acc_O);
  }
}

template <class ProblemShape, class CtaTiler, class T, class QStride, class KStride, class VStride, class OStride, class QSmemLayout, class KVSmemLayout, class OSmemLayout, class TiledCopyQKV, class TiledCopyO, class S2RAtomQ, class S2RAtomK, class S2RAtomV, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void cute_kernel_device(ProblemShape prob_shape, CtaTiler cta_tiler, T const *Q, QStride dQ, T const *K, KStride dK, T const *V, VStride dV, T *out, OStride dO, float softmax_scale_log2, QSmemLayout sQ_layout, KVSmemLayout sKV_layout, OSmemLayout sO_layout, TiledCopyQKV gmem_tiled_copy_QKV,TiledCopyO gmem_tiled_copy_O, S2RAtomQ smem_copy_atom_Q, S2RAtomK smem_copy_atom_K, S2RAtomV smem_copy_atom_V, TiledMma mma) {
  using namespace cute;
  Tensor mQ = make_tensor(make_gmem_ptr(Q), select<0, 1, 3, 4>(prob_shape), dQ);
  Tensor mK = make_tensor(make_gmem_ptr(K), select<0, 2, 3, 4>(prob_shape), dK);
  Tensor mV = make_tensor(make_gmem_ptr(V), select<0, 2, 3, 4>(prob_shape), dV);
  Tensor mO = make_tensor(make_gmem_ptr(out), select<0, 1, 3, 4>(prob_shape), dO);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, blockIdx.y); // m_block, batch_size, num_head
  int n_block_max = ceil_div(size<1>(mK), get<1>(cta_tiler));
  int n_block = n_block_max - 1;
  // (m_block_size, head_dim)
  Tensor gQ = local_tile(mQ(get<1>(cta_coord), _, get<2>(cta_coord), _), select<0, 2>(cta_tiler), make_coord(get<0>(cta_coord), 0));
  // (n_block_size, head_dim, n_block)
  Tensor gK = local_tile(mK(get<1>(cta_coord), _, get<2>(cta_coord), _), select<1, 2>(cta_tiler), make_coord(_, 0));

  Tensor gV = local_tile(mV(get<1>(cta_coord), _, get<2>(cta_coord), _), select<1, 2>(cta_tiler), make_coord(_, 0));
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, QSmemLayout, KVSmemLayout>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sQ = make_tensor(make_smem_ptr(smem.sQ.begin()), sQ_layout);
  Tensor sK = make_tensor(make_smem_ptr(smem.sK.begin()), sKV_layout);
  Tensor sV = make_tensor(make_smem_ptr(smem.sV.begin()), sKV_layout);
  // (head_dim, n_block_size)
  auto sVt_layout = make_layout(
      make_shape(get<2>(cta_tiler), get<1>(cta_tiler)),
      make_stride(get<1>(cta_tiler), _1{}));
  Tensor sVt = composition(sV, sVt_layout);
  ThrCopy gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);
  // (CPY_Atom, CPY_M, CPY_K)
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  
  // (CPY_Atom, CPY_N, CPY_K, n_block)
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

  // (CPY_Atom, CPY_N, CPY_K, n_block)
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  /////////////////////////////////////////////////////////////////
  // Tile MMA compute thread partitions and allocate accumulators
  /////////////////////////////////////////////////////////////////
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  // partition_fragment_A
  Tensor tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ));
  Tensor tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK));
  Tensor tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt));
  auto acc_shape_O = partition_shape_C(mma, select<0, 2>(cta_tiler));
  Tensor acc_O = make_tensor<float>(acc_shape_O);
  clear(acc_O);

  /////////////////////////////////////////////////////////////////
  // Smem copy atom tiling
  /////////////////////////////////////////////////////////////////
  TiledCopy smem_tiled_copy_Q = make_tiled_copy_A(smem_copy_atom_Q, mma);
  TiledCopy smem_tiled_copy_K = make_tiled_copy_B(smem_copy_atom_K, mma);
  TiledCopy smem_tiled_copy_V = make_tiled_copy_B(smem_copy_atom_V, mma);
  ThrCopy smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(threadIdx.x);
  ThrCopy smem_thr_copy_K = smem_tiled_copy_K.get_slice(threadIdx.x);
  ThrCopy smem_thr_copy_V = smem_tiled_copy_V.get_slice(threadIdx.x);

  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
  Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);
  Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
  Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

  /////////////////////////////////////////////////////////////////
  // Predicate: Mark indices that need to copy when problem_shape isn't a multiple of tile_shape
  /////////////////////////////////////////////////////////////////
  // Construct identity layout for Q and KV
  Tensor mcQ = make_identity_tensor(shape(mQ));
  Tensor mcKV = make_identity_tensor(shape(mK));
  Tensor cQ = local_tile(mcQ(get<1>(cta_coord), _, get<2>(cta_coord), _), select<0, 2>(cta_tiler), make_coord(get<0>(cta_coord), 0));
  Tensor cKV = local_tile(mcKV(get<1>(cta_coord), _, get<2>(cta_coord), _), select<1, 2>(cta_tiler), make_coord(n_block, 0));
  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
  auto tQpQ_layout = make_layout(
      make_shape(
          get<1>(get<0>(shape(tQsQ))),
          size<1>(tQsQ),
          size<2>(tQsQ)),
      make_stride(size<2>(tQsQ), _0{}, _1{}));
  Tensor tQpQ = make_tensor<bool>(tQpQ_layout);
  auto tKVpKV_layout = make_layout(
      make_shape(
          get<1>(get<0>(shape(tKsK))),
          size<1>(tKsK),
          size<2>(tKsK)),
      make_stride(size<2>(tKsK), _0{}, _1{}));
  Tensor tKVpKV = make_tensor<bool>(tKVpKV_layout);
  for (int rest_v = 0; rest_v < size<0>(tQpQ); rest_v++) {
    for (int rest_k = 0; rest_k < size<2>(tQpQ); rest_k++) {
      tQpQ(rest_v, 0, rest_k) = elem_less(get<3>(tQcQ(make_coord(_0{}, rest_v), _0{}, rest_k)), size<3>(mQ));
    }
  }
  for (int rest_v = 0; rest_v < size<0>(tKVpKV); rest_v++) {
    for (int rest_k = 0; rest_k < size<2>(tKVpKV); rest_k++) {
      tKVpKV(rest_v, 0, rest_k) = elem_less(get<3>(tKVcKV(make_coord(_0{}, rest_v), _0{}, rest_k)), size<3>(mK));
    }
  }
  // Prefetch Prologue
  // Start async loads of the last mn-tile, where we take care of the mn residue
  for (int m = 0; m < size<1>(tQsQ); m++) {
    if (elem_less(get<1>(tQcQ(0, m, 0)), size<1>(mQ)))
      copy_if(gmem_tiled_copy_QKV, tQpQ(_, m, _), tQgQ(_, m, _), tQsQ(_, m, _));
    else
      clear(tQsQ(_, m, _));
  }

  for (int n = 0; n < size<1>(tKsK); n++) {
    if (elem_less(get<1>(tKVcKV(0, n, 0)), size<1>(mK)))
      copy_if(gmem_tiled_copy_QKV, tKVpKV(_, n, _), tKgK(_, n, _, n_block), tKsK(_, n, _));
    else
      clear(tKsK(_, n, _));
  }
  cp_async_fence();
  ///////////////////////////////////////////////////////////////////////////////
  // Softmax intermediate result: row_max and row_sum
  ///////////////////////////////////////////////////////////////////////////////
  // shape: (atom_v_m * rest_m)
  Tensor row_max = make_tensor<float>(make_layout(make_shape(get<0>(get<0>(shape(acc_O))) * get<1>(shape(acc_O)))));
  // shape: (atom_v_m * rest_m)
  Tensor row_sum = make_tensor<float>(make_layout(make_shape(get<0>(get<0>(shape(acc_O))) * get<1>(shape(acc_O)))));
  fill(row_max, -std::numeric_limits<float>::infinity());
  clear(row_sum);

  // Start processing of the first n-block.
  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not multiple of n_block_size.
  // We also need masking on S if it's causal, for the last ceil_div(m_block_size, n_block_size) blocks.
  // We will have at least 1 "masking" iteration.
  int mask_steps = 1;
  for (int n_tile = 0; n_tile < mask_steps; ++n_tile) {
    int n_block_cur = n_block_max - n_tile - 1;
      if (n_block_cur < 0) {
        continue;
      }
      compute_one_n_block<true, true, T>(
          false,
          mQ,
          mK,
          get<1>(cta_coord),
          get<2>(cta_coord),
          get<0>(cta_coord),
          n_block_cur,
          cta_tiler,
          thr_mma,
          mma,
          acc_O,
          row_max,
          row_sum,
          softmax_scale_log2,
          gmem_tiled_copy_QKV,
          tKgK,
          tKsK,
          tKVcKV,
          tKVpKV,
          tVgV,
          tVsV,
          smem_tiled_copy_Q,
          tSsQ,
          tSrQ_copy_view,
          tSrQ,
          smem_tiled_copy_K,
          tSsK,
          tSrK_copy_view,
          tSrK,
          smem_tiled_copy_V,
          tOsVt,
          tOrVt_copy_view,
          tOrVt);
    
  }

  // for (int n_tile = mask_steps; n_tile < n_block_max; ++n_tile) {
  //   int n_block_cur = n_block_max - n_tile - 1;
  //   if (n_block_cur < 0) {
  //     continue;
  //   }
  //   compute_one_n_block<false, false, T>(
  //       false,
  //       mQ,
  //       mK,
  //       get<1>(cta_coord),
  //       get<2>(cta_coord),
  //       get<0>(cta_coord),
  //       n_block_cur,
  //       cta_tiler,
  //       thr_mma,
  //       mma,
  //       acc_O,
  //       row_max,
  //       row_sum,
  //       softmax_scale_log2,
  //       gmem_tiled_copy_QKV,
  //       tKgK,
  //       tKsK,
  //       tKVcKV,
  //       tKVpKV,
  //       tVgV,
  //       tVsV,
  //       smem_tiled_copy_Q,
  //       tSsQ,
  //       tSrQ_copy_view,
  //       tSrQ,
  //       smem_tiled_copy_K,
  //       tSsK,
  //       tSrK_copy_view,
  //       tSrK,
  //       smem_tiled_copy_V,
  //       tOsVt,
  //       tOrVt_copy_view,
  //       tOrVt);
  // }
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
  auto num_threads = _128{};

  // TODO: make head_dim_padded and smem_k_block_size as template parameter.
  // const int head_dim_padded = (head_dim + 31) / 32 * 32; // padding head_dim to a multiple of 32 as k_block_size
  // auto smem_k_block_size = head_dim_padded % 64 == 0 ? _64{} : _32{};
  auto head_dim_padded = _128{};
  auto smem_k_block_size = _64{};
  auto cta_tiler = make_shape(m_block_size, n_block_size, head_dim_padded);

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
  int smem_size = int(sizeof(SMEMStorage));
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

  Copy_Atom<SM75_U32x4_LDSM_N, half_t> smem_copy_atom_Q;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> smem_copy_atom_K;
  Copy_Atom<SM75_U16x8_LDSM_T, half_t> smem_copy_atom_V;

  // dimGrid: (m_block, batch_size, num_head)
  dim3 dimGrid = (size(ceil_div(seqlen_q, m_block_size)), batch_size, num_head);
  dim3 dimBlock(size(tiled_mma));
  auto kernel_fptr = cute_kernel_device<decltype(prob_shape), decltype(cta_tiler), cute::half_t, decltype(dQ), decltype(dK), decltype(dV), decltype(dO), decltype(sQ_layout), decltype(sKV_layout), decltype(sO_layout), decltype(gmem_tiled_copy_QKV), decltype(gmem_tiled_copy_O), decltype(smem_copy_atom_Q), decltype(smem_copy_atom_K), decltype(smem_copy_atom_V), decltype(tiled_mma)>;
  cudaFuncSetAttribute(
      kernel_fptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);  
  float softmax_scale_log2 = 1;
  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(prob_shape, cta_tiler, q, dQ, k, dK, v, dV, out, dO, softmax_scale_log2, sQ_layout, sKV_layout, sO_layout, gmem_tiled_copy_QKV, gmem_tiled_copy_O, smem_copy_atom_Q, smem_copy_atom_K, smem_copy_atom_V, tiled_mma);
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