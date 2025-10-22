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

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledStoreC, class R2SAtomC, class TiledMma,
          class Alpha, class Beta, class Swizzle>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void cute_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                                                                             TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
                                                                                             TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
                                                                                             TC *C, CStride dC, CSmemLayout sC_layout, TiledStoreC store_c, R2SAtomC r2s_atom_c, TiledMma mma,
                                                                                             Alpha alpha, Beta beta, Swizzle swizzle)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{}); // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == _3{}); // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto swizzle_coord = swizzle.get_tile_offset(swizzle.get_tiled_shape({get<0>(shape_MNK), get<1>(shape_MNK), get<2>(shape_MNK)}, {size<0>(cta_tiler), size<1>(cta_tiler), size<2>(cta_tiler)}, 1));
  auto cta_coord = make_coord(swizzle_coord.m(), swizzle_coord.n(), _);
  // auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  // gA = gmem_ptr[16b](0x7f90f4000000) o (_128,_64,K / BLK_K):(2048,_1,_64)

  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K,PIPE)
  // sA = smem_ptr[16b](0x7f155d000000) o ((_8,_16),((_8,_8),_1),(_1,_3)):((_8,_512),((_1,_64),_0),(_0,_8192))
  // we also test sA = smem_ptr[16b](0x7f7b8d000000) o (_128,_64,_3):(_64,_1,_8192)

  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);

  // Copy_Atom
  //   ThrID:        _1:_0
  //   ValLayoutSrc: (_1,_8):(_0,_1)
  //   ValLayoutDst: (_1,_8):(_0,_1)
  //   ValLayoutRef: (_1,_8):(_0,_1)
  //   ValueType:    16b
  // ThrCopy
  //   ThrIdx: 0
  // TiledCopy
  //   Tiler_MN:       (_16,_64)
  //   TiledLayout_TV: ((_8,_16),_8):((_128,_1),_16)

  // Tiler_MN is the layout of src/dst.
  // given thread id 10, value id 3, what does it copy?
  // 1. change coordinates between nature order and 1D order, read from right to left
  // thread layout (8, 16), thread id 10 = (2, 1), TV layout is ((2, 1), 3)
  // 2. index value is 2 * 128 + 1 + 3 * 16 = 305. in cute, defult is column major
  // in MN layout, it's 305 = 1 + 16*19 (1, 19)

  Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
  // different thread will have different address
  // tAgA = gmem_ptr[16b](0x7f33f4000000) o ((_8,_1),_8,_1,32):((_1,_0),32768,_0,_64)
  // each thread loop CPY times, each time copy 8 on M and 1 on K.

  Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)
  // tAsA = smem_ptr[16b](0x7f3865000000) o ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_1024,_0,(_0,_8192))
  // when sA = (_128,_64,_3):(_64,_1,_8192)
  // tAsA = smem_ptr[16b](0x7f76f9000000) o ((_8,_1),_8,_1,_3):((_1,_0),_1024,_0,_8192)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe)
  {
    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
    cp_async_fence(); // mark previous cp async in one group
    --k_tile_count;
    if (k_tile_count > 0)
    {
      ++k_tile_next;
    }
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  // ThrMMA
  //   Thr VMNK: (0,0,0,_0)
  // TiledMMA
  //   ThrLayoutVMNK:  (_32,_2,_2,_1):(_1,_32,_64,_0)
  //   PermutationMNK: (_32,_32,_16)
  // MMA_Atom (hardcode in traits)
  //   ThrID:      _32:_1
  //   Shape_MNK:  (_16,_8,_16)
  //   LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
  //   LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
  //   LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))

  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)
  // gmem_ptr[16b](0x7f3f88000000) o ((_2,_2),_4,_8):((_1,32768),131072,_16)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA,MMA_M,MMA_K)
  // ptr[16b](0x7f85abfffb60) o ((_2,_2,_2),_4,_4):((_1,_2,_4),_32,_8)
  // thr_mma.partition_A(sA(_, _, 0)) is smem_ptr[16b](0x7f7109000000) o ((_2,_2,_2),_4,_4):((_1,_512,_8),_2048,_16)

  Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA,MMA_N,MMA_K)
  // ptr[16b](0x7f8c0bfffb60) o ((_2,_2),_8,_4):((_1,_2),_16,_4)

  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)
  // ptr[16b](0x7fb03ffffb60) o ((_2,_2),_4,_8):((_1,_2),_4,_16)
  // will be ptr[32b] if accumulate type is f32

  CUTE_STATIC_ASSERT_V((shape(tCrC) == take<0, 3>(shape(tCgC)))); // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));         // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));         // MMA_N

  // Clear the accumulators
  clear(tCrC);

  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);

  // ThrCopy
  //   ThrIdx: 0
  // TiledCopy
  //   Tiler_MN:       (_32,_16)
  //   TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
  // Copy_Atom
  //   ThrID:        _32:_1
  //   ValLayoutSrc: (_32,_8):(_8,_1)
  //   ValLayoutDst: (_32,(_2,_4)):(_2,(_1,_64))
  //   ValLayoutRef: (_32,(_2,_4)):(_2,(_1,_64))
  //   ValueType:    16b

  Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY,MMA_M,MMA_K,PIPE)
  // smem_ptr[16b](0x7f0471000000) o ((_8,_1),_4,_4,(_1,_3)):((_1,_0),_2048,_128,(_0,_8192))
  // if sA is (128, 64, 3) smem_ptr[16b](0x7f8a61000000) o ((_8,_1),_4,_4,_3):((_1,_0),_2048,_16,_8192)

  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA); // (CPY,MMA_M,MMA_K)
  // ptr[16b](0x7f85abfffb60) o ((_8,_1),_4,_4):((_1,_0),_32,_8)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);  // (CPY,MMA_N,MMA_K)

  // Current pipe index in smem to read from
  int smem_pipe_read = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX - 1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
  // smem_ptr[16b](0x7fe771000000) o ((_8,_1),_4,_4):((_1,_0),_2048,_128)
  Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA); // MMA_K = 4
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1)
  {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX - 2>(); // this warp has finished K_PIPE_MAX - 2 cp async groups
    __syncthreads();                 // other warps will use this data, need sync
    // if (thread0()){
    //   for (int index = 0; index < 128*64; index+=8){
    //     print(index);
    //     print(" ");
    //     print((sA.data().get() + index)[0]);
    //     print("\n");
    //   }
    // }

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));

    // if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 16){
    //   for (int index = 0; index < 8; index++){
    //     print(tXsA_p(index));
    //     print("\n");
    //   }
    // }
    copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
  //           and explicit pipelines in shared memory.
  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
  //   Data is computed on registers(b_block).
  //
  //   This allows all copies and compute to overlap:
  //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
  //     Copy from smem->rmem can overlap with compute on rmem.
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX - 1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_, _, _, smem_pipe_read);
        tXsB_p = tXsB(_, _, _, smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
      copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
      copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0)
        {
          ++k_tile_next;
        }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    }
  }

  //
  // Epilogue
  //

  // axpby(alpha, tCrC, beta, tCgC);
  // if add build option "-U__CUDA_NO_HALF_OPERATORS__",
  //                     "-U__CUDA_NO_HALF_CONVERSIONS__",
  //                     "-U__CUDA_NO_HALF2_OPERATORS__",
  //                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
  // we can use axpby or tCrC_fp16(i) = tCrC(i) without cast 
  Tensor tCrC_fp16 = make_tensor_like<half_t>(tCrC);
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); i++)
  {
    tCrC_fp16(i) = static_cast<half_t>(tCrC(i));
  }

  Tensor sC = make_tensor(make_smem_ptr(smem.A.begin()), sC_layout);
  TiledCopy copy_c = make_tiled_copy_C(r2s_atom_c, mma);
  ThrCopy thr_copy_c = copy_c.get_slice(threadIdx.x);
  Tensor tSrC_r2s = thr_copy_c.retile_S(tCrC_fp16);
  Tensor tSsC_r2s = thr_copy_c.partition_D(sC);
  copy(copy_c, tSrC_r2s, tSsC_r2s);

  __syncthreads();

  auto s2g_thr_store = store_c.get_slice(threadIdx.x);

  Tensor tSsC = s2g_thr_store.partition_S(sC);
  Tensor tSgC = s2g_thr_store.partition_D(gC);
  copy(store_c, tSsC, tSgC);
}

cudaError_t cute_example_gemm(
    int M,
    int N,
    int K,
    float alpha,
    cute::half_t const *A,
    int lda,
    cute::half_t const *B,
    int ldb,
    float beta,
    cute::half_t *C,
    int ldc,
    torch::Device device)
{
  using namespace cute;
  using TA = cute::half_t;
  using TB = cute::half_t;
  using TC = cute::half_t;
  using TI = float;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(lda, _1{}); // (dM, dK) row major
  // note B is column major in pyorch, here ldb = K is not changed，
  // shape B is always (N, K) in cutlass gemm api
  auto dB = make_stride(ldb, _1{}); // (dN, dK) row major
  auto dC = make_stride(ldc, _1{}); // (dM, dN) row major

  // Define CTA tile sizes (static)
  // thread_block tile size
  auto bM = _128{};
  auto bN = _128{};
  auto bK = _64{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
  auto bP = _3{};                          // Pipeline

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                  Layout<Shape<_8, Shape<_8, _8>>,
                                         Stride<_8, Stride<_1, _64>>>{});

  // auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  // auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
  // shared memory layout without swizzle
  // tile_to_shape means? <_8, Shape<_8, _8>> tile repeat <bM/8, bK/<8,8>, bP> to get target shape (bM, bK, bP)
  // sA has shape ((8, bM/8)， ((8,8), bK/(8*8)), (1, bP)): ((_8,_512),((_1,_64),_0),(_0,_8192))
  // coalesce(Layout<Shape<_8, Shape<_8, _8>>,Stride<_8, Stride<_1, _64>>>()) = (_8,_8,_8):(_8,_1,_64), k major

  // coalesce(sA) =  (8, bM/8，8, 8, bK/64, bP):(_8,_512,_1,_64,bM * 64)
  // auto sA = tile_to_shape(Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}, make_shape(bM, bK, bP));
  // if we donnot use swizzle, this layout can be replaced.
  // tile_to_shape(Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}, make_shape(bM, bK, bP))
  // We can use any simpler layout with K major (8 element contiguous on k to becompatible with vectorized copy op)
  // TODO: need understand why different layout leads to different performance
  // ((_8,_16),((_8,_8),_1),(_1,_3)):((_8,_512),((_1,_64),_0),(_0,_8192))
  auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  // auto sA = tile_to_shape(Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}, make_shape(bM, bK, bP));
  // auto sA = Layout<Shape<_128, _64, _3>, Stride<_64, _1, Int<128*64>>>{};
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
  // auto sB = tile_to_shape(Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}, make_shape(bN, bK, bP));
  // auto sB = Layout<Shape<_128, _64, _3>, Stride<_64, _1, Int<128*64>>>{};
  auto sC = composition(Swizzle<3, 3, 3>{}, make_layout(make_shape(bM, bN), make_stride(bN, _1{})));
  // make_layout(make_shape(bM, bN), make_stride(bN, _1{})); // TODO: find a layout without bank conflict during store

  // Define the thread layouts (static)
  // 128 threads. 8 element per thread
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape<_1, _8>>{});                 // Val layout  1x8 k-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape<_1, _8>>{});                 // Val layout  1x8 n-major
  TiledCopy storeC = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, cute::half_t>{},
                                     Layout<Shape<_8, _16>, Stride<_16, _1>>{},
                                     Layout<Shape<_1, _8>>{});

  // 2x2x1 means do 2mma on M, 2mma on N, 1mma on K, so need 4 warps, so block size = size(mmaC) = 4 * 32
  // 32x32x16 is the actual mma size
  TiledMMA mmaC = make_tiled_mma(     // SM80_16x8x16_F16F16F16F16_TN{}, // dtype of dabc, D = A * B + C
      SM80_16x8x16_F32F16F16F32_TN{}, // f32 accumulator to reach same accuracy as pytorch
      Layout<Shape<_2, _2>>{},        // 2x2x1 MMA Atoms
      Tile<_32, _32, _16>{});         // 32x32x16 Tiled MMA for LDSM

  // Copy_Atom<DefaultCopy, half_t> s2r_atom_A;
  // Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_A;
  // Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_A;
  // Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_A;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;

  // Copy_Atom<DefaultCopy, half_t> s2r_atom_B;
  // Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_B;
  // Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_B;
  // Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_B;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;

  Copy_Atom<UniversalCopy<int>, half_t> r2s_atom_C;

  int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8> threadblock_swizzle;
  dim3 dimBlock(size(mmaC));
  // dim3 dimGrid(size(ceil_div(M, bM)),
  //              size(ceil_div(N, bN)));
  cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape({M, N, K}, {bM, bN, bK}, 1);
  dim3 dimGrid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);

  auto kernel_fptr = cute_gemm_device<
      decltype(prob_shape), decltype(cta_tiler),
      cute::half_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
      cute::half_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
      cute::half_t, decltype(dC), decltype(sC), decltype(storeC), decltype(r2s_atom_C), decltype(mmaC),
      decltype(alpha), decltype(beta), decltype(threadblock_swizzle)>;

  // Set L1 to be SMEM only
  cudaFuncSetAttribute(
      kernel_fptr,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  // cudaFuncSetAttribute(
  //     kernel_fptr,
  //     cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>(prob_shape, cta_tiler,
                                                        A, dA, sA, copyA, s2r_atom_A,
                                                        B, dB, sB, copyB, s2r_atom_B,
                                                        C, dC, sC, storeC, r2s_atom_C, mmaC,
                                                        alpha, beta, threadblock_swizzle);

  return cudaSuccess;
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

void cute_example(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c)
{
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  const int lda = K;
  const int ldb = K;
  const int ldc = N;
  auto result = cute_example_gemm(M, N, K, 1., reinterpret_cast<cute::half_t *>(a.data_ptr()), lda, reinterpret_cast<cute::half_t *>(b.data_ptr()), ldb, 0., reinterpret_cast<cute::half_t *>(c.data_ptr()), ldc, a.device());
  if (result != cudaSuccess)
  {
    std::cerr << "CUTLASS GEMM kernel failed: "
              << cudaGetErrorString(result) << std::endl;
  }
}
