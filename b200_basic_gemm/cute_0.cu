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

// Cutlass includes
#include <cutlass/half.h> // F16 data type
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe includes
#include <cute/tensor.hpp>                     // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>          // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>  // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>  // TMEM allocator for SM100

// This GEMM kernel performs the following steps:
// 1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) for one MmaTile
//    using auto-vectorizing copy operations.
// 2. Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
// 3. Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
// 4. Read C matrix from global memory (GMEM) to register (RMEM).
// 5. Apply alpha and beta scaling to the MMA accumulator and C matrix.
// 6. Store D matrix from registers (RMEM) to global memory (GMEM).
//
// SM100 tcgen05.mma instructions operate as follows:
// - Read matrix A from SMEM or TMEM
// - Read matrix B from SMEM
// - Write accumulator to TMEM
// The accumulator in TMEM must then be loaded to registers before writing back to GMEM.
//
// The tcgen05.mma instruction requires an Instruction Descriptor that encodes A, B, and Accumulator types
//   and the MMA's M and N dimensions.
// The A and B matrices that are read from SMEM need to be provided to MMA instructions as SMEM Descriptors.
//   These are the A and B fragments of the tcgen05.mma in CuTe terminology.
// CuTe provides these descriptors transparently in the instruction and fragments, shown in this tutorial.
//
// The MMA details:
// We use the tcgen05.mma.f16 instruction (F16xF16 = F32) that performs a 128x256x16 MMA
// operation. F32 accumulator type is chosen since both C and D matrices use F32.
// This example uses F16xF16 = F32 MMA where:
// TypeA = cutlass::half_t;  // MMA A Data Type
// TypeB = cutlass::half_t;  // MMA B Data Type
// TypeC = float;            // MMA C Data Type
// TypeD = float;            // MMA D Data Type
// TypeAccumulator = float;  // Both TypeC and TypeD are float, so we use float accumulator type

// The shared memory buffers for A and B matrices.
template <class TypeA,       // Tensor A data type
          class TypeB,       // Tensor B data type
          class ASmemLayout, // (MmaA, NumMma_M, NumMma_K, ...)
          class BSmemLayout> // (MmaB, NumMma_N, NumMma_K, ...)
struct SharedStorage
{

  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t mma_barrier; // Barrier to track MMA computation on SMEM

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() { return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{}); }
};

// The device kernel
template <class SharedStorage,
          class ATensor, class BTensor, class CTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class Alpha, class Beta>
__global__ static void
gemm_device(ATensor mA,                     // (Gemm_M, Gemm_K)
            BTensor mB,                     // (Gemm_N, Gemm_K)
            CTensor mC,                     // (Gemm_M, Gemm_N)
            MmaTiler_MNK mma_tiler,         // <MmaTile_M, MmaTile_N, MmaTile_K>
            TiledMMA tiled_mma,             // <    Mma_M,     Mma_N,     Mma_K>
            ClusterShape_MNK cluster_shape, // (ClusterM, ClusterN, ClusterK)
            Alpha alpha, Beta beta)
{
  using namespace cute;
  // Step 1: The Prologue.

  // The CTA layout within the Cluster: (V,M,N,K) -> CTA idx
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));

  // Construct the MMA grid coordinate from the CTA grid coordinate
  auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                                   blockIdx.x / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
                                   blockIdx.y,                                //    MMA-N coordinate
                                   _);                                        //    MMA-K coordinate

  // Partition the GMEM tensors with the mma_tiler and mma_coord to get the slices processed
  //   by this mma tile.
  // CuTe provides local_tile partitioning function. local_tile accepts 4 parameters:
  //   * Tensor to partition
  //   * Tiler to use for partitioning
  //   * Coordinate to use for slicing the partitioned tensor
  //   * Projection to ignore unwanted modes of the Tiler and Coordinate
  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);
  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{}); // (MmaTile_M, MmaTile_K, Tiles_K)
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{}); // (MmaTile_N, MmaTile_K, Tiles_K)
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{}); // (MmaTile_M, MmaTile_N)

  // if (thread0()) {
  //   print("mA:\t"); print(mA); print("\n");   // mA:   gmem_ptr[16b](GMEM_ADDR_A) o (512,256):(256,_1)
  //   print("mB:\t"); print(mB); print("\n");   // mB:   gmem_ptr[16b](GMEM_ADDR_B) o (1024,256):(256,_1)
  //   print("mC:\t"); print(mC); print("\n");   // mC:   gmem_ptr[32b](GMEM_ADDR_C) o (512,1024):(1024,_1)
  //   print("mD:\t"); print(mD); print("\n");   // mD:   gmem_ptr[32b](GMEM_ADDR_D) o (512,1024):(1024,_1)

  //   print("gA:\t"); print(gA); print("\n");   // gA:   gmem_ptr[16b](GMEM_ADDR_A + offset_for_mma_tile) o (_128,_64,4):(256,_1,_64)
  //   print("gB:\t"); print(gB); print("\n");   // gB:   gmem_ptr[16b](GMEM_ADDR_B + offset_for_mma_tile) o (_256,_64,4):(_1,256,16384)
  //   print("gC:\t"); print(gC); print("\n");   // gC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile) o (_128,_256):(256,_1)
  //   print("gD:\t"); print(gD); print("\n");   // gD:   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile) o (_128,_256):(256,_1)
  // } __syncthreads();

  // The SMEM tensors

  // Allocate SMEM
  extern __shared__ char shared_memory[];
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

  // Represent the SMEM buffers for A and B
  Tensor tCsA = shared_storage.tensor_sA(); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCsB = shared_storage.tensor_sB(); // (MmaB, NumMma_M, NumMma_K, Tiles_K)

  //
  // Mma partitioning for A and B
  //
  // Note: Partitioned tensors use tXgY naming convention:
  //  tXgY -> The partitioning pattern tX applied to tensor gY

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v); // Use Peer CTA coordinate
  Tensor tCgA = cta_mma.partition_A(gA);       // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCgB = cta_mma.partition_B(gB);       // (MmaB, NumMma_N, NumMma_K, Tiles_K)
  Tensor tCgC = cta_mma.partition_C(gC);       // (MmaC, NumMma_M, NumMma_N)

  // if (thread0()) {
  //   print("tCgA:\t"); print(tCgA); print("\n");  // tCgA:   gmem_ptr[16b](GMEM_ADDR_A + offset_for_mma_tile + offset_for_mma) o ((_128,_16),_1,_4,4):((256,_1),_0,_16,_64)
  //   print("tCgB:\t"); print(tCgB); print("\n");  // tCgB:   gmem_ptr[16b](GMEM_ADDR_B + offset_for_mma_tile + offset_for_mma) o ((_256,_16),_1,_4,4):((_1,256),_0,4096,16384)
  //   print("tCgC:\t"); print(tCgC); print("\n");  // tCgC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((256,_1),_0,_0)
  //   print("tCgD:\t"); print(tCgD); print("\n");  // tCgD:   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((256,_1),_0,_0)
  // } __syncthreads();

  // MMA Fragment Allocation
  // We allocate "fragments" which are SMEM descriptors that serve as inputs to cute::gemm operations.
  // For tcgen05.mma operations:
  // - Matrices A and B are sourced from SMEM
  // - tCrA and tCrB provide descriptor views of tCsA and tCsB respectively
  // - The first mode of each descriptor represents the SMEM for a single MMA operation
  Tensor tCrA = cta_mma.make_fragment_A(tCsA); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCrB = cta_mma.make_fragment_B(tCsB); // (MmaB, NumMma_M, NumMma_K, Tiles_K)

  // TMEM Allocation
  // On SM100 architecture, accumulators are stored exclusively in tensor memory (TMEM).
  // ThrMma's make_fragment_C() creates a TMEM tensor with the appropriate layout for the accumulator.
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC); // (MmaC, NumMma_M, NumMma_N)

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp)
  {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
  }
  __syncthreads(); // Wait for all threads until warp0 allocates TMEM
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  // if (thread0()) {
  //   print("tCsA:\t"); print(tCsA); print("\n");     // tCsA:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_A) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
  //   print("tCsB:\t"); print(tCsB); print("\n");     // tCsB:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_256,_16),_1,_4):((_64,_1),_0,_16)
  //   print("tCrA:\t"); print(tCrA); print("\n");     // tCrA:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
  //   print("tCrB:\t"); print(tCrB); print("\n");     // tCrB:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
  //   print("tCtAcc:\t"); print(tCtAcc); print("\n"); // tCtAcc: tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
  // } __syncthreads();

  // Barrier Initialization
  // Barriers in SMEM initialized by a single thread.
  if (elect_one_warp && elect_one_thr)
  {
    cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ 1);
  }
  int mma_barrier_phase_bit = 0; // Each barrier has an associated phase_bit.
  __syncthreads();               // Make sure all threads observe barrier initialization.

  // Step 2: The Mainloop.

  // Set mma accumulate option to zero so that the first MMA instruction will clear the TMEM accumulator.
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // Execute a MmaTile_M x MmaTile_N x GEMM_K GEMM
  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
  {
    // Step 2a: Load A and B tiles

    // Using auto-vectorized copy operation:
    // - Utilizes 128 threads for parallel data transfer
    // - Copy operations are distributed efficiently across all threads
    // - CuTe can automatically determine optimal vector width
    cooperative_copy<128>(threadIdx.x, tCgA(_, _, _, k_tile), tCsA); // Load MmaTile_M x MmaTile_K A tile
    cooperative_copy<128>(threadIdx.x, tCgB(_, _, _, k_tile), tCsB); // Load MmaTile_N x MmaTile_K B tile

    // Step 2b: Execute the MMAs for this tile

    // Wait for loads to SMEM to complete with __syncthreads()
    __syncthreads();

    // tcgen05.mma instructions require single-thread execution:
    // - Only one warp performs the MMA-related loop operations
    // - CuTe operations internally manage the single-thread execution of tcgen05.mma and tcgen05.cp
    // - No explicit elect_one_sync region is needed from the user
    if (elect_one_warp)
    {
      // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block)
      {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // Ensure MMAs are completed, only then we can reuse the A and B SMEM.
      cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
    }
    // Wait MMAs to complete to avoid overwriting the A and B SMEM.
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  // Step 3: The Epilogue.

  // Create the tiled copy operation for the accumulator (TMEM -> RMEM)
  TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDgC = thr_t2r_copy.partition_D(tCgC); // (CpyD, NumCpy_M, NumCpy_N)
  Tensor tDrC = make_fragment_like(tDgC);       // (CpyD, NumCpy_M, NumCpy_N)
  // Load C tensor GMEM -> RMEM
  // copy(tDgC, tDrC);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc); // (CpyS, NumCpy_M, NumCpy_N)
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgC)); // (CpyD, NumCpy_M, NumCpy_N)
  // Load TMEM -> RMEM
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  Tensor tDrAcc_fp16 = make_tensor_like<half_t>(tDrAcc);
  CUTE_UNROLL
  for (int i = 0; i < size(tDrAcc); i++)
  {
    tDrAcc_fp16(i) = static_cast<half_t>(tDrAcc(i));
  }
  // Store RMEM -> GMEM
  copy(tDrAcc, tDgC);

  __syncthreads();

  // Release the right to allocate before deallocations so that the next CTA can rasterize
  // Then deallocate TMEM
  if (elect_one_warp)
  {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
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

  // Define TN strides (mixed)
  auto dA = make_stride(lda, _1{}); // (dM, dK) row major
  // note B is column major in pyorch, here ldb = K is not changed，
  // shape B is always (N, K) in cutlass gemm api
  auto dB = make_stride(ldb, _1{}); // (dN, dK) row major
  auto dC = make_stride(ldc, _1{}); // (dM, dN) row major

  Layout layout_A = make_layout(make_shape(M, K), dA);
  Layout layout_B = make_layout(make_shape(N, K), dB);
  Layout layout_C = make_layout(make_shape(M, N), dC);

  // Represent the full tensors in global memory
  Tensor mA = make_tensor(make_gmem_ptr(A), layout_A); // (Gemm_M, Gemm_K)
  Tensor mB = make_tensor(make_gmem_ptr(B), layout_B); // (Gemm_N, Gemm_K)
  Tensor mC = make_tensor(make_gmem_ptr(C), layout_C); // (Gemm_M, Gemm_N)

  ////////////////////////////////////////////////////////////
  //
  // Initialize the GEMM kernel parameters
  //
  ////////////////////////////////////////////////////////////

  // Create TiledMma. make_tiled_mma takes the target instructions and an (optional) instruction layout as parameters to create a
  // larger TiledMma from the given mma instruction.
  // See cute/arch/mma_sm100_umma.hpp for all tcgen05.mma instructions
  // SS and TS means the A and B matrix types are sourced from SMEM or TMEM respectively.
  TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, TI,                         // Mma's A, B, and Accumulator types
                                                           128, 256,                           // Mma M and N dimensions
                                                           UMMA::Major::K, UMMA::Major::K>{}); // A and B layouts

  // We can also print and inspect the tiled_mma
  // print(tiled_mma);
  // TiledMMA
  //   ThrLayoutVMNK:  (_1,_1,_1,_1):(_0,_0,_0,_0)
  //   PermutationMNK: (_,_,_)
  // MMA_Atom
  //   ThrID:      _1:_0
  //   Shape_MNK:  (_128,_256,_16)                      // MmaM, MmaN, MmaK instruction size
  //   LayoutA_TV: (_1,(_128,_16)):(_0,(_1,_128))       // TV -> MmaCoordinate mapping for A matrix
  //   LayoutB_TV: (_1,(_256,_16)):(_0,(_1,_256))       // TV -> MmaCoordinate mapping for B matrix
  //   LayoutC_TV: (_1,(_128,_256)):(_0,(_1,_128))      // TV -> MmaCoordinate mapping for C matrix

  // Define MMA tiler sizes (static)
  auto bM = tile_size<0>(tiled_mma);            // MMA Tile M. We'll use 1 MMAs per MMA Tile M.
  auto bN = tile_size<1>(tiled_mma);            // MMA Tile N. We'll use 1 MMAs per MMA Tile M.
  auto bK = tile_size<2>(tiled_mma) * Int<4>{}; // MMA Tile K. We'll use 4 MMAs per MMA Tile K. For 16b types, tcgen05.mma has K16.
  auto mma_tiler = make_shape(bM, bN, bK);      // (MMA_M, MMA_N, MMA_K)

  // In SM90,  the MMAs are CTA-local and perform thread-level partitioning.
  // In SM100, the MMAs are Cluster-local and perform CTA-level partitioning.
  // Thus, SM90 uses a cta_tiler to extract portions of the Problem for the CTA
  //  and SM100 uses a mma_tiler to extract portions of the Problem for the MMA.
  //  The MMA's partitioning then yields the CTA-local work.

  if (not evenly_divides(shape(mma_tiler), tile_shape(tiled_mma)))
  {
    std::cerr << "The MMA Shape should evenly divide the MMA Tiler." << std::endl;
    return cudaErrorUnknown;
  }

  if (not evenly_divides(make_shape(M, N, K), mma_tiler))
  {
    std::cerr << "OOB accesses are not supported. MmaTiler_MNK should evenly divide ProblemShape_MNK." << std::endl;
    return cudaErrorUnknown;
  }

  //
  // Determine the SMEM layouts:
  //

  //  * SMEM layouts for A and B must match the post-partitioned (CTA-local) shapes expected by the MMA instructions.
  //  * CuTe provides partition_shape_[A|B] functions to determine the post-partitioned shape.
  //    These functions take the TiledMma, and the MMA Tile Shape as inputs and returns a shape that is at least rank-3
  //    where the first mode has the same shape as the MMA instruction, 2nd and 3rd mode expresses the number of time
  //    MMA instr is repeated in M/N mode and K mode of MMA tile, respectively.
  //  * Note that SMEM layouts are needed to determine SMEM allocation for kernel launch.

  // Pre-partitioned Tile Shape (MmaTile_M, MmaTile_K) to post-partitioned (MmaA, NumMma_M, NumMma_K)
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_K) to post-partitioned (MmaB, NumMma_N, NumMma_K)
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  // Print and inspect mma_shape_A, and mma_shape_B for this example.
  // print("mma_shape_A:\t"); print(mma_shape_A); print("\n");  // mma_shape_A:  ((_128,_16),_1,_4)
  // print("mma_shape_B:\t"); print(mma_shape_B); print("\n");  // mma_shape_B:  ((_256,_16),_1,_4)

  // A and B tensors are swizzled in SMEM to improve MMA performance.
  //  * However, expressing swizzled layouts is very hard.
  //  * CuTe provides tile_to_mma_shape functions for SM100 to create swizzled layouts for post-partitioned Mma Shapes
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<half_t>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<half_t>{}, mma_shape_B);

  // Print and inspect sA_layout and sB_layout for this example.
  // print("sA_layout:\t"); print(sA_layout); print("\n");      // sA_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
  // print("sB_layout:\t"); print(sB_layout); print("\n");      // sB_layout:   Sw<3,4,3> o smem_ptr[16b](unset) o ((_256,_16),_1,_4):((_64,_1),_0,_16)

  // Now we can find the SMEM allocation size
  using SMEMStorage = SharedStorage<half_t, half_t, decltype(sA_layout), decltype(sB_layout)>;

  // The cluster shape and layout
  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  ////////////////////////////////////////////////////////////
  //
  // Launch GEMM kernel
  //
  ////////////////////////////////////////////////////////////

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  dim3 dimGrid(size(ceil_div(M, bM * size<1>(cluster_layout_vmnk))) * dimCluster.x,
               size(ceil_div(N, bN * size<2>(cluster_layout_vmnk))) * dimCluster.y);
  int smemBytes = sizeof(SMEMStorage);

  auto *kernel_ptr = &gemm_device<SMEMStorage,
                                  decltype(mA), decltype(mB), decltype(mC),
                                  decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                                  float, float>;

  // Set kernel attributes (set SMEM)
  cudaFuncSetAttribute(kernel_ptr,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smemBytes);

  // printf("Grid launched: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  // printf("Cluster launched: %d, %d, %d\n", dimCluster.x, dimCluster.y, dimCluster.z);

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const *)kernel_ptr,
                                                             mA, mB, mC,
                                                             mma_tiler, tiled_mma, cluster_shape,
                                                             alpha, beta);

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
