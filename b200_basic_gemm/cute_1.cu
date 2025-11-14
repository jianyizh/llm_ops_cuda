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

// 1. Introduce ClusterShape for coordinated execution across thread blocks
// 2. Introduce TMA multicast
// 3. Enhanced TMA <-> MMA synchronization for cluster-wide operations

// This GEMM kernel performs the following steps:
// 1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) for one MmaTile
//    using auto-vectorizing copy operations.
// 2. Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
// 3. Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
// 4. Read C matrix from global memory (GMEM) to register (RMEM).
// 5. Apply alpha and beta scaling to the MMA accumulator and C matrix.
// 6. Store D matrix from registers (RMEM) to global memory (GMEM).
//
// This GEMM kernel will perform the following steps:
// 1. Load A and B matrices from GMEM to SMEM using Multicasted TMA load operations.
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
  alignas(16) cute::uint64_t tma_barrier; // Barrier to track TMA data transfers to SMEM

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() { return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{}); }
};

// The device kernel
template <class SharedStorage,
          class ATensor, class BTensor, class CTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB,
          class Alpha, class Beta>
__global__ static void
gemm_device_1(ATensor mA,                     // (Gemm_M, Gemm_K)
              BTensor mB,                     // (Gemm_N, Gemm_K)
              CTensor mC,                     // (Gemm_M, Gemm_N)
              MmaTiler_MNK mma_tiler,         // <MmaTile_M, MmaTile_N, MmaTile_K>
              TiledMMA tiled_mma,             // <    Mma_M,     Mma_N,     Mma_K>
              ClusterShape_MNK cluster_shape, // (ClusterM, ClusterN, ClusterK)
              CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
              CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
              Alpha alpha, Beta beta)
{
  using namespace cute;
  // Step 1: The Prologue.

  // The CTA layout within the Cluster: (V,M,N,K) -> CTA idx
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  //((_1),_4,_4,_1):((_0),_1,_4,_0)
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
  //   print("mA:\t"); print(mA); print("\n");   // mA:   gmem_ptr[16b](GMEM_ADDR_A) o (16384,8192):(8192,_1)
  //   print("mB:\t"); print(mB); print("\n");   // mB:   gmem_ptr[16b](GMEM_ADDR_B) o (8192,8192):(8192,_1)
  //   print("mC:\t"); print(mC); print("\n");   // mC:   gmem_ptr[32b](GMEM_ADDR_C) o (16384,8192):(8192,_1)

  //   print("gA:\t"); print(gA); print("\n");   // gA:   gmem_ptr[16b](GMEM_ADDR_A + offset_for_mma_tile) o (_128,_64,128):(8192,_1,_64)
  //   print("gB:\t"); print(gB); print("\n");   // gB:   gmem_ptr[16b](GMEM_ADDR_B + offset_for_mma_tile) o (_256,_64,128):(8192,_1,_64)
  //   print("gC:\t"); print(gC); print("\n");   // gC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile) o (_128,_256):(8192,_1)
  // } __syncthreads();

  // The SMEM tensors

  // Allocate SMEM
  extern __shared__ char shared_memory[];
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

  // Represent the SMEM buffers for A and B
  Tensor tCsA = shared_storage.tensor_sA(); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCsB = shared_storage.tensor_sB(); // (MmaB, NumMma_M, NumMma_K, Tiles_K)
  // tCsA:   Sw<3,4,3>_smem_ptr[16b](0x78ed00000400) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
  // tCsB:   Sw<3,4,3>_smem_ptr[16b](0x78ed00004400) o ((_256,_16),_1,_4):((_64,_1),_0,_16)

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
  //   print("tCgA:\t"); print(tCgA); print("\n");  // tCgA:   gmem_ptr[16b](GMEM_ADDR_A + offset_for_mma_tile + offset_for_mma) o ((_128,_16),_1,_4,128):((8192,_1),_0,_16,_64)
  //   print("tCgB:\t"); print(tCgB); print("\n");  // tCgB:   gmem_ptr[16b](GMEM_ADDR_B + offset_for_mma_tile + offset_for_mma) o ((_256,_16),_1,_4,128):((8192,_1),_0,_16,_64)
  //   print("tCgC:\t"); print(tCgC); print("\n");  // tCgC:   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile + offset_for_mma) o ((_128,_256),_1,_1):((8192,_1),_0,_0)
  // } __syncthreads();

  // MMA Fragment Allocation
  // We allocate "fragments" which are SMEM descriptors that serve as inputs to cute::gemm operations.
  // For tcgen05.mma operations:
  // - Matrices A and B are sourced from SMEM
  // - tCrA and tCrB provide descriptor views of tCsA and tCsB respectively
  // - The first mode of each descriptor represents the SMEM for a single MMA operation
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
  Tensor tCrA = cta_mma.make_fragment_A(tCsA); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  Tensor tCrB = cta_mma.make_fragment_B(tCsB); // (MmaB, NumMma_M, NumMma_K, Tiles_K)
  // if (thread0()) {
  //   print("tCrA:\t"); print(tCrA); print("\n");     // tCrA:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
  //   print("tCrB:\t"); print(tCrB); print("\n");     // tCrB:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
  // } __syncthreads();

  // TMEM Allocation
  // On SM100 architecture, accumulators are stored exclusively in tensor memory (TMEM).
  // ThrMma's make_fragment_C() creates a TMEM tensor with the appropriate layout for the accumulator.

  // Tensor memory has address of 32 bits， 128 rows * 512 columns of fp32
  // 31 - 16 is row index, 15 - 0 is column index.
  // each warp can access 32 rows, i.e. warp 0 access row 0 - row 31, warp 1 access row 32 - row 63, ...
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC); // (MmaC, NumMma_M, NumMma_N)
  // tmem_[32b](0x0000.0000) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
  // stride = 65536 = 1 << 16, i.e. address (1, 0) is 0x0001.0000

  uint32_t elect_one_thr = cute::elect_one_sync();
  // thread 0 of each warp
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

  // TMA Setup
  //
  //   These are TMA partitionings, which have a dedicated custom partitioner.
  //   In this example, the TMA multicasts the loads across multiple CTAs.
  //   Loads of A are multicasted along the N dimension of the cluster_shape_MNK and
  //   Loads of B are multicasted along the M dimension of the cluster_shape_MNK.
  //      Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
  //   For A tensor: The group_modes<0,3> transforms the (MmaA, NumMma_M, NumMma_K, Tiles_K)-shaped tensor
  //      into ((MmaA, NumMma_M, NumMma_K), Tiles_K). The partitioning only pays attention to mode-0, the MMA Tile MK.
  //   For B tensor: The group_modes<0,3> transforms the (MmaB, NumMma_M, NumMma_K, Tiles_K)-shaped tensor
  //      into ((MmaB, NumMma_M, NumMma_K), Tiles_K). The partitioning only pays attention to mode-0, the MMA Tile NK.
  //   Simply put, the TMA will be responsible for everything in mode-0 with a single call to cute::copy.
  //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.

  // Each CTA with the same m-coord will load a portion of A
  // Each CTA with the same n-coord will load a portion of B
  // Multicast behavior for CTA 1,2 in the cluster
  //   A multicast            B multicast
  //    0  1  2  3             0  1  2  3
  // 0  -  -  -  -          0  -  -  X  -
  // 1  X  X  X  X          1  -  -  X  -
  // 2  -  -  -  -          2  -  -  X  -
  // 3  -  -  -  -          3  -  -  X  -
  // tma_multicast_mask_A = 0x2222
  // tma_multicast_mask_B = 0x0F00
  // mma_multicast_mask_C = 0x2F22

  // Construct the CTA-in-Cluster coordinate for multicasting
  auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));

  // Project the cluster_layout for tma_A along the N-modes
  auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                    get<2>(cta_in_cluster_coord_vmnk),         // The CTA coordinate along N mode of the cluster
                                    make_layout(size<2>(cluster_layout_vmnk)), // The CTA layout along N mode of the cluster
                                    group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));

  // Project the cluster_layout for tma_B along the M-modes
  auto [tBgB, tBsB] = tma_partition(tma_atom_B,
                                    get<1>(cta_in_cluster_coord_vmnk),         // The CTA coordinate along M mode of the cluster
                                    make_layout(size<1>(cluster_layout_vmnk)), // The CTA layout along M mode of the cluster
                                    group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

  // Project the cluster_layout and cta_coord along the N-mode to determine the multicast mask for A
  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  // Project the cluster_layout and cta_coord along the M-mode to determine the multicast mask for B
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  // Project the cluster_layout and cta_coord along the VM + VN-modes to determine the multicast mask for C
  uint16_t mma_mcast_mask_c = create_tma_multicast_mask<0, 1>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk) |
                              create_tma_multicast_mask<0, 2>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

  // Calculate total bytes that TMA will transfer each tile to track completion
  int tma_transaction_bytes = sizeof(make_tensor_like(tAsA)) + sizeof(make_tensor_like(tBsB));

  // if (thread0()) {
  //   print("tAgA:\t"); print(tAgA); print("\n");  // tAgA:   ArithTuple(_0,0) o (((_64,_128),_1),4):(((_1@0,_1@1),_0),_64@0)
  //   print("tAsA:\t"); print(tAsA); print("\n");  // tAsA:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_A) o ((_8192,_1)):((_1,_0))
  //   print("tBgB:\t"); print(tBgB); print("\n");  // tBgB:   ArithTuple(_0,0) o (((_64,_256),_1),4):(((_1@0,_1@1),_0),_64@0)
  //   print("tBsB:\t"); print(tBsB); print("\n");  // tBsB:   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_16384,_1)):((_1,_0))
  //   printf("tma_transaction_bytes: %d\n", tma_transaction_bytes);
  //   printf("tma_mcast_mask_a: %x\n", tma_mcast_mask_a);
  //   printf("tma_mcast_mask_b: %x\n", tma_mcast_mask_b);
  //   printf("mma_mcast_mask_c: %x\n", mma_mcast_mask_c);
  // } __syncthreads();

  // Barrier Initialization
  // Barriers in SMEM initialized by a single thread.
  if (elect_one_warp && elect_one_thr)
  {
    // The number of CTAs that participates in multicast operation with this CTA (for both A and B matrices)
    int num_mcast_participants = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ num_mcast_participants);
    cute::initialize_barrier(shared_storage.tma_barrier, /* num_threads */ 1);
  }
  //   The phase of an mbarrier object is the number of times the mbarrier object has been used to synchronize threads and asynchronous operations. In each phase {0, 1, 2, …}, threads perform in program order :

  // arrive-on operations to complete the current phase and

  // test_wait / try_wait operations to check for the completion of the current phase.

  // An mbarrier object is automatically reinitialized upon completion of the current phase for immediate use in the next phase. The current phase is incomplete and all prior phases are complete.

  // For each phase of the mbarrier object, at least one test_wait or try_wait operation must be performed which returns True for waitComplete before an arrive-on operation in the subsequent phase.

  int mma_barrier_phase_bit = 0; // Each barrier has an associated phase_bit.
  int tma_barrier_phase_bit = 0; // Each barrier has an associated phase_bit.
  cute::cluster_sync();          // Make sure all threads across all CTAs in Cluster observe barrier initialization.

  // Step 2: The Mainloop.

  // Set mma accumulate option to zero so that the first MMA instruction will clear the TMEM accumulator.
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // Execute a MmaTile_M x MmaTile_N x GEMM_K GEMM
  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
  {
    // Step 2a: Load A and B tiles

    // TMA Load Operations:
    // - Execute asynchronous TMA loads with single thread
    // - Set transaction bytes and execute with barrier
    if (elect_one_warp && elect_one_thr)
    {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
      copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a), tAgA(_, k_tile), tAsA); // Load MmaTile_M x MmaTile_K A tile
      copy(tma_atom_B.with(shared_storage.tma_barrier, tma_mcast_mask_b), tBgB(_, k_tile), tBsB); // Load MmaTile_N x MmaTile_K B tile
    }

    // Step 2b: Execute the MMAs for this tile

    // Wait for TMA loads to SMEM to complete
    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;

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
      cutlass::arch::umma_arrive_multicast(&shared_storage.mma_barrier, mma_mcast_mask_c); // All multicasting CTAs encoded in mask.
    }
    // Wait MMAs to complete to avoid overwriting the A and B SMEM.
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
    // phase bit has to be changed after each wait_barrier,
    // otherwise, next iteration, wait_barrier will not wait using old phase
  }

  // Step 3: The Epilogue.

  // Create the tiled copy operation for the accumulator (TMEM -> RMEM)
  TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  // ThrCopy
  //   ThrIdx: 0
  // TiledCopy
  //   Tiler_MN:       ((_4,_32):(_32,_1),_1:_0,_1:_0)
  //   TiledLayout_TV: ((_32,_4),_32):((_0,_1),_4)
  // Copy_Atom
  //   ThrID:        _32:_1
  //   ValLayoutSrc: (_32,_32):(_0,_1)
  //   ValLayoutDst: (_32,_1):(_1,_1)
  //   ValLayoutRef: (_32,_32):(_0,_1)
  //   ValueType:    32b

  Tensor tDgC = thr_t2r_copy.partition_D(tCgC); // (CpyD, NumCpy_M, NumCpy_N)
  // Tensor tDrC = make_fragment_like(tDgC);       // (CpyD, NumCpy_M, NumCpy_N)
  // We don't need to load C from GMEM for beta = 0
  // Load C tensor GMEM -> RMEM
  // copy(tDgC, tDrC);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc); // (CpyS, NumCpy_M, NumCpy_N)
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgC)); // (CpyD, NumCpy_M, NumCpy_N)
  // Load TMEM -> RMEM
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  // AXPBY RMEM -> RMEM: tDrC = alpha * tDrAcc + beta * tDrC
  // axpby(alpha, tDrAcc, beta, tDrC);

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

cudaError_t cute_cluster_example_gemm(
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
  // all use T0,layout is simple
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
  auto mma_tiler = make_shape(bM, bN, bK);      // (MMA_M, MMA_N, MMA_K) 128, 256, 64

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
  auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  // cluster_layout_vmnk:  ((_1),_4,_4,_1):((_0),_1,_4,_0)

  // Create TMA descriptors for A and B matrices
  // TMA is cp.async with 2D ~ 5D
  Copy_Atom tma_atom_A = make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{},   // TMA load operation with multicast
      mA,                          // Source GMEM tensor
      sA_layout,                   // Destination SMEM layout
      select<0, 2>(mma_tiler),     // MK Tiler for TMA operation
      size<2>(cluster_layout_vmnk) // The number of CTAs in the N-mode for multicasting
  );
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA)); // (Gemm_M, Gemm_K)
  // ArithTuple(_0,_0) o (16384,8192):(_1@1,_1@0)

  // print("tma_atom_A:\t"); print(tma_atom_A); print("\n");
  // tma_atom_A:     Copy_Atom
  //  ThrID:        _1:_0
  //  ValLayoutSrc: (_1,_8192):(_0,_1)
  //  ValLayoutDst: (_1,_8192):(_0,_1)
  //  ValLayoutRef: (_1,_8192):(_0,_1)
  //  ValueType:    16b

  Copy_Atom tma_atom_B = make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{},   // TMA Load Op
      mB,                          // Source GMEM tensor
      sB_layout,                   // Destination SMEM layout
      select<1, 2>(mma_tiler),     // NK Tiler for TMA operation
      size<1>(cluster_layout_vmnk) // The number of CTAs in the M-mode for multicasting
  );
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB)); // (Gemm_N, Gemm_K)

  // print("tma_atom_B:\t"); print(tma_atom_B); print("\n");
  // tma_atom_B:     Copy_Atom
  //  ThrID:        _1:_0
  //  ValLayoutSrc: (_1,_16384):(_0,_1)
  //  ValLayoutDst: (_1,_16384):(_0,_1)
  //  ValLayoutRef: (_1,_16384):(_0,_1)
  //  ValueType:    16b

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

  auto *kernel_ptr = &gemm_device_1<SMEMStorage,
                                    decltype(mA_tma), decltype(mB_tma), decltype(mC), // decltype(mA), decltype(mB), decltype(mC)
                                    decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                                    decltype(tma_atom_A), decltype(tma_atom_B), // Includes the TMA descriptor.
                                    float, float>;

  // Set kernel attributes (set SMEM)
  cudaFuncSetAttribute(kernel_ptr,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smemBytes);

  // printf("Grid launched: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  // printf("Cluster launched: %d, %d, %d\n", dimCluster.x, dimCluster.y, dimCluster.z);

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const *)kernel_ptr,
                                                             mA_tma, mB_tma, mC, // mA, mB, mC
                                                             mma_tiler, tiled_mma, cluster_shape,
                                                             tma_atom_A, tma_atom_B,
                                                             alpha, beta);

  return cudaSuccess;
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

void cute_cluster_example(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c)
{
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  const int lda = K;
  const int ldb = K;
  const int ldc = N;
  auto result = cute_cluster_example_gemm(M, N, K, 1., reinterpret_cast<cute::half_t *>(a.data_ptr()), lda, reinterpret_cast<cute::half_t *>(b.data_ptr()), ldb, 0., reinterpret_cast<cute::half_t *>(c.data_ptr()), ldc, a.device());
  if (result != cudaSuccess)
  {
    std::cerr << "CUTLASS GEMM kernel failed: "
              << cudaGetErrorString(result) << std::endl;
  }
}
