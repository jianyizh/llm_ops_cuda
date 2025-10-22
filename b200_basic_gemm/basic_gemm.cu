#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;

cudaError_t basic_cutlass_gemm(
    int M,
    int N,
    int K,
    float alpha,
    cutlass::half_t const *A,
    int lda,
    cutlass::half_t const *B,
    int ldb,
    float beta,
    cutlass::half_t *C,
    int ldc,
    torch::Device device)
{

  half_t alpha_half = half_t(alpha);
  half_t beta_half = half_t(beta);

  // A matrix configuration
  using ElementA = half_t;                                                // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;                              // Layout type for A matrix operand
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = half_t;                                                // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;                           // Layout type for B matrix operand
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementC = half_t;                                                // Element type for C and D matrix operands
  using LayoutC = cutlass::layout::RowMajor;                              // Layout type for C and D matrix operands
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // Kernel functional config
  using ElementAccumulator = float;                     // Element type for internal accumulation
  using ArchTag = cutlass::arch::Sm100;                 // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag

  // MMA and Cluster Tile Shapes
  // Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape %2 == 0
  using MmaTileShape_MNK = Shape<_256, _128, _64>;
  // Shape of the threadblocks in a cluster
  using ClusterShape_MNK = Shape<_2, _2, _1>;

  // Build the epilogue
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  // Build the mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  // Compose into a kernel
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      void>; // Default to ClusterLaunchControl (CLC) based tile scheduler

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {A, stride_A, B, stride_B},
      {{alpha, beta}, C, stride_C, C, stride_D}};

  arguments.scheduler.max_swizzle_size = 0; // Cluster rasterization swizzle

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::dtype(torch::kUInt8).device(device));
  // Check if the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm.initialize(arguments, workspace.data_ptr<uint8_t>());

  status = gemm.run();
  if (status != cutlass::Status::kSuccess)
  {
    cutlass::Status error = status;
    std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << std::endl;
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

void basic_gemm(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c)
{
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  const int lda = K;
  const int ldb = K;
  const int ldc = N;
  auto result = basic_cutlass_gemm(M, N, K, 1., reinterpret_cast<cutlass::half_t *>(a.data_ptr()), lda, reinterpret_cast<cutlass::half_t *>(b.data_ptr()), ldb, 0., reinterpret_cast<cutlass::half_t *>(c.data_ptr()), ldc, a.device());
  if (result != cudaSuccess)
  {
    std::cerr << "CUTLASS GEMM kernel failed: "
              << cudaGetErrorString(result) << std::endl;
  }
}

// extern void cute_example(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  TORCH_BINDING_COMMON_EXTENSION(basic_gemm)
  // TORCH_BINDING_COMMON_EXTENSION(cute_example)
}
