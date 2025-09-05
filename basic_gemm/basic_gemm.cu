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
#include <cutlass/gemm/device/gemm.h>

#ifndef TORCH_CURRENT_DEVICE
#define TORCH_CURRENT_DEVICE cutlass::arch::Sm80
#endif

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

  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t,                // Data-type of A matrix
                                                  RowMajor,                       // Layout of A matrix
                                                  cutlass::half_t,                // Data-type of B matrix
                                                  ColumnMajor,                    // Layout of B matrix
                                                  cutlass::half_t,                // Data-type of C matrix
                                                  RowMajor,                       // Layout of C matrix
                                                  float,                          // Data-type of accumulator
                                                  cutlass::arch::OpClassTensorOp, // Use Tensor Cores check include/cutlass/arch/mma.h
                                                  TORCH_CURRENT_DEVICE            // GPU architecture
                                                  // third_party/cutlass/include/cutlass/gemm/device/default_gemm_configuration.h
                                                  // cutlass::gemm::GemmShape<128, 256, 64>, // Threadblock tile size
                                                  // cutlass::gemm::GemmShape<64, 64, 64>,   // Warp tile size
                                                  // cutlass::gemm::GemmShape<16, 8, 16>,    // mma Instruction tile size
                                                  // cutlass::gemm::device::DefaultGemmConfiguration<
                                                  //     cutlass::arch::OpClassTensorOp, TORCH_CURRENT_DEVICE, cutlass::half_t, cutlass::half_t, cutlass::half_t,
                                                  //     float>::EpilogueOutputOp,
                                                  // cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Threadblock swizzling function
                                                  // 3                                                             // Stages, default is 3
                                                  >;

  //

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M, N, K},     // Gemm Problem dimensions
                              {A, lda},      // Tensor-ref for source matrix A
                              {B, ldb},      // Tensor-ref for source matrix B
                              {C, ldc},      // Tensor-ref for source matrix C
                              {C, ldc},      // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}, // Scalars used in the Epilogue
                              1);            // split_k_slices
  //
  // Launch the CUTLASS GEMM kernel.
  //
  // workspace_size should be 0 becuase split_k_slices = 1, we can also directly call gemm_operator(args)
  size_t workspace_size = CutlassGemm::get_workspace_size(args);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::dtype(torch::kUInt8).device(device));
  gemm_operator.initialize(args, workspace.data_ptr<uint8_t>());
  cutlass::Status status = gemm_operator();
  // cutlass::Status status = gemm_operator(args);

  /*
    device/gemm.h -> kernel/default_gemm.h (partial specilaization on arch::sm80, choose mma and gemm kernel)
    -> kernel/gemm.h
    About gemm kernel:
      1. launch config:
        dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(GemmKernel::kThreadCount, 1, 1);
        So grid =(M / ThreadblockShape_M, N / ThreadblockShape_N, split_k_slices = 1)
           using WarpCount = typename Mma::WarpCount;
           block = 32 * WarpCount::kCount = 32 * (ThreadblockShape / WarpShape)
      2. kernel implementation starts in kernel/gemm.h
        void operator()(Params const &params, SharedStorage &shared_storage)

        threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile)
          template <int N = 1>
          struct GemmIdentityThreadblockSwizzle
          when N = 1, no swizzle. params.swizzle_log_tile = 0

        using Mma = typename cutlass::gemm::threadblock::DefaultMma<...>::ThreadblockMma;
        call mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

        semaphore is not used when split_k_slices = 1
      3. main loop in mma. gemm/threadblock/default_mma.h  Specialization for row-major output (OperatorClass TensorOp)
         using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage
         stage = 2 will use MmaPipelined
         also check MmaSingleStage
         -> gemm/threadblock/mma_multistage.h
         CUTLASS_DEVICE void operator()(...)
         it will call prologue(), gmem_wait(), gemm_iters()
         call warp mma
      4. warp mma
         cutlass/gemm/warp/mma_tensor_op.h
         it will call ArchMmaOperator
      5. cutlass/arch/mma_sm80.h
  */

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  TORCH_BINDING_COMMON_EXTENSION(basic_gemm)
}
