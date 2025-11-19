#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cute/tensor.hpp>
#include "cute/config.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <cutlass/util/print_error.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;
using cutlass::bfloat16_t;

__device__ __forceinline__
float warp_sum_full(float v) {
    unsigned mask = 0xFFFFFFFFu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

template <class ElementA, class ElementB,
          class SmemLayoutA, // (M,K,P)
          class SmemLayoutB> // (N,K,P)
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  //uint64_t intermediate_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler, class SmemLayoutA, class TmaA,
          class SmemLayoutB, class TmaB, class TiledMma>
__global__ static __launch_bounds__(384) void gemm_{{KERNEL_ID}}(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    CUTLASS_GRID_CONSTANT TmaA const tma_a,
    CUTLASS_GRID_CONSTANT TmaB const tma_b,
    TiledMma mma,
    {{ARGS}}) {
  const uint32_t aux_threads = 128;
  const uint32_t mma_threads = 256;

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K)); // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K)); // (N,K) TMA Tensor
  bfloat16_t *temp = nullptr;
  Tensor mC = make_tensor(make_gmem_ptr(temp), make_shape(M, N), make_stride(_1{}, M)); // (M,N)

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                         Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                         Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  extern __shared__ char shared_memory[];
  using SharedStorage =
      SharedStorage<bfloat16_t, bfloat16_t, SmemLayoutA, SmemLayoutB>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()),
                          SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()),
                          SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

  auto [tAgA, tAsA] =
      tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                    group_modes<0, 2>(gA)); // (TMA,k) and (TMA,PIPE)

  auto [tBgB, tBsB] =
      tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                    group_modes<0, 2>(gB)); // (TMA,k) and (TMA,PIPE)

  constexpr int tma_transaction_bytes =
      sizeof(make_tensor_like(tensor<0>(tAsA))) +
      sizeof(make_tensor_like(tensor<0>(tBsB)));

  auto K_PIPE_MAX = size<1>(tAsA);

  int k_tile_count = size<1>(tAgA);
  int k_tile = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t *producer_mbar = smem.tma_barrier;
  uint64_t *consumer_mbar = smem.mma_barrier;
  //uint64_t *intermediate_mbar = smem.intermediate_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
  //using IntermediateBarType = cutlass::arch::ClusterBarrier;
  using ConsumerBarType = cutlass::arch::ClusterBarrier; // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe], 1);
      //IntermediateBarType::init(&intermediate_mbar[pipe], aux_threads);
      ConsumerBarType::init(&consumer_mbar[pipe], mma_threads);
    }
  }
  cluster_sync();

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x - aux_threads);
  Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
  Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // Allocate "fragments"
  Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

  struct EpilogueSmem {
    bfloat16_t out_tile[128][258];
    bfloat16_t pre_reduction_scratch[128][258];
  };

  if (threadIdx.x < aux_threads) {
    // Warp specialized to auxiliary
    //asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n");
    auto producer_state = cutlass::PipelineState<K_PIPE_MAX>();
    //auto intermediate_state = cutlass::PipelineState<K_PIPE_MAX>();
    while (k_tile_count > -K_PIPE_MAX) {

      if (warp_idx == 0 && lane_predicate) {
        int pipe = producer_state.index();
        if (producer_state.count() >= K_PIPE_MAX) {
          // Wait for next slot to be available (from previous phase)
          ConsumerBarType::wait(&consumer_mbar[pipe],
                                producer_state.phase() ^ 1);
        }

        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                              tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
      }

      if (producer_state.count() > 0) {
        //ProducerBarType::wait(&producer_mbar[intermediate_state.index()],
        //                      intermediate_state.phase());
        // --- intermediate ops (mainloop fusion) ----
        {{MAINLOOP}}
        //IntermediateBarType::arrive(
        //    &intermediate_mbar[intermediate_state.index()]);
        //++intermediate_state;
      }

      ++producer_state;
      ++k_tile;
      --k_tile_count;
    }

    // finish intermediate ops
    for (int i = 0; i < 1; i++) {
      //ProducerBarType::wait(&producer_mbar[intermediate_state.index()],
      //                      intermediate_state.phase());
      // --- intermediate ops (mainloop fusion) ----
      //IntermediateBarType::arrive(
      //    &intermediate_mbar[intermediate_state.index()]);
      {{MAINLOOP}}
      //++intermediate_state;
    }
  } else {
    // Warp specialized to MMA
    //asm volatile("setmaxnreg.inc.sync.aligned.u32 200;\n");
    auto producer_state = cutlass::PipelineState<K_PIPE_MAX>();
    auto mma_state = cutlass::PipelineState<K_PIPE_MAX>();

    while (k_tile_count > -K_PIPE_MAX) {
      // Wait for intermediate to complete
      int producer_pipe = producer_state.index();

      //IntermediateBarType::wait(&intermediate_mbar[producer_pipe],
      //                          producer_state.phase());
      ProducerBarType::wait(&producer_mbar[producer_pipe], producer_state.phase());

      // MMAs to cover 1 K_TILE
      warpgroup_arrive();
      gemm(mma, tCrA(_, _, _, producer_pipe), tCrB(_, _, _, producer_pipe),
           tCrC); // (V,M) x (V,N) => (V,M,N)
      warpgroup_commit_batch();

      // Wait for previous k tile to finish
      warpgroup_wait<1>();

      if (producer_state.count() > 0) {

        // Notify that consumption of previous tile is done
        ConsumerBarType::arrive(&consumer_mbar[mma_state.index()]);
        ++mma_state;
      }

      ++producer_state;
      --k_tile_count;
      ++k_tile;
    }

    warpgroup_wait<0>();
    ConsumerBarType::arrive(&consumer_mbar[mma_state.index()]);
    ++mma_state;

    // MMA done; prep epilogue

    auto *epilogue_smem = reinterpret_cast<EpilogueSmem *>(shared_memory);

    auto sC = make_tensor(
        make_smem_ptr(reinterpret_cast<bfloat16_t *>(epilogue_smem->out_tile)),
        Layout<Shape<Shape<_8, _16>, Shape<_8, _32>>, Stride<Stride<_8, _2048>, Stride<_1, _64>>>{});

    auto sPreReduction = make_tensor(
        make_smem_ptr(reinterpret_cast<bfloat16_t *>(epilogue_smem->pre_reduction_scratch)),
        Layout<Shape<_128, _256>, Stride<Int<258>, _1>>{});

    auto tCrC_converted = make_tensor<bfloat16_t>(tCrC.layout());
    transform(tCrC, tCrC_converted, [](float x) {
      return static_cast<bfloat16_t>(__float2bfloat16_rn(x));
    });

    auto tiled_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, bfloat16_t>{}, mma);
    auto thr_copy = tiled_copy.get_slice(threadIdx.x - aux_threads);
    auto tCsC = thr_copy.partition_D(sC);
    copy(tiled_copy, thr_copy.retile_S(tCrC_converted), tCsC);
  }

  __syncthreads();

  auto *epilogue_smem = reinterpret_cast<EpilogueSmem *>(shared_memory);

  auto sC = make_tensor(
      make_smem_ptr(reinterpret_cast<bfloat16_t *>(epilogue_smem->out_tile)),
      Layout<Shape<_128, _256>, Stride<Int<256>, _1>>{});

  auto sPreReduction = make_tensor(
      make_smem_ptr(reinterpret_cast<bfloat16_t *>(epilogue_smem->pre_reduction_scratch)),
      Layout<Shape<_128, _256>, Stride<Int<258>, _1>>{});

  auto epilogue_slicer =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, bfloat16_t>{},
                      Layout<Shape<_2, _128>, Stride<_128, _1>>{},
                      Layout<Shape<_64, _2>, Stride<_2, _1>>{});
  auto epilogue_partitioner =
      epilogue_slicer.get_slice(threadIdx.x);

  auto dummy = make_identity_tensor(Shape<_128, _256>{});
  auto coord_slice = epilogue_partitioner.partition_D(dummy);

  auto thread_c = epilogue_partitioner.partition_D(sC);
  auto thread_pre_reduction = epilogue_partitioner.partition_D(sPreReduction);

 {{EPILOGUE}}
}

void {{KERNEL_ID}}({{ARGS}},
             cudaStream_t stream = 0) {
  auto M = Int<{{M}}>{};
  auto N = Int<{{N}}>{};
  auto K = Int<{{K}}>{};
  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  auto bM = Int<128>{};
  auto bN = Int<256>{};
  auto bK = Int<64>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<4>{};                      // Pipeline

  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<bfloat16_t>{},
                          make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<bfloat16_t>{},
                          make_shape(bN, bK, bP));

  TiledMMA tiled_mma = make_tiled_mma(
      SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::MN, GMMA::Major::MN>{},
      Layout<Shape<_2, _1>>{});

  Tensor mA = make_tensor({{SYM_MATMUL_A}}, make_layout(make_shape(M, K), make_stride(_1{}, M)));
  Tensor mB = make_tensor({{SYM_MATMUL_B}}, make_layout(make_shape(N, K), make_stride(_1{}, N)));

  Copy_Atom tmaA =
      make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
  Copy_Atom tmaB =
      make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

  int smem_size = int(sizeof(
      SharedStorage<bfloat16_t, bfloat16_t, decltype(sA), decltype(sB)>));
  dim3 dimBlock(384);
  dim3 dimCluster(2, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(M, 192)), dimCluster.x),
               round_up(size(ceil_div(N, 256)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smem_size};

  void const *kernel_ptr = reinterpret_cast<void const *>(
      &gemm_{{KERNEL_ID}}<decltype(prob_shape), decltype(cta_tiler), decltype(sA),
                   decltype(tmaA), decltype(sB), decltype(tmaB),
                   decltype(tiled_mma)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr, prob_shape, cta_tiler, tmaA, tmaB, tiled_mma, {{ARGS_PASSED}});
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel launch" << std::endl;
  }
}
