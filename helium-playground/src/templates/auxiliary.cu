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

   struct Smem {
        bfloat16_t pre_reduction_scratch[128 * 258];
           bfloat16_t out_tile[1];
    };

__global__ __launch_bounds__(256) void auxiliary_{{KERNEL_ID}}({{ARGS}}) {


    uint32_t M = {{M}};
    uint32_t N = {{N}};

    extern __shared__ char shared_memory[];

    Smem *epilogue_smem = reinterpret_cast<Smem*>(shared_memory);

    uint32_t tile_item = blockIdx.z;
    uint32_t tile_m = blockIdx.x;
    uint32_t tile_n = blockIdx.y;

     auto sPreReduction = make_tensor(
          make_smem_ptr(reinterpret_cast<bfloat16_t *>(epilogue_smem->pre_reduction_scratch)),
          Layout<Shape<_128, _256>, Stride<Int<258>, _1>>{}
      );

      auto epilogue_slicer =
          make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, bfloat16_t>{},
                          Layout<Shape<_8, _32>, Stride<_32, _1>>{},
                          Layout<Shape<_16, _8>, Stride<_8, _1>>{});
      auto epilogue_partitioner =
          epilogue_slicer.get_slice(threadIdx.x);

      auto dummy = local_tile(make_identity_tensor(Shape<Int<{{M}}>, Int<{{N}}>>{}), make_shape(_128{}, _256{}), make_coord(blockIdx.x, blockIdx.y));
      auto coord_slice = epilogue_partitioner.partition_D(dummy);

        bfloat16_t *temp = nullptr;
        Tensor mC = make_tensor(make_gmem_ptr(temp), make_shape(Int<128>{}, Int<256>{}), make_stride(_1{}, Int<128>{}));

      auto thread_c = epilogue_partitioner.partition_D(mC);
      auto thread_pre_reduction = epilogue_partitioner.partition_D(sPreReduction);

      auto thread_pred = make_tensor<bool>(thread_c.layout());
      transform(coord_slice, thread_pred, [&](auto coord) {
        return get<0>(coord) < M && get<1>(coord) < N;
      });

      {{EPILOGUE}}
}

void {{KERNEL_ID}}({{ARGS}},
             cudaStream_t stream = 0) {
  auto M = {{M}};
  auto N = {{N}};

  int smem_size = int(sizeof(Smem));
  dim3 dimCluster(1, 1, 1);
  dim3 dimBlock(256);
  dim3 dimGrid((M + 127) / 128, (N + 255) / 256);
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smem_size};

  void const *kernel_ptr = reinterpret_cast<void const *>(
      &auxiliary_{{KERNEL_ID}});

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr, {{ARGS_PASSED}});
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel launch" << std::endl;
  }
}
