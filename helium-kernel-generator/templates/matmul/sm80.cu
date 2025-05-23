// SM80+ generic matmul, with flexible epilogue & mainloop fusions, v2

// struct IdentityMainloopFusion {
//   template <typename TensorA> __device__ void apply_thr_a(TensorA &tensor_a) {}

//   template <typename TensorB> __device__ void apply_thr_b(TensorB &tensor_b) {}
// };

// template <typename LayoutGmemC, typename TiledCopyR2G_C, typename TileSize>
// struct SimpleEpilogue {
//   LayoutGmemC _layout_gmem_c;
//   TiledCopyR2G_C _tiled_copy_r2g_c;
//   float *_raw_gmem_c;

//   __device__ auto _gmem_c() { return make_tensor(_raw_gmem_c, _layout_gmem_c); }

//   template <typename TensorC> __device__ void apply_tile_c(TensorC tensor_c) {
//     const auto coord_c = make_coord(blockIdx.x, blockIdx.y);
//     auto tile_gmem_c = local_tile(
//         _gmem_c(), make_shape(size<0>(TileSize{}), size<1>(TileSize{})),
//         coord_c);

//     auto thr_copy_r2g_c = _tiled_copy_r2g_c.get_slice(threadIdx.x);
//     auto thr_smem_c = thr_copy_r2g_c.partition_S(tensor_c);
//     auto thr_gmem_c = thr_copy_r2g_c.partition_D(tile_gmem_c);

//     // Convert to float
//     auto thr_reg_c = make_fragment_like(thr_smem_c);
//     copy(thr_smem_c, thr_reg_c);
//     auto thr_reg_c_converted = make_tensor<float>(thr_reg_c.layout());
//     transform(thr_reg_c, thr_reg_c_converted,
//               [](auto x) { return static_cast<float>(x); });

//     // Check if we need to predicate the copy
//     const auto end_m = (blockIdx.x + 1) * size<0>(TileSize{});
//     const auto end_n = (blockIdx.y + 1) * size<1>(TileSize{});
//     if (end_m <= size<0>(_layout_gmem_c) && end_n <= size<1>(_layout_gmem_c)) {
//       copy(_tiled_copy_r2g_c, thr_reg_c_converted, thr_gmem_c);
//     } else {
//       auto dummy = make_identity_tensor(
//           make_shape(size<0>(_layout_gmem_c), size<1>(_layout_gmem_c)));
//       auto tile_dummy = local_tile(
//           dummy, make_shape(size<0>(TileSize{}), size<1>(TileSize{})), coord_c);
//       auto thr_dummy = thr_copy_r2g_c.partition_D(tile_dummy);

//       auto thr_pred = make_tensor<bool>(thr_reg_c_converted.layout());
//       for (int i = 0; i < size(thr_pred); i++) {
//         const auto coord = thr_dummy(i);
//         thr_pred(i) = get<0>(coord) < size<0>(_layout_gmem_c) &&
//                       get<1>(coord) < size<1>(_layout_gmem_c);
//       }

//       copy_if(thr_pred, thr_reg_c_converted, thr_gmem_c);
//     }
//   }
// };

template<typename T>
auto make_smem_ptr_restrict(T *__restrict__ ptr) {
    return smem_ptr<T *__restrict__>{ptr};
}

template<typename T>
auto make_gmem_ptr_restrict(T *__restrict__ ptr) {
    return gmem_ptr<T *__restrict__>{ptr};
}

template <typename InDtypeA, typename InDtypeB, typename GemmDtypeA,
          typename GemmDtypeB, typename DtypeAccum, typename DtypeOut,
          typename LayoutGmemA, typename LayoutGmemB, typename MainloopFusion,
          typename Epilogue, typename TileSize, typename LayoutSmemA,
          typename LayoutSmemB, typename LayoutSmemC, typename TiledCopyG2S_A,
          typename TiledCopyG2S_B, typename TiledCopyS2R_A,
          typename TiledCopyS2R_B, typename TiledMma, uint32_t PIPELINE, bool SpecializedCopyA, bool SpecializedCopyB,
          typename ThreadBlockSwizzle>
struct Matmul {
  LayoutGmemA _layout_gmem_a;
  LayoutGmemB _layout_gmem_b;
  MainloopFusion _mainloop_fusion;
  Epilogue _epilogue;
  TileSize _tile_size;
  TiledCopyG2S_A _tiled_copy_g2s_a;
  TiledCopyG2S_B _tiled_copy_g2s_b;
  TiledCopyS2R_A _tiled_copy_s2r_a;
  TiledCopyS2R_B _tiled_copy_s2r_b;
  TiledMma _tiled_mma;
  const InDtypeA* __restrict__ _raw_gmem_a;
  const InDtypeB* __restrict__ _raw_gmem_b;
  uint32_t _num_k_blocks = 0;
  ThreadBlockSwizzle _thread_block_swizzle;

  struct MainloopStorage {
    InDtypeA smem_storage_a[PIPELINE + 1][cosize_v<LayoutSmemA>];
    InDtypeB smem_storage_b[PIPELINE + 1][cosize_v<LayoutSmemB>];
  };

  struct EpilogueStorage {
    DtypeOut smem_storage_c[cosize_v<LayoutSmemC>];
  };

  union SharedStorage {
    MainloopStorage mainloop;
    EpilogueStorage epilogue;
  };

  SharedStorage* __restrict__ _shared_storage;

  __device__ auto _smem_a(uint32_t pipeline_slot) {
    return make_tensor(
        make_smem_ptr_restrict(reinterpret_cast<InDtypeA* __restrict__>(
            _shared_storage->mainloop.smem_storage_a[pipeline_slot])),
        LayoutSmemA{});
  }

  __device__ auto _smem_b(uint32_t pipeline_slot) {
    return make_tensor(
        make_smem_ptr_restrict(reinterpret_cast<InDtypeB* __restrict__>(
            _shared_storage->mainloop.smem_storage_b[pipeline_slot])),
        LayoutSmemB{});
  }

  __device__ auto _smem_c() {
    return make_tensor(make_smem_ptr_restrict(reinterpret_cast<DtypeOut* __restrict__>(
                           _shared_storage->epilogue.smem_storage_c)),
                       LayoutSmemC{});
  }

  __device__ auto _gmem_a() { return make_tensor(make_gmem_ptr_restrict(_raw_gmem_a), _layout_gmem_a); }

  __device__ auto _gmem_b() { return make_tensor(make_gmem_ptr_restrict(_raw_gmem_b), _layout_gmem_b); }

  __device__ auto _thr_copy_g2s_a() {
    return _tiled_copy_g2s_a.get_slice(threadIdx.x);
  }

  __device__ auto _thr_copy_g2s_b() {
    return _tiled_copy_g2s_b.get_slice(threadIdx.x);
  }

  __device__ auto _thr_copy_s2r_a() {
    return _tiled_copy_s2r_a.get_slice(threadIdx.x);
  }

  __device__ auto _thr_copy_s2r_b() {
    return _tiled_copy_s2r_b.get_slice(threadIdx.x);
  }

  __device__ auto _thr_mma() { return _tiled_mma.get_slice(threadIdx.x); }

  template<typename SmemA, typename SmemB>
  __device__ void async_load_k_block(uint32_t k_block, SmemA smem_a, SmemB smem_b) {
    const auto pipeline_slot = k_block % (PIPELINE + 1);

    const auto out_coord =
        idx2crd(_thread_block_swizzle(blockIdx.x),
                make_shape((size<0>(_layout_gmem_a) + size<0>(TileSize{}) - 1) / size<0>(TileSize{}),
                           (size<0>(_layout_gmem_b) + size<1>(TileSize{}) - 1) / size<1>(TileSize{})));

    const auto coord_a = make_coord(get<0>(out_coord), k_block);
    const auto coord_b = make_coord(get<1>(out_coord), k_block);

    auto tile_gmem_a = local_tile(
        _gmem_a(), make_shape(get<0>(TileSize{}), get<2>(TileSize{})), coord_a);
    auto tile_gmem_b = local_tile(
        _gmem_b(), make_shape(get<1>(TileSize{}), get<2>(TileSize{})), coord_b);
//
//     auto smem_a = _smem_a(pipeline_slot);
//     auto smem_b = _smem_b(pipeline_slot);

    auto thr_gmem_a = _thr_copy_g2s_a().partition_S(tile_gmem_a);
    auto thr_gmem_b = _thr_copy_g2s_b().partition_S(tile_gmem_b);

    auto thr_smem_a = _thr_copy_g2s_a().partition_D(smem_a);
    auto thr_smem_b = _thr_copy_g2s_b().partition_D(smem_b);

    const auto end_m = (blockIdx.x + 1) * size<0>(TileSize{});
    const auto end_n = (blockIdx.y + 1) * size<1>(TileSize{});
    const auto end_k = (k_block + 1) * size<2>(TileSize{});

    // Statically disable predication if the tile sizes divide the matrix sizes
    constexpr bool needs_predication_check = size<0>(LayoutGmemA{}) % size<0>(TileSize{}) != 0
                                                || size<0>(LayoutGmemB{}) % size<1>(TileSize{}) != 0
                                                || size<1>(LayoutGmemA{}) % size<2>(TileSize{}) != 0;
    if constexpr (needs_predication_check) {
        if (end_m <= size<0>(_layout_gmem_a) && end_k <= size<1>(_layout_gmem_a)) {
          copy(_tiled_copy_g2s_a, thr_gmem_a, thr_smem_a);
        } else {
          auto dummy = make_identity_tensor(
              make_shape(size<0>(_layout_gmem_a), size<1>(_layout_gmem_a)));
          auto tile_dummy = local_tile(
              dummy, make_shape(size<0>(TileSize{}), size<2>(TileSize{})), coord_a);
          auto thr_dummy = _thr_copy_g2s_a().partition_D(tile_dummy);

          auto thr_pred = make_tensor<bool>(thr_smem_a.layout());
          for (int i = 0; i < size(thr_pred); i++) {
            const auto coord = thr_dummy(i);
            thr_pred(i) = get<0>(coord) < size<0>(_layout_gmem_a) &&
                          get<1>(coord) < size<1>(_layout_gmem_a);
            if (!thr_pred(i)) {
              thr_smem_a(i) = static_cast<InDtypeA>(0.0f);
            }
          }

          copy_if(thr_pred, thr_gmem_a, thr_smem_a);
        }

        if (end_n <= size<0>(_layout_gmem_b) && end_k <= size<1>(_layout_gmem_b)) {
          copy(_tiled_copy_g2s_b, thr_gmem_b, thr_smem_b);
        } else {
          auto dummy = make_identity_tensor(
              make_shape(size<0>(_layout_gmem_b), size<1>(_layout_gmem_b)));
          auto tile_dummy = local_tile(
              dummy, make_shape(size<1>(TileSize{}), size<2>(TileSize{})), coord_b);
          auto thr_dummy = _thr_copy_g2s_b().partition_D(tile_dummy);

          auto thr_pred = make_tensor<bool>(thr_smem_b.layout());
          for (int i = 0; i < size(thr_pred); i++) {
            const auto coord = thr_dummy(i);
            thr_pred(i) = get<0>(coord) < size<0>(_layout_gmem_b) &&
                          get<1>(coord) < size<1>(_layout_gmem_b);
            if (!thr_pred(i)) {
              thr_smem_b(i) = static_cast<InDtypeB>(0.0f);
            }
          }

          copy_if(thr_pred, thr_gmem_b, thr_smem_b);
        }
    } else { // if constexpr
        copy(_tiled_copy_g2s_a, thr_gmem_a, thr_smem_a);
        copy(_tiled_copy_g2s_b, thr_gmem_b, thr_smem_b);
    }
  }

  template <typename ThrRegC, typename SmemA, typename SmemB>
  __device__ void do_mma(ThrRegC &thr_reg_c, SmemA smem_a, SmemB smem_b, uint32_t k_block) {
    uint32_t pipeline_slot = k_block % (PIPELINE + 1);
//     auto smem_a = _smem_a(pipeline_slot);
//     auto smem_b = _smem_b(pipeline_slot);

    // Load data into registers for MMA
    auto thr_reg_a = _thr_mma().partition_fragment_A(smem_a);
    auto thr_reg_b = _thr_mma().partition_fragment_B(smem_b);

    if constexpr(SpecializedCopyA) {
        auto thr_smem_a = _thr_copy_s2r_a().partition_S(smem_a);
        copy(_tiled_copy_s2r_a, thr_smem_a, _thr_copy_s2r_a().retile_D(thr_reg_a));
    } else {
        copy(_thr_mma().partition_A(smem_a), thr_reg_a);
    }
    if constexpr(SpecializedCopyB) {
        auto thr_smem_b = _thr_copy_s2r_b().partition_S(smem_b);
        copy(_tiled_copy_s2r_b, thr_smem_b, _thr_copy_s2r_b().retile_D(thr_reg_b));
    } else {
        copy(_thr_mma().partition_B(smem_b), thr_reg_b);
    }

    // Apply mainloop fusions in registers
    _mainloop_fusion.apply_thr_a(thr_reg_a);
    _mainloop_fusion.apply_thr_b(thr_reg_b);
    
    auto thr_reg_a_converted = make_tensor<GemmDtypeA>(thr_reg_a.layout());
    auto thr_reg_b_converted = make_tensor<GemmDtypeB>(thr_reg_b.layout());
    transform(thr_reg_a, thr_reg_a_converted, [](auto x) { return static_cast<GemmDtypeA>(x); });
    transform(thr_reg_b, thr_reg_b_converted, [](auto x) { return static_cast<GemmDtypeB>(x); });

    // MMA
    gemm(_tiled_mma, thr_reg_a_converted, thr_reg_b_converted, thr_reg_c);
  }

  __device__ void run() {
    // ----
    // Initialization
    // ----
    _num_k_blocks = (size<1>(_layout_gmem_a) + size<2>(TileSize{}) - 1) /
                    size<2>(TileSize{});

    // ----
    // Fill the pipeline with initial loads of k-blocks
    // ----
    for (uint32_t k_block = 0; k_block < PIPELINE && k_block < _num_k_blocks;
         ++k_block) {
             auto smem_next_a = _smem_a((k_block + PIPELINE) % (PIPELINE + 1));
                 auto smem_next_b = _smem_b((k_block + PIPELINE) % (PIPELINE + 1));
      async_load_k_block(k_block, smem_next_a, smem_next_b);
    }

    // Allocate accumulator registers
    auto thr_reg_c = _thr_mma().partition_fragment_C(_smem_c());

    // ----
    // Main loop
    // ----
    for (uint32_t k_block = 0; k_block < _num_k_blocks - PIPELINE; ++k_block) {
      // Wait for the oldest async copy to complete
      cp_async_wait<PIPELINE - 1>();
      // (All threads must have completed their copy)
      __syncthreads();

//       if (threadIdx.x < 128) {
        uint32_t pipeline_slot = k_block % (PIPELINE + 1);
        auto smem_a = _smem_a(pipeline_slot);
        auto smem_b = _smem_b(pipeline_slot);
        auto smem_next_a = _smem_a((k_block + PIPELINE) % (PIPELINE + 1));
        auto smem_next_b = _smem_b((k_block + PIPELINE) % (PIPELINE + 1));
        do_mma(thr_reg_c, smem_a, smem_b, k_block);

        // Issue copy of next k-block in the pipeline, if one is available
        //if (k_block + PIPELINE < _num_k_blocks) {
          async_load_k_block(k_block + PIPELINE, smem_next_a, smem_next_b);
        //}
//       } else {
//         // reverse order
//         // Issue copy of next k-block in the pipeline, if one is available
//         if (k_block + PIPELINE < _num_k_blocks) {
//           async_load_k_block(k_block + PIPELINE);
//         }
//         do_mma(thr_reg_c, k_block);
//       }

      cp_async_fence();
     }

    cp_async_wait<0>();

    // ----
    // Main loop tail - last PIPELINE k-blocks
    // don't fetch additional data
    // ----
    for (uint32_t k_block = _num_k_blocks - PIPELINE; k_block < _num_k_blocks; ++k_block) {
        uint32_t pipeline_slot = k_block % (PIPELINE + 1);
            auto smem_a = _smem_a(pipeline_slot);
            auto smem_b = _smem_b(pipeline_slot);
        do_mma(thr_reg_c, smem_a, smem_b, k_block);
    }

    // ----
    // Copy accumulators to smem, converting from DtypeAccum
    // to DtypeOut in case they are different
    // ----
    auto thr_reg_c_converted = make_tensor<DtypeOut>(thr_reg_c.layout());
    transform(thr_reg_c, thr_reg_c_converted,
              [](DtypeAccum x) { return static_cast<DtypeOut>(x); });
    __syncthreads();
    auto thr_smem_c = _thr_mma().partition_C(_smem_c());
    copy(thr_reg_c_converted, thr_smem_c);

    // ----
    // Epilogue
    // ----
    __syncthreads();
    _epilogue.apply_tile_c(_smem_c());
  }
};

// using TileSize = Shape<_256, _128, _32>;

// __global__
// __launch_bounds__(256) void matmul_bf16_into_f32(bfloat16_t *a, bfloat16_t *b,
//                                                  float *c, uint32_t m,
//                                                  uint32_t n, uint32_t k) {
//   auto tiled_copy_g2s_a = make_tiled_copy(
//       Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bfloat16_t>{},
//       Layout<Shape<_64, _4>, Stride<_4, _1>>{},
//       Layout<Shape<_4, _8>, Stride<_8, _1>>{});
//   auto tiled_copy_g2s_b = make_tiled_copy(
//       Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, bfloat16_t>{},
//       Layout<Shape<_16, _16>>{}, Layout<Shape<_8, _2>>{});

//   auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
//                                   Layout<Shape<_2, _4>>{}, TileSize{});

//   auto tiled_copy_s2r_a =
//       make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, tiled_mma);
//   auto tiled_copy_s2r_b =
//       make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, tiled_mma);

//   auto tiled_copy_r2g_c =
//       make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
//                       Layout<Shape<_8, _32>, Stride<_32, _1>>{},
//                       Layout<Shape<_32, _4>, Stride<_4, _1>>{});

//   auto layout_gmem_a = make_layout(make_shape(m, k), make_stride(k, _1{}));
//   auto layout_gmem_b = make_layout(make_shape(n, k), make_stride(_1{}, n));
//   auto layout_gmem_c = make_layout(make_shape(m, n), make_stride(n, _1{}));

//   using Epilogue = SimpleEpilogue<decltype(layout_gmem_c),
//                                   decltype(tiled_copy_r2g_c), TileSize>;
//   using MatmulInstance =
//       Matmul<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, float, bfloat16_t,
//              decltype(layout_gmem_a), decltype(layout_gmem_b),
//              IdentityMainloopFusion, Epilogue, TileSize,
//              Layout<Shape<_256, _32>, Stride<Int<40>, _1>>,
//              Layout<Shape<_128, _32>, Stride<_1, Int<136>>>,
//              Layout<Shape<_256, _128>, Stride<Int<136>, _1>>,
//              decltype(tiled_copy_g2s_a), decltype(tiled_copy_g2s_b),
//              decltype(tiled_copy_s2r_a), decltype(tiled_copy_s2r_b),
//              decltype(tiled_mma), 2>;

//   extern __shared__ char shared_storage_raw[];

//   static_assert(sizeof(typename MatmulInstance::SharedStorage) <= 100000);
//   auto shared_storage =
//       reinterpret_cast<typename MatmulInstance::SharedStorage *>(
//           shared_storage_raw);

//   auto matmul = MatmulInstance{
//       ._layout_gmem_a = layout_gmem_a,
//       ._layout_gmem_b = layout_gmem_b,
//       ._mainloop_fusion = IdentityMainloopFusion{},
//       ._epilogue = Epilogue{._layout_gmem_c = layout_gmem_c,
//                             ._tiled_copy_r2g_c = tiled_copy_r2g_c,
//                             ._raw_gmem_c = c},
//       ._tile_size = {},
//       ._tiled_copy_g2s_a = tiled_copy_g2s_a,
//       ._tiled_copy_g2s_b = tiled_copy_g2s_b,
//       ._tiled_copy_s2r_a = tiled_copy_s2r_a,
//       ._tiled_copy_s2r_b = tiled_copy_s2r_b,
//       ._tiled_mma = tiled_mma,
//       ._raw_gmem_a = a,
//       ._raw_gmem_b = b,
//       ._shared_storage = shared_storage};
//   matmul.run();
// }
