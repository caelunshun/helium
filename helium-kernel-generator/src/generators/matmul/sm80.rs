use crate::{
    architecture::Architecture,
    builder::{KernelBuilder, cpp_data_type, cpp_raw_byte_type, cpp_sizeof},
    cute::Layout,
    generators::matmul::{
        MatmulGenerator,
        copy_synthesis::{CopyPattern, CopyVectorizationType, find_gmem_copy_pattern},
    },
    pointwise::PointwiseContext,
};
use helium_ir::{
    data_type::{DataClass, DataType},
    opgraph::op::precision::{F8Mode, Precision},
};
use indoc::formatdoc;
use std::{
    cmp::Ordering,
    fmt::{Display, Formatter},
};

const NUM_THREADS: u32 = 256;
const PIPELINE: u32 = 2;
const MAINLOOP_TILE_SLOTS: u32 = PIPELINE + 1;

struct Sm80MatmulGenerator<'a> {
    matmul: &'a MatmulGenerator,
    builder: KernelBuilder,
    tile_layout: TileLayout,
}

/// Generates a fused matmul kernel that uses up to SM80 (Ampere) features
/// and a shared memory limit corresponding to `actual_architecture`.
pub fn generate_sm80(
    matmul: &MatmulGenerator,
    actual_architecture: Architecture,
) -> (KernelBuilder, TileLayout) {
    let (tile_layout, needed_smem) = find_tile_layout(matmul, actual_architecture);
    let mut generator = Sm80MatmulGenerator {
        matmul,
        builder: KernelBuilder::new("matmul_sm80"),
        tile_layout,
    };
    generator.emit();
    generator.builder.add_dynamic_smem(needed_smem);
    (generator.builder, generator.tile_layout)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLayout {
    a: Layout,
    b: Layout,
    c: Layout,
}

impl TileLayout {
    pub fn m(&self) -> u32 {
        self.a.nth_child(0).size()
    }

    pub fn n(&self) -> u32 {
        self.b.nth_child(0).size()
    }

    pub fn k(&self) -> u32 {
        self.a.nth_child(1).size()
    }

    // Use of LDSM instructions (ldmatrix in PTX)
    // requires one of the following conditions:
    // 1) The SMEM layout matches the MMA atom layout
    // for the matrix, e.g. both are row-major or both are column-major.
    // 2) The SMEM layout does not match the MMA atom layout,
    // but the data type is 16 bits in size, allowing the use
    // of ldmatrix.trans to efficiently transpose on the fly.
    //
    // All the MMA atoms expect A in "row-major" format and B in "column-major"
    // format (to understand what this means, see the PTX ISA docs).

    pub fn supports_ldsm_a(&self, matmul: &MatmulGenerator) -> bool {
        let a_row_major = self.a.nth_child(1).unwrap_single().stride == 1;
        let mma_atom = MmaAtom::new(matmul.matmul_op.precision);
        mma_atom.supports_ldsm()
            && (a_row_major || cpp_sizeof(matmul.load_a_dtype) == 2)
            && mma_atom.cpp_data_type_a() == cpp_data_type(matmul.load_a_dtype)
    }

    pub fn requires_ldsm_transpose_a(&self) -> bool {
        let a_row_major = self.a.nth_child(1).unwrap_single().stride == 1;
        !a_row_major
    }

    pub fn supports_ldsm_b(&self, matmul: &MatmulGenerator) -> bool {
        let b_row_major = self.b.nth_child(0).unwrap_single().stride == 1;

        let mma_atom = MmaAtom::new(matmul.matmul_op.precision);
        mma_atom.supports_ldsm()
            && (!b_row_major || cpp_sizeof(matmul.load_b_dtype) == 2)
            && mma_atom.cpp_data_type_b() == cpp_data_type(matmul.load_b_dtype)
    }

    pub fn requires_ldsm_transpose_b(&self) -> bool {
        let b_row_major = self.b.nth_child(0).unwrap_single().stride == 1;
        b_row_major
    }
}

fn find_tile_layout(matmul: &MatmulGenerator, architecture: Architecture) -> (TileLayout, u32) {
    let compute_smem_usage = |tile_layout: &TileLayout| {
        let mainloop_usage =
            tile_layout.a.cosize() * MAINLOOP_TILE_SLOTS * cpp_sizeof(matmul.load_a_dtype)
                + tile_layout.b.cosize() * MAINLOOP_TILE_SLOTS * cpp_sizeof(matmul.load_b_dtype);
        let epilogue_usage = tile_layout.c.cosize() * cpp_sizeof(matmul.accumulator_dtype);
        mainloop_usage.max(epilogue_usage)
    };

    #[derive(Debug)]
    struct CandidateLayout {
        tile_layout: TileLayout,
        mn_product: u32,
        k: u32,
        copy_pattern_g2s_a: CopyPattern,
        copy_pattern_g2s_b: CopyPattern,
        can_use_ldsm_a: bool,
        can_use_ldsm_b: bool,
    }

    impl CandidateLayout {
        pub fn is_better_than(&self, other: &CandidateLayout) -> bool {
            // Prefer tile layouts that allow at least 32-bit copy vectorization
            // (to get good coalescing and to allow cp.async instructions for pipelining).
            // Higher vectorization than 32 bits is nice to have, but having
            // a larger tile size is more important, so we don't account for higher vectorizations.
            let mut better_vectorization = 0i32;
            for (a, b) in [
                (&self.copy_pattern_g2s_a, &other.copy_pattern_g2s_a),
                (&self.copy_pattern_g2s_b, &other.copy_pattern_g2s_b),
            ] {
                if a.vectorization_type < CopyVectorizationType::Uint32
                    || b.vectorization_type < CopyVectorizationType::Uint32
                {
                    match a.vectorization_type.cmp(&b.vectorization_type) {
                        Ordering::Less => better_vectorization -= 1,
                        Ordering::Greater => better_vectorization += 1,
                        Ordering::Equal => {}
                    }
                }
            }
            if better_vectorization > 0 {
                return true;
            }

            let better_ldsm = (self.can_use_ldsm_a as u32 + self.can_use_ldsm_b as u32)
                > (other.can_use_ldsm_a as u32 + other.can_use_ldsm_b as u32);
            if better_ldsm {
                return true;
            }

            if self.mn_product > other.mn_product {
                return true;
            }

            self.k > other.k
        }
    }

    let mut best_layout: Option<CandidateLayout> = None;

    // Bank conflict avoidance padding.
    // To enable use of ldsm instructions,
    // rows or columns need to be aligned to 16 bytes,
    // thus the higher than usual amount of padding.
    let padding_a = 16 / cpp_sizeof(matmul.load_a_dtype);
    let padding_b = 16 / cpp_sizeof(matmul.load_b_dtype);
    let padding_c = 16 / cpp_sizeof(matmul.accumulator_dtype);

    for (m, n, k) in [
        (256, 128, 32),
        (128, 256, 32),
        (256, 128, 64),
        (128, 256, 64),
        (128, 128, 32),
        (128, 128, 64),
    ] {
        let tile_layout_c = Layout::from_sizes_and_strides([(m, n + padding_c), (n, 1)]);
        let candidate_layouts_a = [
            Layout::from_sizes_and_strides([(m, k + padding_a), (k, 1)]),
            Layout::from_sizes_and_strides([(m, 1), (k, m + padding_a)]),
        ];
        let candidate_layouts_b = [
            Layout::from_sizes_and_strides([(n, k + padding_b), (k, 1)]),
            Layout::from_sizes_and_strides([(n, 1), (k, n + padding_b)]),
        ];
        for tile_layout_a in &candidate_layouts_a {
            for tile_layout_b in &candidate_layouts_b {
                let tile_layout = TileLayout {
                    a: tile_layout_a.clone(),
                    b: tile_layout_b.clone(),
                    c: tile_layout_c.clone(),
                };
                if compute_smem_usage(&tile_layout) > architecture.max_smem_size() {
                    continue;
                }

                let can_use_ldsm_a = tile_layout.supports_ldsm_a(matmul);
                let can_use_ldsm_b = tile_layout.supports_ldsm_b(matmul);

                let candidate_layout = CandidateLayout {
                    tile_layout,
                    mn_product: m * n,
                    k,
                    copy_pattern_g2s_a: find_gmem_copy_pattern(
                        &matmul.layout_gmem_a,
                        tile_layout_a,
                        NUM_THREADS,
                        cpp_sizeof(matmul.load_a_dtype),
                    ),
                    copy_pattern_g2s_b: find_gmem_copy_pattern(
                        &matmul.layout_gmem_b,
                        tile_layout_b,
                        NUM_THREADS,
                        cpp_sizeof(matmul.load_b_dtype),
                    ),
                    can_use_ldsm_a,
                    can_use_ldsm_b,
                };

                match &best_layout {
                    Some(best) => {
                        if candidate_layout.is_better_than(best) {
                            best_layout = Some(candidate_layout);
                        }
                    }
                    None => best_layout = Some(candidate_layout),
                }
            }
        }
    }

    let layout = best_layout
        .expect("no valid tile layouts found?")
        .tile_layout;
    let needed_smem = compute_smem_usage(&layout);
    (layout, needed_smem)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum MmaAtom {
    UniversalFma,
    SM80_16x8x8_F32TF32TF32F32_TN,
    SM80_16x8x16_F32BF16BF16F32_TN,
    SM80_16x8x16_F32F16F16F32_TN,
    SM80_16x8x16_F16F16F16F16_TN,
    SM89_16x8x32_F32E4M3E4M3F32_TN,
    SM89_16x8x32_F32E4M3E5M2F32_TN,
    SM89_16x8x32_F32E5M2E4M3F32_TN,
    SM89_16x8x32_F32E5M2E5M2F32_TN,
}

impl MmaAtom {
    pub fn new(precision: Precision) -> Self {
        match precision {
            Precision::MulF32AccumF32 => MmaAtom::UniversalFma,
            Precision::MulTf32AccumF32 => MmaAtom::SM80_16x8x8_F32TF32TF32F32_TN,
            Precision::MulBf16AccumF32 => MmaAtom::SM80_16x8x16_F32BF16BF16F32_TN,
            Precision::MulF16AccumF32 => MmaAtom::SM80_16x8x16_F32F16F16F32_TN,
            Precision::MulF16AccumF16 => MmaAtom::SM80_16x8x16_F16F16F16F16_TN,
            Precision::MulF8AccumF32 {
                mode_a: F8Mode::E4M3,
                mode_b: F8Mode::E4M3,
            } => MmaAtom::SM89_16x8x32_F32E4M3E4M3F32_TN,
            Precision::MulF8AccumF32 {
                mode_a: F8Mode::E4M3,
                mode_b: F8Mode::E5M2,
            } => MmaAtom::SM89_16x8x32_F32E4M3E5M2F32_TN,
            Precision::MulF8AccumF32 {
                mode_a: F8Mode::E5M2,
                mode_b: F8Mode::E4M3,
            } => MmaAtom::SM89_16x8x32_F32E5M2E4M3F32_TN,
            Precision::MulF8AccumF32 {
                mode_a: F8Mode::E5M2,
                mode_b: F8Mode::E5M2,
            } => MmaAtom::SM89_16x8x32_F32E5M2E5M2F32_TN,
            Precision::MulF8AccumF16 { .. } => todo!(), // not exposed in CUTE, probably should submit PR
        }
    }

    pub fn cpp_data_type_a(&self) -> &'static str {
        match self {
            MmaAtom::UniversalFma => "float",
            MmaAtom::SM80_16x8x8_F32TF32TF32F32_TN => "tfloat32_t",
            MmaAtom::SM80_16x8x16_F32BF16BF16F32_TN => "bfloat16_t",
            MmaAtom::SM80_16x8x16_F32F16F16F32_TN => "half_t",
            MmaAtom::SM80_16x8x16_F16F16F16F16_TN => "half_t",
            MmaAtom::SM89_16x8x32_F32E4M3E4M3F32_TN => "float_e4m3_t",
            MmaAtom::SM89_16x8x32_F32E4M3E5M2F32_TN => "float_e4m3_t",
            MmaAtom::SM89_16x8x32_F32E5M2E4M3F32_TN => "float_e5m2_t",
            MmaAtom::SM89_16x8x32_F32E5M2E5M2F32_TN => "float_e5m2_t",
        }
    }

    pub fn cpp_data_type_b(&self) -> &'static str {
        match self {
            MmaAtom::UniversalFma => "float",
            MmaAtom::SM80_16x8x8_F32TF32TF32F32_TN => "tfloat32_t",
            MmaAtom::SM80_16x8x16_F32BF16BF16F32_TN => "bfloat16_t",
            MmaAtom::SM80_16x8x16_F32F16F16F32_TN => "half_t",
            MmaAtom::SM80_16x8x16_F16F16F16F16_TN => "half_t",
            MmaAtom::SM89_16x8x32_F32E4M3E4M3F32_TN => "float_e4m3_t",
            MmaAtom::SM89_16x8x32_F32E4M3E5M2F32_TN => "float_e5m2_t",
            MmaAtom::SM89_16x8x32_F32E5M2E4M3F32_TN => "float_e4m3_t",
            MmaAtom::SM89_16x8x32_F32E5M2E5M2F32_TN => "float_e5m2_t",
        }
    }

    pub fn supports_ldsm(&self) -> bool {
        *self != MmaAtom::UniversalFma
    }
}

impl Display for MmaAtom {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MmaAtom::UniversalFma => write!(f, "UniversalFMA<float>"),
            _ => write!(f, "{self:?}"),
        }
    }
}

impl Sm80MatmulGenerator<'_> {
    pub fn emit(&mut self) {
        self.builder
            .add_section("includes")
            .emit(include_str!("../../../templates/matmul/sm80_includes.cu"));
        self.builder
            .add_section("template")
            .emit(include_str!("../../../templates/matmul/sm80.cu"));
        self.builder.add_section("mainloop_fusion");
        self.builder.add_section("epilogue");
        self.builder.add_section("kernel");

        self.emit_mainloop_fusion();
        self.emit_epilogue();
        self.emit_kernel_entrypoint();
    }

    fn emit_mainloop_fusion(&mut self) {
        let sym_input_a = self.builder.new_symbol();
        let sym_input_b = self.builder.new_symbol();

        let mut pointwise_ctx_a = PointwiseContext::default();
        pointwise_ctx_a.insert(self.matmul.input_a_root, sym_input_a, DataClass::Float);
        let mut code_a = self.builder.dangling_section();
        let sym_result_a = pointwise_ctx_a.emit(
            &self.matmul.op_subgraph,
            self.matmul.matmul_op.input_a,
            &mut code_a,
        );

        let mut pointwise_ctx_b = PointwiseContext::default();
        pointwise_ctx_b.insert(self.matmul.input_b_root, sym_input_b, DataClass::Float);
        let mut code_b = self.builder.dangling_section();
        let sym_result_b = pointwise_ctx_b.emit(
            &self.matmul.op_subgraph,
            self.matmul.matmul_op.input_b,
            &mut code_b,
        );

        self.builder.section("mainloop_fusion").emit(formatdoc! {r#"
            struct MainloopFusion {{
                template <typename TensorA>
                __device__ void apply_thr_a(TensorA &tensor_a) {{
                    transform(tensor_a, [](auto x) {{
                        float {sym_input_a} = static_cast<float>(x);
                        {code_a}
                        return static_cast<decltype(x)>({sym_result_a});
                    }});
                }}

                template <typename TensorB>
                __device__ void apply_thr_b(TensorB &tensor_b) {{
                     transform(tensor_b, [](auto x) {{
                        float {sym_input_b} = static_cast<float>(x);
                        {code_b}
                        return static_cast<decltype(x)>({sym_result_b});
                    }});
                }}
            }};
        "#});
    }

    fn emit_epilogue(&mut self) {
        let mut code = self.builder.dangling_section();
        let mut fields = self.builder.dangling_section();
        for (leaf_index, leaf) in self.matmul.leafs.iter().enumerate() {
            let mut pointwise_code = self.builder.dangling_section();
            let mut pointwise_ctx = PointwiseContext::default();
            let input_sym = pointwise_code.new_symbol();
            pointwise_ctx.insert(self.matmul.matmul_node, input_sym, DataClass::Float);
            let output_sym =
                pointwise_ctx.emit(&self.matmul.op_subgraph, leaf.node, &mut pointwise_code);

            assert_ne!(
                leaf.output_dtype,
                DataType::Bool,
                "bool output not yet supported"
            );
            let copy_pattern = find_gmem_copy_pattern(
                &leaf.output_mapping,
                &self.tile_layout.c,
                NUM_THREADS,
                cpp_sizeof(leaf.output_dtype),
            );

            let layout_gmem_out = &leaf.output_mapping;
            let out_type = cpp_data_type(leaf.output_dtype);

            fields.emit(formatdoc! { r#"
                {out_type} *leaf{leaf_index};
            "#});
            code.emit(formatdoc! {r#"
                {{
                    const auto coord_c = make_coord(blockIdx.x, blockIdx.y);
                    auto layout_gmem_out = {layout_gmem_out}{{}};
                    auto gmem_out = make_tensor(leaf{leaf_index}, layout_gmem_out);
                    auto tile_gmem_out = local_tile(gmem_out, make_shape(size<0>(TileSize{{}}), size<1>(TileSize{{}})), coord_c);

                    auto tiled_copy = make_tiled_copy(Copy_Atom<UniversalCopy<{}>, {}>{{}},
                        {}{{}},
                        {}{{}});
                    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
                    auto thr_smem_c = thr_copy.partition_S(tensor_c);
                    auto thr_reg_c = make_fragment_like(thr_smem_c);
                    auto thr_gmem_out = thr_copy.partition_D(tile_gmem_out);
                    copy(thr_smem_c, thr_reg_c);
                    
                    auto thr_reg_c_converted = make_tensor<{}>(thr_reg_c.layout());
                    transform(thr_reg_c, thr_reg_c_converted, [](auto x) {{
                        float {input_sym} = static_cast<float>(x);
                        {pointwise_code}
                        return static_cast<{}>({output_sym});
                    }});

                    // Check if we need to predicate the copy
                     const auto end_m = (blockIdx.x + 1) * size<0>(TileSize{{}});
                     const auto end_n = (blockIdx.y + 1) * size<1>(TileSize{{}});
                     if (end_m <= size<0>(layout_gmem_out) && end_n <= size<1>(layout_gmem_out)) {{
                        copy(tiled_copy, thr_reg_c_converted, thr_gmem_out);
                     }} else {{
                       auto dummy = make_identity_tensor(
                           make_shape(size<0>(layout_gmem_out), size<1>(layout_gmem_out)));
                       auto tile_dummy = local_tile(
                           dummy, make_shape(size<0>(TileSize{{}}), size<1>(TileSize{{}})), coord_c);
                       auto thr_dummy = thr_copy.partition_D(tile_dummy);
                
                       auto thr_pred = make_tensor<bool>(thr_reg_c_converted.layout());
                       for (int i = 0; i < size(thr_pred); i++) {{
                         const auto coord = thr_dummy(i);
                         thr_pred(i) = get<0>(coord) < size<0>(layout_gmem_out) &&
                                       get<1>(coord) < size<1>(layout_gmem_out);
                       }}

                       copy_if(thr_pred, thr_reg_c_converted, thr_gmem_out);
                     }}
                }}
                "#,
                copy_pattern.vectorization_type,
                cpp_data_type(leaf.output_dtype),
                copy_pattern.thread_layout,
                copy_pattern.value_layout,
                cpp_data_type(leaf.output_dtype),
                cpp_data_type(leaf.output_dtype),
            });
        }

        self.builder.section("epilogue").emit(formatdoc! {r#"
            template<typename TileSize>
            struct Epilogue {{
                {fields}
                template <typename TensorC>
                __device__ void apply_tile_c(TensorC tensor_c) {{
                    using LayoutSmemC = decltype(tensor_c.layout());
                    {code}
                }}
            }};
        "#});
    }

    fn emit_kernel_entrypoint(&mut self) {
        let in_dtype_a = cpp_data_type(self.matmul.load_a_dtype);
        let in_dtype_b = cpp_data_type(self.matmul.load_b_dtype);

        let mut leaf_out_args = self.builder.dangling_section();
        let mut epilogue_args = self.builder.dangling_section();
        for (leaf_index, leaf) in self.matmul.leafs.iter().enumerate() {
            let dtype = cpp_data_type(leaf.output_dtype);
            leaf_out_args.emit(format!("{dtype} *leaf{leaf_index}"));
            if leaf_index != self.matmul.leafs.len() - 1 {
                leaf_out_args.emit(", ");
            }

            epilogue_args.emit(format!(".leaf{leaf_index} = leaf{leaf_index}"));
            if leaf_index != self.matmul.leafs.len() - 1 {
                epilogue_args.emit(", ");
            }
        }

        let CopyPattern {
            thread_layout: copy_g2s_a_thr_layout,
            value_layout: copy_g2s_a_val_layout,
            vectorization_type: copy_g2s_a_vector,
        } = find_gmem_copy_pattern(
            &self.matmul.layout_gmem_a,
            &self.tile_layout.a,
            NUM_THREADS,
            cpp_sizeof(self.matmul.load_a_dtype),
        );
        let copy_g2s_a_atom = if copy_g2s_a_vector == CopyVectorizationType::Uint128 {
            "SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>".to_string()
        } else if copy_g2s_a_vector >= CopyVectorizationType::Uint32 {
            format!("SM80_CP_ASYNC_CACHEALWAYS<{copy_g2s_a_vector}>")
        } else {
            format!("UniversalCopy<{copy_g2s_a_vector}>")
        };

        let CopyPattern {
            thread_layout: copy_g2s_b_thr_layout,
            value_layout: copy_g2s_b_val_layout,
            vectorization_type: copy_g2s_b_vector,
        } = find_gmem_copy_pattern(
            &self.matmul.layout_gmem_b,
            &self.tile_layout.b,
            NUM_THREADS,
            cpp_sizeof(self.matmul.load_b_dtype),
        );
        let copy_g2s_b_atom = if copy_g2s_b_vector == CopyVectorizationType::Uint128 {
            "SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>".to_string()
        } else if copy_g2s_b_vector >= CopyVectorizationType::Uint32 {
            format!("SM80_CP_ASYNC_CACHEALWAYS<{copy_g2s_b_vector}>")
        } else {
            format!("UniversalCopy<{copy_g2s_b_vector}>")
        };

        let mma_atom = MmaAtom::new(self.matmul.matmul_op.precision);

        let m = self.tile_layout.m();
        let n = self.tile_layout.n();
        let k = self.tile_layout.k();

        let copy_atom_s2r_a = if self.tile_layout.supports_ldsm_a(self.matmul) {
            if self.tile_layout.requires_ldsm_transpose_a() {
                "SM75_U16x8_LDSM_T".into()
            } else {
                "SM75_U32x4_LDSM_N".into()
            }
        } else {
            format!(
                "UniversalCopy<{}>",
                cpp_raw_byte_type(cpp_sizeof(self.matmul.load_a_dtype))
            )
        };
        let copy_atom_s2r_b = if self.tile_layout.supports_ldsm_b(self.matmul) {
            if self.tile_layout.requires_ldsm_transpose_b() {
                "SM75_U16x8_LDSM_T".into()
            } else {
                "SM75_U32x4_LDSM_N".into()
            }
        } else {
            format!(
                "UniversalCopy<{}>",
                cpp_raw_byte_type(cpp_sizeof(self.matmul.load_b_dtype))
            )
        };

        let specialized_copy_a = self.tile_layout.supports_ldsm_a(self.matmul);
        let specialized_copy_b = self.tile_layout.supports_ldsm_b(self.matmul);

        let layout_gmem_a = &self.matmul.layout_gmem_a;
        let layout_gmem_b = &self.matmul.layout_gmem_b;

        let layout_smem_a = &self.tile_layout.a;
        let layout_smem_b = &self.tile_layout.b;
        let layout_smem_c = &self.tile_layout.c;

        let gemm_dtype_a = mma_atom.cpp_data_type_a();
        let gemm_dtype_b = mma_atom.cpp_data_type_b();
        let dtype_accum = cpp_data_type(self.matmul.matmul_op.precision.accumulator_type());

        self.builder.section("kernel").emit(formatdoc! {r#"
            extern "C" __global__ __launch_bounds__(256) void matmul({in_dtype_a} *a, {in_dtype_b} *b, {leaf_out_args}) {{
                using TileSize = Shape<Int<{m}>, Int<{n}>, Int<{k}>>;
            
                auto tiled_copy_g2s_a = make_tiled_copy(Copy_Atom<{copy_g2s_a_atom}, {in_dtype_a}>{{}}, {copy_g2s_a_thr_layout}{{}}, {copy_g2s_a_val_layout}{{}});
                auto tiled_copy_g2s_b = make_tiled_copy(Copy_Atom<{copy_g2s_b_atom}, {in_dtype_b}>{{}}, {copy_g2s_b_thr_layout}{{}}, {copy_g2s_b_val_layout}{{}});
                
                auto tiled_mma = make_tiled_mma({mma_atom}{{}}, Layout<Shape<_2, _4>>{{}}, TileSize{{}});
                
                auto tiled_copy_s2r_a = make_tiled_copy_A(Copy_Atom<{copy_atom_s2r_a}, {in_dtype_a}>{{}}, tiled_mma);
                auto tiled_copy_s2r_b = make_tiled_copy_B(Copy_Atom<{copy_atom_s2r_b}, {in_dtype_b}>{{}}, tiled_mma);
                
                using MatmulInstance = Matmul<{in_dtype_a}, {in_dtype_b}, {gemm_dtype_a}, {gemm_dtype_b}, {dtype_accum},
                    {dtype_accum}, {layout_gmem_a}, {layout_gmem_b}, MainloopFusion, Epilogue<TileSize>,
                    TileSize, {layout_smem_a}, {layout_smem_b}, {layout_smem_c}, decltype(tiled_copy_g2s_a),
                    decltype(tiled_copy_g2s_b), decltype(tiled_copy_s2r_a), decltype(tiled_copy_s2r_b),
                    decltype(tiled_mma), {PIPELINE}, {specialized_copy_a}, {specialized_copy_b}>;
                    
                extern __shared__ char shared_storage_raw[];
                
                auto shared_storage = reinterpret_cast<typename MatmulInstance::SharedStorage *>(shared_storage_raw);
                
                auto matmul = MatmulInstance {{
                    ._layout_gmem_a = {{}},
                    ._layout_gmem_b = {{}},
                    ._mainloop_fusion = MainloopFusion{{}},
                    ._epilogue = Epilogue<TileSize>{{ {epilogue_args} }},
                    ._tile_size = {{}},
                    ._tiled_copy_g2s_a = tiled_copy_g2s_a,
                    ._tiled_copy_g2s_b = tiled_copy_g2s_b,
                    ._tiled_copy_s2r_a = tiled_copy_s2r_a,
                    ._tiled_copy_s2r_b = tiled_copy_s2r_b,
                    ._tiled_mma = tiled_mma,
                    ._raw_gmem_a = a,
                    ._raw_gmem_b = b,
                    ._shared_storage = shared_storage}};
                matmul.run();
            }}
        "#});
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helium_ir::{
        data_type::DataType,
        opgraph::{
            Descriptor, OpGraph,
            op::{Matmul, Op},
            subgraph::OpSubgraph,
        },
        shape::Shape,
    };
    use std::sync::Arc;

    #[test]
    fn best_tile_layout_gmem_row_major() {
        let mut graph = OpGraph::new();
        let a = graph.new_input(Descriptor {
            shape: Shape::new([256, 256]),
            data_type: DataType::F16,
        });
        let b = graph.new_input(Descriptor {
            shape: Shape::new([256, 256]),
            data_type: DataType::F32, // should prevent ldsm.transpose from working
        });
        let c = graph.new_op(Op::Matmul(Matmul {
            input_a: a,
            input_b: b,
            precision: Precision::MulF16AccumF32,
        }));
        graph.new_output(c);

        let matmul =
            MatmulGenerator::new(&OpSubgraph::from_nodes(&Arc::new(graph), vec![c])).unwrap();
        let tile_layout = find_tile_layout(&matmul, Architecture::Sm80a).0;

        assert_eq!(
            tile_layout,
            TileLayout {
                a: Layout::from_sizes_and_strides([(128, 72), (64, 1)]),
                b: Layout::from_sizes_and_strides([(128, 68), (64, 1)]),
                c: Layout::from_sizes_and_strides([(128, 132), (128, 1)]),
            }
        );
    }
}
