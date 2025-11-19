use crate::{
    cute::Layout,
    opgraph::{BinaryPointwiseOp, Op, OpGraph, OpNode, ReductionDepth, UnaryPointwiseOp},
    util::kernel_builder::{KernelBuilder, Section, Symbol},
};
use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use indoc::formatdoc;
use std::{collections::BTreeMap, sync::Arc};

pub struct KernelGenerator {
    graph: Arc<OpGraph>,
    anchor: Option<OpNode>,
    builder: KernelBuilder,
    mainloop: Section,
    epilogue: Section,

    input_syms: BTreeMap<OpNode, Symbol>,
    output_syms: BTreeMap<OpNode, Symbol>,
}

impl KernelGenerator {
    pub fn new(opgraph: &Arc<OpGraph>) -> Self {
        let anchor = opgraph.nodes().find(|n| opgraph.get(*n).is_anchor());
        let mut builder = KernelBuilder::new(if anchor.is_some() {
            "fused_matmul"
        } else {
            "fused_auxiliary"
        });

        let mut input_syms = BTreeMap::new();
        for input in opgraph.roots() {
            input_syms.insert(input, builder.new_symbol());
        }

        let mut output_syms = BTreeMap::new();
        for output in opgraph.leaves() {
            output_syms.insert(output, builder.new_symbol());
        }

        Self {
            graph: opgraph.clone(),
            anchor,
            epilogue: builder.dangling_section(),
            mainloop: builder.dangling_section(),
            builder,
            input_syms,
            output_syms,
        }
    }

    pub fn generate(&mut self, kernel_id: &str) -> String {
        let mut args = Vec::new();
        let mut args_passed = Vec::new();
        for input_sym in self.input_syms.values() {
            args.push(format!("const bfloat16_t *{input_sym}"));
            args_passed.push(input_sym.to_string());
        }
        for output_sym in self.output_syms.values() {
            args.push(format!("bfloat16_t *{output_sym}"));
            args_passed.push(output_sym.to_string());
        }
        let args = args.join(", ");
        let args_passed = args_passed.join(", ");

        self.synthesize_epilogue_leaves();

        match self.anchor {
            Some(anchor) => {
                let Op::Matmul { a, b } = self.graph.get(anchor) else {
                    unreachable!()
                };
                let m = self.graph.shape(*a).dim(1);
                let n = self.graph.shape(*b).dim(2);
                let k = self.graph.shape(*a).dim(2);

                let template = include_str!("templates/matmul.cu")
                    .replace("{{ARGS}}", &args)
                    .replace("{{MAINLOOP}}", &self.mainloop.to_string())
                    .replace("{{EPILOGUE}}", &self.epilogue.to_string())
                    .replace(
                        "{{SYM_MATMUL_A}}",
                        &self.input_syms[&find_largest_root(&self.graph, *a)].to_string(),
                    )
                    .replace(
                        "{{SYM_MATMUL_B}}",
                        &self.input_syms[&find_largest_root(&self.graph, *b)].to_string(),
                    )
                    .replace("{{ARGS_PASSED}}", &args_passed)
                    .replace("{{KERNEL_ID}}", kernel_id)
                    .replace("{{M}}", &m.to_string())
                    .replace("{{N}}", &n.to_string())
                    .replace("{{K}}", &k.to_string());
                self.builder.add_section("main").emit(template);
            }
            None => {
                let pseudo_anchor = self
                    .graph
                    .roots()
                    .max_by_key(|r| self.graph.shape(*r).element_count())
                    .unwrap();

                let m = self.graph.shape(pseudo_anchor).dim(1);
                let n = self.graph.shape(pseudo_anchor).dim(2);

                let template = include_str!("templates/auxiliary.cu")
                    .replace("{{ARGS}}", &args)
                    .replace("{{EPILOGUE}}", &self.epilogue.to_string())
                    .replace("{{ARGS_PASSED}}", &args_passed)
                    .replace("{{KERNEL_ID}}", kernel_id)
                    .replace("{{M}}", &m.to_string())
                    .replace("{{N}}", &n.to_string());
                self.builder.add_section("main").emit(template);
            }
        }

        self.builder.build_source()
    }

    pub fn generate_header(&self, kernel_id: &str) -> String {
        let mut args = Vec::new();
        for input_sym in self.input_syms.values() {
            args.push(format!("const cutlass::bfloat16_t *{input_sym}"));
        }
        for output_sym in self.output_syms.values() {
            args.push(format!("cutlass::bfloat16_t *{output_sym}"));
        }
        let args = args.join(", ");

        let mut builder = KernelBuilder::new("header");
        builder.add_section("main").emit(formatdoc! {"
        #include <cutlass/bfloat16.h>
        void {kernel_id}({args}, cudaStream_t stream = 0);
        "});

        builder.build_source()
    }

    pub fn args_passed(&self) -> impl Iterator<Item = OpNode> {
        self.input_syms
            .keys()
            .copied()
            .chain(self.output_syms.keys().copied().map(|node| {
                let Op::Consumer { x } = self.graph.get(node) else {
                    panic!()
                };
                *x
            }))
    }

    fn synthesize_epilogue_leaves(&mut self) {
        for leaf in self.graph.clone().leaves().take(1) {
            self.synthesize_epilogue_leaf(leaf);
        }
    }

    fn synthesize_epilogue_leaf(&mut self, leaf: OpNode) {
        if self.graph.get(leaf).is_reduction() {
            self.synthesize_reduction_epilogue_leaf(leaf, leaf);
        } else if let Op::Consumer { x } = self.graph.get(leaf)
            && self.graph.get(*x).is_reduction()
        {
            self.synthesize_reduction_epilogue_leaf(*x, leaf);
        } else {
            self.synthesize_non_reduction_epilogue_leaf(leaf);
        }
    }

    fn synthesize_non_reduction_epilogue_leaf(&mut self, leaf: OpNode) {
        let (mut section, out_sym) = self.synthesize_intermediate_calculations(leaf);

        let data_sym = self.output_syms[&leaf];
        let temp_sym = self.builder.new_symbol();
        let layout = Layout::new_row_major(self.graph.shape(leaf).dims());

        if self.anchor.is_some() {
            section.emit(formatdoc! {"
                auto {temp_sym} = local_tile(make_tensor(make_gmem_ptr({data_sym}), {layout}{{}})(blockIdx.z, _, _), Shape<_32, _256>{{}}, make_coord(blockIdx.x * 4 + i, blockIdx.y));
                if (threadIdx.x < 256) {{
                    copy({out_sym}, epilogue_partitioner.partition_D({temp_sym}));
                }}
        "});
            self.epilogue.emit(formatdoc! {"
                for (int i = 0; i < 4; i++) {{
                    auto epilogue_slicer = make_tiled_copy(Copy_Atom<UniversalCopy<uint32_t>, bfloat16_t>{{}},
                        Layout<Shape<_2, _128>, Stride<_128, _1>>{{}},
                        Layout<Shape<_16, _2>, Stride<_2, _1>>{{}});
                    auto epilogue_partitioner = epilogue_slicer.get_slice(threadIdx.x);
                    auto sC = make_tensor(
                          make_smem_ptr(reinterpret_cast<bfloat16_t *>(epilogue_smem->out_tile) + i * 256 * 32),
                          Layout<Shape<_32, _256>, Stride<Int<256>, _1>>{{}});
                    auto dummy = make_identity_tensor(Shape<_32, _256>{{}});
                    auto thread_c = epilogue_partitioner.partition_D(sC);
                    auto coord_slice = epilogue_partitioner.partition_D(dummy);
                    {section}
                }}
            "});
        } else {
            // predicated
            section.emit(formatdoc! {"
            auto {temp_sym} = local_tile(make_tensor(make_gmem_ptr({data_sym}), {layout}{{}})(blockIdx.z, _, _), Shape<_128, _256>{{}}, make_coord(blockIdx.x, blockIdx.y));
            if (threadIdx.x < 256) {{
                copy_if(thread_pred, {out_sym}, epilogue_partitioner.partition_D({temp_sym}));
            }}
        "});
            self.epilogue.emit(section.to_string());
        }
    }

    fn synthesize_reduction_epilogue_leaf(&mut self, leaf: OpNode, consumer: OpNode) {
        let (mut section, out_sym) = self.synthesize_intermediate_calculations(leaf);

        let data_sym = self.output_syms[&consumer];
        section.emit(formatdoc! {"
            if (threadIdx.x < 256) {{
                copy({out_sym}, thread_pre_reduction);
            }}
        "});

        let Op::Reduction { depth, .. } = self.graph.get(leaf) else {
            unreachable!()
        };
        let template = match *depth {
            ReductionDepth::Rows => include_str!("templates/epilogue_reduction_row.cu"),
            ReductionDepth::Columns => include_str!("templates/epilogue_reduction_col.cu"),
            ReductionDepth::Item => include_str!("templates/epilogue_reduction_item.cu"),
            ReductionDepth::Batch => include_str!("templates/epilogue_reduction_batch.cu"),
        };
        let template = template.replace("{{OUT_DATA}}", &data_sym.to_string());

        self.epilogue.emit(formatdoc! {"
        __syncthreads(); // wait for any previous reductions....
        for (int i = 0; i < 4; i++) {{
               auto epilogue_slicer = make_tiled_copy(Copy_Atom<UniversalCopy<uint32_t>, bfloat16_t>{{}},
                        Layout<Shape<_2, _128>, Stride<_128, _1>>{{}},
                        Layout<Shape<_16, _2>, Stride<_2, _1>>{{}});
                auto epilogue_partitioner = epilogue_slicer.get_slice(threadIdx.x);
                auto sC = make_tensor(
                      make_smem_ptr(reinterpret_cast<bfloat16_t *>(epilogue_smem->out_tile) + i * 256 * 32),
                      Layout<Shape<_32, _256>, Stride<Int<256>, _1>>{{}});
                auto dummy = make_identity_tensor(Shape<_32, _256>{{}});
                auto thread_c = epilogue_partitioner.partition_D(sC);
                auto coord_slice = epilogue_partitioner.partition_D(dummy);
              auto sPreReduction = make_tensor(
                  make_smem_ptr(reinterpret_cast<bfloat16_t *>(epilogue_smem->pre_reduction_scratch) + i * 258 * 32),
                  Layout<Shape<_32, _256>, Stride<Int<258>, _1>>{{}});
            auto thread_pre_reduction = epilogue_partitioner.partition_D(sPreReduction);
            {section}
        }}

        __syncthreads();
        {template}
        "});
    }

    fn synthesize_intermediate_calculations(&mut self, leaf: OpNode) -> (Section, Symbol) {
        let mut section = self.builder.dangling_section();
        let out_sym = self.builder.new_symbol();
        let mut temp_section = self.builder.dangling_section();
        let anchor_sym = self.builder.new_symbol();
        let result_sym = self.op_node_in_reg(
            &mut HashMap::new(),
            leaf,
            &mut temp_section,
            &mut Vec::new(),
            anchor_sym,
        );
        if self.anchor.is_some() {
            // unpredicated
            section.emit(formatdoc! {"
                auto {out_sym} = make_fragment_like(thread_c);
                if (threadIdx.x < 256) {{
                    transform(thread_c, coord_slice, {out_sym}, [&](bfloat16_t c, auto coord) {{
                        __nv_bfloat16 {anchor_sym} = static_cast<__nv_bfloat16>(c);
                        {temp_section}
                        return static_cast<bfloat16_t>({result_sym});
                    }});
                }}
            "});
        } else {
            // predicated
            section.emit(formatdoc! {"
                auto {out_sym} = make_fragment_like(thread_c);
                if (threadIdx.x < 256) {{
                    transform(thread_pred, coord_slice, {out_sym}, [&](bool pred, auto coord) {{
                        if (pred) {{
                            {temp_section}
                            return static_cast<bfloat16_t>({result_sym});
                        }} else {{
                            return static_cast<bfloat16_t>(0.0f);
                        }}
                    }});
                }}
            "});
        }
        (section, out_sym)
    }

    fn op_node_in_reg(
        &mut self,
        cx: &mut HashMap<OpNode, Symbol>,
        node: OpNode,
        dst: &mut Section,
        broadcast: &mut Vec<(u32, u32)>,
        anchor_symbol: Symbol,
    ) -> Symbol {
        if let Some(sym) = cx.get(&node) {
            return *sym;
        }

        if Some(node) == self.anchor {
            return anchor_symbol;
        }

        let sym = match self.graph.clone().get(node) {
            Op::Producer { .. } => {
                let data_sym = self.input_syms[&node];
                let new_sym = self.builder.new_symbol();

                let mut sizes = self.graph.shape(node).dims();

                let mut strides = [0u32; 3];
                let mut stride = 1;
                for (d, s) in sizes.into_iter().rev().zip(strides.iter_mut().rev()) {
                    *s = stride;
                    stride *= d;
                }

                for (axis, amount) in broadcast {
                    sizes[*axis as usize] = *amount;
                    strides[*axis as usize] = 0;
                }

                let layout = Layout::from_sizes_and_strides(sizes.into_iter().zip(strides));

                dst.emit(format!(
                    "auto {new_sym}_tensor = local_tile(make_tensor(make_gmem_ptr({data_sym}), {layout}{{}})(blockIdx.z, _, _), Shape<_128, _256>{{}}, make_coord(blockIdx.x, blockIdx.y));"
                ));
                dst.emit(format!(
                    "auto {new_sym} = static_cast<__nv_bfloat16>({new_sym}_tensor(make_coord(blockIdx.x % 128, blockIdx.y % 256)));"
                ));

                new_sym
            }
            Op::Constant { value, .. } => {
                let sym = self.builder.new_symbol();
                dst.emit(format!(
                    "__nv_bfloat16 {sym} = __float2bfloat16({value:.10}f);"
                ));
                sym
            }
            Op::Reduction { x, .. } | Op::Transpose { x } => {
                // handled in leaf synthesis
                self.op_node_in_reg(cx, *x, dst, broadcast, anchor_symbol)
            }
            Op::Matmul { .. } => unreachable!(),
            Op::Broadcast { x, axis, amount } => {
                broadcast.push((*axis, *amount));
                let s = self.op_node_in_reg(cx, *x, dst, broadcast, anchor_symbol);
                broadcast.pop();
                s
            }
            Op::Consumer { x } => self.op_node_in_reg(cx, *x, dst, broadcast, anchor_symbol),
            Op::UnaryPointwise { x, op } => {
                let x = self.op_node_in_reg(cx, *x, dst, broadcast, anchor_symbol);
                let sym = self.builder.new_symbol();
                dst.emit(format!("__nv_bfloat16 {sym} = "));
                match *op {
                    UnaryPointwiseOp::Neg => dst.emit(format!("__hneg({x});")),
                    UnaryPointwiseOp::Relu => {
                        dst.emit(format!("__hmax({x}, __float2bfloat16(0.0f));"))
                    }
                    UnaryPointwiseOp::Gelu => todo!(),
                    UnaryPointwiseOp::Tanh => dst.emit(format!("htanh({x});")),
                    UnaryPointwiseOp::Exp => dst.emit(format!("hexp({x});")),
                    UnaryPointwiseOp::Ln => dst.emit(format!("hlog({x});")),
                    UnaryPointwiseOp::Sqrt => dst.emit(format!("hsqrt({x});")),
                    UnaryPointwiseOp::AddConstant(c) => {
                        dst.emit(format!("__hadd({x}, __float2bfloat16({c:.10}f));"))
                    }
                    UnaryPointwiseOp::MulConstant(c) => {
                        dst.emit(format!("__hmul({x}, __float2bfloat16({c:.10}f));"))
                    }
                    UnaryPointwiseOp::PowConstant(c) => {
                        if c == 2.0 {
                            dst.emit(format!("__hmul({x}, {x});"))
                        } else {
                            dst.emit(format!(
                                "hexp(__hmul({x}, __float2bfloat16({:.10}f)));",
                                c.ln()
                            ))
                        }
                    }
                };
                sym
            }
            Op::BinaryPointwise { lhs, rhs, op } => {
                let lhs = self.op_node_in_reg(cx, *lhs, dst, broadcast, anchor_symbol);
                let rhs = self.op_node_in_reg(cx, *rhs, dst, broadcast, anchor_symbol);
                let sym = self.builder.new_symbol();
                dst.emit(format!("__nv_bfloat16 {sym} = "));
                match *op {
                    BinaryPointwiseOp::Add => dst.emit(format!("__hadd({lhs}, {rhs});")),
                    BinaryPointwiseOp::Sub => dst.emit(format!("__hsub({lhs}, {rhs});")),
                    BinaryPointwiseOp::Mul => dst.emit(format!("__hmul({lhs}, {rhs});")),
                    BinaryPointwiseOp::Div => dst.emit(format!("__hdiv({lhs}, {rhs});")),
                    BinaryPointwiseOp::Pow => {
                        dst.emit(format!("hexp(__hmul({lhs}, hlog({rhs})));"))
                    }
                    BinaryPointwiseOp::Drelu => dst.emit(format!(
                        "{rhs} > __float2bfloat16(0.0f) ? {lhs} : __float2bfloat16(0.0f);"
                    )),
                };
                sym
            }
        };
        cx.insert(node, sym);
        sym
    }
}

fn find_largest_root(opgraph: &OpGraph, descendent: OpNode) -> OpNode {
    let mut stack = vec![descendent];
    let mut visited = HashSet::new();

    let mut largest = None;

    while let Some(current) = stack.pop() {
        if let Op::Producer { shape } = opgraph.get(current) {
            if largest.is_none() {
                largest = Some((current, shape.element_count()));
            } else if let Some(current_largest) = largest {
                if shape.element_count() > current_largest.1 {
                    largest = Some((current, shape.element_count()));
                }
            }
        }

        for pred in opgraph.inbound_edges(current) {
            if visited.insert(pred) {
                stack.push(pred);
            }
        }
    }

    largest.unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opgraph::{ReductionOp, Shape};

    #[test]
    fn epilogue_fusion() {
        let mut graph = OpGraph::new();
        let root = graph.insert(Op::Producer {
            shape: Shape::new([1, 512, 512]),
        });

        let anchor = graph.insert(Op::Matmul { a: root, b: root });
        let epilogue1 = graph.insert(Op::UnaryPointwise {
            x: anchor,
            op: UnaryPointwiseOp::Relu,
        });
        let epilogue2 = graph.insert(Op::BinaryPointwise {
            lhs: anchor,
            rhs: epilogue1,
            op: BinaryPointwiseOp::Add,
        });
        let epilogue3 = graph.insert(Op::Reduction {
            x: epilogue2,
            op: ReductionOp::Sum,
            depth: ReductionDepth::Rows,
        });

        let _out = graph.insert(Op::Consumer { x: epilogue2 });
        let _out2 = graph.insert(Op::Consumer { x: epilogue3 });

        let kernel = KernelGenerator::new(&Arc::new(graph)).generate("fused_relu_add");

        //std::fs::write("test.cu", &kernel).unwrap();
        drop(kernel);
    }

    #[test]
    fn aux_fusion() {
        let mut graph = OpGraph::new();
        let root = graph.insert(Op::Producer {
            shape: Shape::new([1, 512, 512]),
        });

        let epilogue1 = graph.insert(Op::UnaryPointwise {
            x: root,
            op: UnaryPointwiseOp::Relu,
        });
        let epilogue2 = graph.insert(Op::BinaryPointwise {
            lhs: root,
            rhs: epilogue1,
            op: BinaryPointwiseOp::Add,
        });
        let epilogue3 = graph.insert(Op::Reduction {
            x: epilogue2,
            op: ReductionOp::Sum,
            depth: ReductionDepth::Rows,
        });

        let _out = graph.insert(Op::Consumer { x: epilogue2 });
        let _out2 = graph.insert(Op::Consumer { x: epilogue3 });

        let kernel = KernelGenerator::new(&Arc::new(graph)).generate("fused_aux");

        //std::fs::write("test.cu", &kernel).unwrap();
        drop(kernel);
    }
}
