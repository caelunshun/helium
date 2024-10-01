use crate::{
    cuda::{
        context::CudaContext,
        error::CudaError,
        kernel::{pointwise, reduction},
        plan::{Instr, MatmulInstr, Plan, ReshapeInstr, Step, UploadTensorInstr},
    },
    opgraph::{
        op::{Op, Reduce},
        subgraph::OpSubgraph,
        Intermediate, Node, NodeId, OpGraph,
    },
};
use ahash::AHashSet;
use cudarc::cublaslt::sys::cublasLtEpilogue_t;
use slotmap::{SecondaryMap, SlotMap};
use std::{collections::VecDeque, mem, sync::Arc};

/// Compiles an `OpGraph` into a `Plan` for CUDA execution.
pub fn compile_plan(cx: &CudaContext, graph: &Arc<OpGraph>) -> Result<Plan, CudaError> {
    Planner::new(cx, graph).generate_plan()
}

struct Planner<'a> {
    cx: &'a CudaContext,

    graph: Arc<OpGraph>,

    /// Nodes whose inputs are already in the plan
    roots: Vec<NodeId>,

    indegrees: SecondaryMap<NodeId, usize>,

    /// Nodes that have been covered in the plan
    covered: AHashSet<NodeId>,

    matmul_transpose_a: SecondaryMap<NodeId, NodeId>,
    matmul_transpose_b: SecondaryMap<NodeId, NodeId>,

    instr_graph: InstrGraph,
}

impl<'a> Planner<'a> {
    pub fn new(cx: &'a CudaContext, graph: &Arc<OpGraph>) -> Self {
        let mut roots = Vec::new();
        let mut indegrees = SecondaryMap::default();
        let mut covered = AHashSet::new();

        for node in graph.nodes() {
            let indegree = graph
                .inbound_edges(node)
                .iter()
                .filter(|&&id| !matches!(graph.get(id), Node::Input(_)))
                .count();
            indegrees.insert(node, indegree);
            let is_input = matches!(graph.get(node), Node::Input(_));
            if indegree == 0 && !is_input {
                roots.push(node);
            }
            if is_input {
                covered.insert(node);
            }
        }

        Self {
            cx,
            graph: Arc::clone(graph),
            roots,
            covered,
            indegrees,
            instr_graph: InstrGraph::new(),
            matmul_transpose_a: SecondaryMap::default(),
            matmul_transpose_b: SecondaryMap::default(),
        }
    }

    pub fn generate_plan(mut self) -> Result<Plan, CudaError> {
        while let Some(root) = self.roots.pop() {
            if !self.covered.contains(&root) {
                self.cover_from_node(root)?;
            }
        }
        Ok(self.instr_graph.into_plan(&self.graph))
    }

    fn mark_covered(&mut self, node_id: NodeId) {
        self.covered.insert(node_id);
        for &next in self.graph.outbound_edges(node_id) {
            let indegree = &mut self.indegrees[next];
            *indegree = indegree.checked_sub(1).unwrap();
            if *indegree == 0 {
                self.roots.push(next);
            }
        }
    }

    /// Given a root, attempts to greedily
    /// consume more nodes that can be fused with the
    /// root. Generates instructions
    /// for those nodes.
    fn cover_from_node(&mut self, node_id: NodeId) -> Result<(), CudaError> {
        let node = self.graph.get(node_id);
        let node = match node {
            Node::Intermediate(int) => int,
            Node::Output(_) => {
                let prev = self.graph.inbound_edges(node_id)[0];
                self.instr_graph.outputs.push(prev);
                return Ok(());
            }
            Node::Input(_) => unreachable!(),
        };

        match &node.op {
            Op::UploadTensor(op) => {
                self.instr_graph
                    .insert(Instr::UploadTensor(UploadTensorInstr {
                        data_var: op.data_var,
                        data_type: node.descriptor.data_type,
                        output: node_id,
                    }));
                self.mark_covered(node_id);
            }
            Op::Matmul(op) => {
                let (a_input, transpose_a) = match self.matmul_transpose_a.get(node_id) {
                    Some(input) => (*input, true),
                    None => (op.input_a, false),
                };
                let (b_input, transpose_b) = match self.matmul_transpose_b.get(node_id) {
                    Some(input) => (*input, true),
                    None => (op.input_b, false),
                };

                self.instr_graph.insert(Instr::Matmul(MatmulInstr {
                    a_input,
                    b_input,
                    output_type: node.descriptor.data_type,
                    transpose_a,
                    transpose_b,
                    bias_input: None,
                    epilogue: cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
                    output: node_id,
                }));
                self.mark_covered(node_id);
            }
            Op::Transpose(op) => {
                for next_node in self.graph.outbound_edges(node_id) {
                    let Node::Intermediate(Intermediate {
                        op: Op::Matmul(matmul),
                        ..
                    }) = self.graph.get(*next_node)
                    else {
                        unimplemented!("transpose op without matmul")
                    };
                    if node_id == matmul.input_a {
                        self.matmul_transpose_a.insert(*next_node, op.input);
                    } else {
                        self.matmul_transpose_b.insert(*next_node, op.input);
                    }
                }
            }
            Op::UnaryPointwise(_)
            | Op::BinaryPointwise(_)
            | Op::ChangeDataType(_)
            | Op::Restructure(_) => {
                let greedy = self.cover_greedy_pointwise(node_id);
                let subgraph = OpSubgraph::from_nodes(&self.graph, greedy.covered_nodes);

                let instr = match greedy.reduction_node {
                    Some(reduction_node_id) => {
                        let kernel = self.cx.get_or_init_kernel(&subgraph, || {
                            reduction::generate_kernel(&subgraph)
                        })?;
                        let Node::Intermediate(Intermediate {
                            op:
                                Op::Reduce(Reduce {
                                    depth,
                                    op: reduce_op,
                                    ..
                                }),
                            ..
                        }) = self.graph.get(reduction_node_id)
                        else {
                            unreachable!()
                        };

                        Instr::ReductionKernel {
                            kernel,
                            reduction_depth: *depth,
                            initial_reduced_value: reduce_op.default_value(),
                            output_shape: self
                                .graph
                                .get(reduction_node_id)
                                .descriptor()
                                .shape
                                .clone(),
                        }
                    }
                    None => {
                        let kernel = self.cx.get_or_init_kernel(&subgraph, || {
                            pointwise::generate_kernel(&subgraph)
                        })?;
                        Instr::PointwiseKernel { kernel }
                    }
                };

                self.instr_graph.insert(instr);
            }
            Op::Reduce(op) => {
                // Can't fuse anything after a reduction.
                let subgraph = OpSubgraph::from_nodes(&self.graph, vec![node_id]);
                let kernel = self
                    .cx
                    .get_or_init_kernel(&subgraph, || reduction::generate_kernel(&subgraph))?;
                self.instr_graph.insert(Instr::ReductionKernel {
                    kernel,
                    reduction_depth: op.depth,
                    initial_reduced_value: op.op.default_value(),
                    output_shape: self.graph.get(node_id).descriptor().shape.clone(),
                });
                self.mark_covered(node_id);
            }
            Op::Reshape(op) => {
                self.instr_graph.insert(Instr::Reshape(ReshapeInstr {
                    input: op.input,
                    output: node_id,
                    new_shape: op.new_shape.clone(),
                }));
                self.mark_covered(node_id);
            }
        }
        Ok(())
    }

    fn cover_greedy_pointwise(&mut self, root: NodeId) -> GreedyPointwise {
        let mut nodes = Vec::new();
        let mut queue = VecDeque::new();
        let mut reduction_node = None;
        queue.push_back((root, true));
        self.mark_covered(root);
        while let Some((node, should_visit_children)) = queue.pop_front() {
            nodes.push(node);

            if !should_visit_children {
                continue;
            }

            // Heuristic to maximize parallelism
            let mut next = self.graph.outbound_edges(node).to_vec();
            next.sort_by_key(|node| {
                self.graph
                    .inbound_edges(*node)
                    .iter()
                    .filter(|&id| !self.covered.contains(id))
                    .count()
            });

            'outer: for next_node_id in next {
                // To fuse with the next node, it must
                // 1) be a pointwise operator, or be a reduction operator
                //    if `reduction_node` was previously `None`
                // 2) all its dependencies must already be covered
                //
                // Note that a reduction operator cannot be fused
                // with any following pointwise ops, so we don't
                // visit the children of a reduction.

                let next_node = self.graph.get(next_node_id);
                let is_reduction = matches!(
                    next_node,
                    Node::Intermediate(Intermediate {
                        op: Op::Reduce(_),
                        ..
                    })
                );
                let is_pointwise = matches!(next_node, Node::Intermediate(Intermediate { op, .. }) if op.is_pointwise());

                if !is_reduction && !is_pointwise {
                    continue;
                }

                if is_reduction {
                    if reduction_node.is_none() {
                        reduction_node = Some(next_node_id);
                    } else {
                        continue;
                    }
                }

                for dep in self.graph.inbound_edges(next_node_id) {
                    if !self.covered.contains(dep) {
                        continue 'outer;
                    }
                }

                self.mark_covered(next_node_id);
                queue.push_back((next_node_id, !is_reduction));
            }
        }
        GreedyPointwise {
            covered_nodes: nodes,
            reduction_node,
        }
    }
}

struct GreedyPointwise {
    covered_nodes: Vec<NodeId>,
    reduction_node: Option<NodeId>,
}

slotmap::new_key_type! {
    struct InstrId;
}

#[derive(Default)]
struct InstrGraph {
    node_to_instr: SecondaryMap<NodeId, InstrId>,
    instrs: SlotMap<InstrId, Instr>,
    inbound_edges: SecondaryMap<InstrId, Vec<InstrId>>,
    outbound_edges: SecondaryMap<InstrId, Vec<InstrId>>,
    optimized_node_dependents: SecondaryMap<NodeId, Vec<InstrId>>,
    outputs: Vec<NodeId>,
}

impl InstrGraph {
    pub fn new() -> Self {
        Self::default()
    }

    #[expect(unused)]
    pub fn get(&self, id: InstrId) -> &Instr {
        &self.instrs[id]
    }

    pub fn insert(&mut self, instr: Instr) -> InstrId {
        let dependencies = instr.dependencies();
        let outputs = instr.outputs();
        let id = self.instrs.insert(instr);
        for output in outputs {
            self.node_to_instr.insert(output, id);
        }

        for &dep in &dependencies {
            if let Some(dep) = self.node_to_instr.get(dep).copied() {
                self.outbound_edges
                    .entry(dep)
                    .unwrap()
                    .or_default()
                    .push(id);
            }

            self.optimized_node_dependents
                .entry(dep)
                .unwrap()
                .or_default()
                .push(id);
        }
        self.inbound_edges.insert(
            id,
            dependencies
                .into_iter()
                .filter_map(|d| self.node_to_instr.get(d).copied())
                .collect(),
        );
        id
    }

    /// Generates a `Plan` that aims to maximize parallelism
    /// based on this graph.
    ///
    /// This also inserts `FreeTensor` instructions as soon
    /// as particular tensors are no longer needed.
    pub fn into_plan(self, op_graph: &OpGraph) -> Plan {
        let mut plan = Plan::new();

        let mut indegrees = SecondaryMap::default();
        let mut outdegrees = SecondaryMap::default();
        let mut ready_to_run = Vec::new();
        for instr in self.instrs.keys() {
            let indegree = self
                .inbound_edges
                .get(instr)
                .map(Vec::as_slice)
                .unwrap_or_default()
                .len();
            indegrees.insert(instr, indegree);
            if indegree == 0 {
                ready_to_run.push(instr);
            }

            for output in self.instrs[instr].outputs() {
                let outdegree = self
                    .optimized_node_dependents
                    .get(output)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .len();
                outdegrees.insert(output, outdegree);
            }
        }

        for &input in op_graph.inputs() {
            outdegrees.insert(input, op_graph.outbound_edges(input).len());
        }

        let mut free_tensors = Vec::new();

        while !ready_to_run.is_empty() {
            // All nodes in `ready_to_run` have all their dependencies
            // satisfied, so they can run in parallel in a single step.
            let mut step = Step::new(ready_to_run.iter().map(|&id| self.instrs[id].clone()));

            for tensor in free_tensors.drain(..) {
                step.push(Instr::FreeTensor(tensor));
            }

            plan.push_step(step);

            for instr in mem::take(&mut ready_to_run) {
                for &outbound in self
                    .outbound_edges
                    .get(instr)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                {
                    let indegree = &mut indegrees[outbound];
                    *indegree = indegree.checked_sub(1).unwrap();
                    if *indegree == 0 {
                        ready_to_run.push(outbound);
                    }
                }

                for node in self.instrs[instr].dependencies() {
                    let outdegree = &mut outdegrees[node];
                    *outdegree = outdegree.checked_sub(1).unwrap();
                    if *outdegree == 0 && !self.outputs.contains(&node) {
                        free_tensors.push(node);
                    }
                }
            }
        }

        if !free_tensors.is_empty() {
            let mut cleanup_step = Step::new([]);
            for tensor in free_tensors.drain(..) {
                cleanup_step.push(Instr::FreeTensor(tensor));
            }
            plan.push_step(cleanup_step);
        }

        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data_type::DataType,
        opgraph::{
            op::{
                BinaryPointwise, BinaryPointwiseOp, Matmul, ReduceOp, UnaryPointwise,
                UnaryPointwiseOp,
            },
            Descriptor,
        },
        shape::Shape,
    };

    #[test]
    fn test_generate_plan() {
        let mut graph = OpGraph::new();

        let input_a = graph.new_input(Descriptor {
            shape: Shape::new([1, 1]),
            data_type: DataType::Bf16,
        });
        let input_b = graph.new_input(Descriptor {
            shape: Shape::new([1, 1]),
            data_type: DataType::Bf16,
        });
        let input_c = graph.new_input(Descriptor {
            shape: Shape::new([1, 1]),
            data_type: DataType::F16,
        });

        let a = graph.new_op(Op::BinaryPointwise(BinaryPointwise {
            op: BinaryPointwiseOp::Add,
            lhs: input_a,
            rhs: input_c,
        }));
        let b = graph.new_op(Op::UnaryPointwise(UnaryPointwise {
            op: UnaryPointwiseOp::Sigmoid,
            input: a,
        }));
        let c = graph.new_op(Op::Reduce(Reduce {
            input: b,
            depth: 2,
            op: ReduceOp::Max,
        }));

        let d = graph.new_op(Op::BinaryPointwise(BinaryPointwise {
            op: BinaryPointwiseOp::Mul,
            lhs: input_b,
            rhs: a,
        }));

        let e = graph.new_op(Op::Matmul(Matmul {
            input_a: c,
            input_b: d,
        }));

        graph.new_output(e);

        let plan = compile_plan(&CudaContext::new(0).unwrap(), &Arc::new(graph)).unwrap();
        insta::assert_debug_snapshot!(plan);
    }
}
