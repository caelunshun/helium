use crate::{
    data_type::DataType,
    opgraph::{op::Op, NodeId, OpGraph},
    shape::Shape,
};
use slotmap::SecondaryMap;

mod optimization;

/// Trait for backends implementing tensor ops.
pub trait Backend: Copy + Sized {
    type Device: Copy;
    type Instr: Instruction<Self>;
    type TensorStorage: Clone;
    type Executor: Executor<Self>;

    fn make_instr_for_op(&self, op: &Op, graph: &OpGraph, node_id: NodeId) -> Self::Instr;
    fn begin_execute(&self) -> Self::Executor;
    fn allocate_tensor(&self, data_type: DataType, len: usize) -> Self::TensorStorage;
}

pub trait BackendExt: Backend {
    fn execute_graph(&self, mut graph: OpGraph, inputs: SecondaryMap<NodeId, Self::TensorStorage>) {
        graph.optimize();
        let plan = optimization::generate_plan(&graph, self);

        let mut tensors = TensorMap::new(&graph, inputs);

        let mut executor = self.begin_execute();
        for step in plan.steps() {
            executor.begin_step();

            // Allocate outputs for this step (backend expects
            // output tensors to already exist in the tensor map)
            for instr in step.instrs() {
                for output_node in instr.outputs() {
                    let descriptor = graph.get(output_node).descriptor();
                    let tensor =
                        self.allocate_tensor(descriptor.data_type, descriptor.shape.num_elements());
                    tensors.insert(output_node, tensor);
                }

                executor.execute_instr(instr, &mut tensors);
            }

            for released_tensor in step.tensors_to_release() {
                tensors.free(*released_tensor);
            }

            executor.end_step();
        }
    }
}

impl<B: Backend> BackendExt for B {}

pub trait Instruction<B: Backend> {
    fn inputs(&self) -> Vec<NodeId>;
    fn outputs(&self) -> Vec<NodeId>;

    #[must_use]
    fn can_fuse_with(&self, next: &Self) -> bool;
    #[must_use]
    fn fuse_with(&self, next: &Self) -> Self;

    fn perf(&self) -> InstrPerf;
}

pub trait Executor<B: Backend> {
    /// A step is a list of instructions that may execute
    /// concurrently.
    fn begin_step(&mut self);
    fn execute_instr(&mut self, instr: &B::Instr, tensors: &mut TensorMap<B>);
    fn end_step(&mut self);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InstrPerf {
    ComputeBound,
    MemoryBound,
}

pub struct TensorMap<'a, B: Backend> {
    tensors: SecondaryMap<NodeId, B::TensorStorage>,
    op_graph: &'a OpGraph,
}

impl<'a, B: Backend> TensorMap<'a, B> {
    fn new(op_graph: &'a OpGraph, inputs: SecondaryMap<NodeId, B::TensorStorage>) -> Self {
        Self {
            tensors: inputs,
            op_graph,
        }
    }

    fn insert(&mut self, node: NodeId, tensor: B::TensorStorage) {
        self.tensors.insert(node, tensor);
    }

    pub fn tensor_storage(&self, node: NodeId) -> &B::TensorStorage {
        &self.tensors[node]
    }

    pub fn tensor_shape(&self, node: NodeId) -> &Shape {
        &self.op_graph.get(node).descriptor().shape
    }

    fn free(&mut self, node: NodeId) {
        self.tensors.remove(node);
    }
}
