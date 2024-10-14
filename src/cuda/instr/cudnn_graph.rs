use crate::{
    backend::{InstrPerf, Instruction, TensorMap},
    cuda::{
        allocator::Memory,
        context::CudaContext,
        cudnn,
        cudnn::{
            CudnnContext, Engine, PointwiseMode, PointwiseOpDescriptor, TensorDescriptor,
            TensorKind, VariantPack, VariantPackBuilder,
        },
        Cuda,
    },
    opgraph::{
        op::{BinaryPointwiseOp, Op, UnaryPointwiseOp},
        subgraph::OpSubgraph,
        Node, NodeId, OpGraph,
    },
    DataType,
};
use ahash::AHashSet;
use cudarc::cudnn::sys::cudaStream_t;
use slotmap::SecondaryMap;
use std::{ffi::c_void, ptr, sync::Arc};

#[derive(Debug, Clone)]
pub struct CudnnGraph {
    subgraph: OpSubgraph,
}

impl CudnnGraph {
    pub fn new(subgraph: OpSubgraph) -> Self {
        Self { subgraph }
    }
}

impl CudnnGraph {
    pub fn execute(
        &self,
        tensors: &TensorMap<Cuda>,
        stream: cudaStream_t,
        cx: &CudaContext,
        hold_allocations: &mut Vec<Memory>,
    ) {
        let cudnn = cx.cudnn_handle();
        let (graph, tensor_desc_map) = self.build(cudnn);

        let engine = Engine::choose_with_heuristic(&graph).expect("failed to get engine");
        let workspace_size = engine
            .workspace_size()
            .expect("failed to get workspace size");

        let workspace_mem;
        let workspace = if workspace_size == 0 {
            ptr::null_mut()
        } else {
            workspace_mem = cx
                .allocator()
                .alloc(workspace_size as u64, 256)
                .expect("failed to allocate workspace");
            let ptr = workspace_mem.device_ptr() as *mut c_void;
            // Prevent deallocation of workspace memory until after this
            // step
            hold_allocations.push(workspace_mem);
            ptr
        };

        let varpack = unsafe {
            Self::build_varpack(tensors, &tensor_desc_map)
                .build(workspace)
                .expect("failed to build varpack")
        };

        unsafe {
            engine
                .execute(&varpack, stream)
                .expect("failed to execute cuDNN graph");
        }
    }

    /// Builds the cuDNN graph structure.
    fn build(
        &self,
        cudnn: &CudnnContext,
    ) -> (
        cudnn::OperationGraph,
        SecondaryMap<NodeId, TensorDescriptor>,
    ) {
        let mut builder = cudnn::OperationGraph::builder();

        let mut tensor_descriptors: SecondaryMap<NodeId, TensorDescriptor> =
            SecondaryMap::default();

        let mut stack = self.subgraph.inputs().collect::<Vec<_>>();

        let mut visited = AHashSet::new();

        while let Some(node_id) = stack.pop() {
            visited.insert(node_id);
            let Node::Intermediate(node) = self.subgraph.graph().get(node_id) else {
                unreachable!("cuDNN subgraph cannot contain input or output nodes")
            };

            let is_virtual = !(self.subgraph.leafs().any(|l| l == node_id)
                || self.subgraph.inputs().any(|i| i == node_id));

            let tensor_desc = TensorDescriptor::new(
                if is_virtual {
                    TensorKind::Virtual
                } else {
                    TensorKind::Concrete
                },
                node.descriptor.data_type,
                &node.descriptor.shape,
            )
            .expect("failed to create tensor descriptor");

            tensor_descriptors.insert(node_id, tensor_desc);

            if !self.subgraph.inputs().any(|i| i == node_id) {
                Self::build_op(&node.op, node_id, &tensor_descriptors, &mut builder);
            }

            for next in self.subgraph.graph().outbound_edges(node_id) {
                if self.subgraph.contains_node(*next)
                    && self
                        .subgraph
                        .graph()
                        .inbound_edges(*next)
                        .iter()
                        .all(|&dependency| visited.contains(&dependency))
                {
                    stack.push(*next);
                }
            }
        }

        let graph = builder.build(cudnn).expect("failed to build cuDNN graph");
        (graph, tensor_descriptors)
    }

    fn build_op(
        op: &Op,
        output: NodeId,
        tensor_descriptors: &SecondaryMap<NodeId, TensorDescriptor>,
        builder: &mut cudnn::OperationGraphBuilder,
    ) {
        match op {
            Op::Matmul(_) => {}
            Op::Transpose(_) => todo!(),
            Op::UnaryPointwise(op) => {
                let mode = Self::map_unary_pointwise_op(op.op);
                builder.add_op(
                    PointwiseOpDescriptor::new(
                        mode,
                        DataType::F32,
                        &tensor_descriptors[op.input],
                        None,
                        &tensor_descriptors[output],
                    )
                    .expect("failed to build unary pointwise op"),
                );
            }
            Op::BinaryPointwise(op) => {
                let mode = Self::map_binary_pointwise_op(op.op);
                builder.add_op(
                    PointwiseOpDescriptor::new(
                        mode,
                        DataType::F32,
                        &tensor_descriptors[op.lhs],
                        Some(&tensor_descriptors[op.rhs]),
                        &tensor_descriptors[output],
                    )
                    .expect("failed to build binary pointwise op"),
                );
            }
            Op::ChangeDataType(_) => todo!(),
            Op::Reduce(_) => todo!(),
            Op::Broadcast(_) => todo!(),
            _ => unreachable!("not supported by cuDNN"),
        }
    }

    fn map_unary_pointwise_op(op: UnaryPointwiseOp) -> PointwiseMode {
        match op {
            UnaryPointwiseOp::Recip => PointwiseMode::Recip,
            UnaryPointwiseOp::Neg => PointwiseMode::Neg,
            UnaryPointwiseOp::Exp => PointwiseMode::Exp,
            UnaryPointwiseOp::Sin => PointwiseMode::Sin,
            UnaryPointwiseOp::Cos => PointwiseMode::Cos,
            UnaryPointwiseOp::Tan => PointwiseMode::Tan,
            UnaryPointwiseOp::Sigmoid => PointwiseMode::Sigmoid,
            UnaryPointwiseOp::Tanh => PointwiseMode::Tanh,
        }
    }

    fn map_binary_pointwise_op(op: BinaryPointwiseOp) -> PointwiseMode {
        match op {
            BinaryPointwiseOp::Add => PointwiseMode::Add,
            BinaryPointwiseOp::Mul => PointwiseMode::Mul,
            BinaryPointwiseOp::Pow => PointwiseMode::Pow,
            BinaryPointwiseOp::Min => PointwiseMode::Min,
            BinaryPointwiseOp::Max => PointwiseMode::Max,
        }
    }

    fn build_varpack(
        tensors: &TensorMap<Cuda>,
        tensor_desc_mapping: &SecondaryMap<NodeId, TensorDescriptor>,
    ) -> VariantPackBuilder {
        let mut builder = VariantPack::builder();

        for (node_id, tensor_desc) in tensor_desc_mapping {
            if tensor_desc.is_virtual() {
                continue;
            }
            let storage = tensors.get_storage(node_id);
            builder.add_tensor(tensor_desc.id(), storage.device_ptr() as *mut c_void);
        }

        builder
    }
}

impl Instruction<Cuda> for CudnnGraph {
    fn inputs(&self) -> Vec<NodeId> {
        self.subgraph.inputs().collect()
    }

    fn outputs(&self) -> Vec<NodeId> {
        self.subgraph.leafs().collect()
    }

    fn can_fuse_with(&self, _next: &Self, _op_graph: &Arc<OpGraph>) -> bool {
        false
    }

    fn fuse_with(&self, _next: &Self, _op_graph: &Arc<OpGraph>) -> Self {
        todo!()
    }

    fn perf(&self) -> InstrPerf {
        todo!()
    }
}
