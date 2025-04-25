use crate::{
    DataType,
    backend::{InstrPerf, Instruction, TensorMap},
    cache::Cache,
    conv::Conv2dParams,
    cuda::{
        Cuda,
        allocator::{DeviceMemory, StreamId},
        context::{CudaContext, CudaStream},
        cudnn::{
            self, ConvDescriptor, ConvolutionBackwardDataDescriptor,
            ConvolutionBackwardFilterDescriptor, ConvolutionForwardOpDescriptor, CudnnContext,
            Engine, MatmulOpDescriptor, PointwiseMode, PointwiseOpDescriptor, TensorDescriptor,
            TensorKind, VariantPack, VariantPackBuilder,
        },
    },
    opgraph::{
        Intermediate, Node, NodeId, OpGraph,
        op::{BinaryPointwiseOp, Op, UnaryPointwiseOp},
        subgraph::OpSubgraph,
    },
    shape::Shape,
};
use ahash::AHashSet;
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
    pub fn precompile(&self, _cx: &CudaContext) {
        // Temporarily disabled due to needing to figure out
        // how to share cuDNN handles between threads.
        // self.get_engine(cx.cudnn_handle());
    }

    fn get_engine(
        &self,
        cudnn: &CudnnContext,
    ) -> (Arc<Engine>, Arc<SecondaryMap<NodeId, TensorDescriptor>>) {
        static ENGINE_CACHE: Cache<
            (usize, OpSubgraph),
            (Arc<Engine>, Arc<SecondaryMap<NodeId, TensorDescriptor>>),
        > = Cache::with_capacity(256);
        ENGINE_CACHE.get_or_insert(&(cudnn.id(), self.subgraph.clone()), || {
            let (graph, tensor_desc_map) = self.build(cudnn);
            let engine = Arc::new(Engine::choose_with_heuristic(&graph).unwrap_or_else(|e| {
                panic!("failed to get engine: {e}, graph = {:#?}", self.subgraph)
            }));
            (engine, Arc::new(tensor_desc_map))
        })
    }

    #[profiling::function]
    pub fn execute(
        &self,
        tensors: &TensorMap<Cuda>,
        stream: &CudaStream,
        cx: &CudaContext,
        hold_allocations: &mut Vec<DeviceMemory>,
        allocation_stream: StreamId,
    ) {
        let cudnn = cx.cudnn_handle();
        let (engine, tensor_desc_map) = self.get_engine(cudnn);

        let workspace_size = engine
            .workspace_size()
            .expect("failed to get workspace size");

        let workspace_mem;
        let workspace = if workspace_size == 0 {
            ptr::null_mut()
        } else {
            workspace_mem = cx
                .allocator()
                .allocate_in_stream(workspace_size as u64, 256, allocation_stream)
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
                .execute(&varpack, stream.raw() as _)
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

        let mode = Mode::of(&self.subgraph);

        while let Some(node_id) = stack.pop() {
            visited.insert(node_id);

            let node = self.subgraph.graph().get(node_id);

            let is_virtual = !(self.subgraph.leafs().any(|l| l == node_id)
                || self.subgraph.inputs().any(|i| i == node_id));

            let (augmented_shape, strides) = mode.augment_tensor_shape(&node.descriptor().shape);

            let tensor_desc = TensorDescriptor::with_strides(
                if is_virtual {
                    TensorKind::Virtual
                } else {
                    TensorKind::Concrete
                },
                node.descriptor().data_type,
                &augmented_shape,
                &strides,
            )
            .expect("failed to create tensor descriptor");

            tensor_descriptors.insert(node_id, tensor_desc);

            if !self.subgraph.inputs().any(|i| i == node_id) {
                let Node::Intermediate(node) = self.subgraph.graph().get(node_id) else {
                    unreachable!("cuDNN subgraph cannot contain input or output nodes")
                };
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
            Op::Matmul(op) => {
                builder.add_op(
                    MatmulOpDescriptor::new(
                        DataType::F32,
                        &tensor_descriptors[op.input_a],
                        &tensor_descriptors[op.input_b],
                        &tensor_descriptors[output],
                    )
                    .expect("failed to build matmul op"),
                );
            }
            Op::Conv(op) => {
                let conv = make_conv_descriptor(&op.settings);
                builder.add_op(
                    ConvolutionForwardOpDescriptor::new(
                        &conv,
                        &tensor_descriptors[op.filter],
                        &tensor_descriptors[op.image],
                        &tensor_descriptors[output],
                    )
                    .expect("failed to build forward conv op"),
                );
            }
            Op::ConvBackwardData(op) => {
                let conv = make_conv_descriptor(&op.settings);
                builder.add_op(
                    ConvolutionBackwardDataDescriptor::new(
                        &conv,
                        &tensor_descriptors[op.filter],
                        &tensor_descriptors[op.flow],
                        &tensor_descriptors[output],
                    )
                    .expect("failed to build conv backward data op"),
                );
            }
            Op::ConvBackwardFilter(op) => {
                let conv = make_conv_descriptor(&op.settings);
                builder.add_op(
                    ConvolutionBackwardFilterDescriptor::new(
                        &conv,
                        &tensor_descriptors[op.image],
                        &tensor_descriptors[op.flow],
                        &tensor_descriptors[output],
                    )
                    .expect("failed to build conv backward filter op"),
                );
            }
            Op::SwapDims(_) => todo!(),
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
            UnaryPointwiseOp::Relu => PointwiseMode::Relu,
            UnaryPointwiseOp::Log => PointwiseMode::Log,
            UnaryPointwiseOp::Sqrt => PointwiseMode::Sqrt,
        }
    }

    fn map_binary_pointwise_op(op: BinaryPointwiseOp) -> PointwiseMode {
        match op {
            BinaryPointwiseOp::Add => PointwiseMode::Add,
            BinaryPointwiseOp::Mul => PointwiseMode::Mul,
            BinaryPointwiseOp::Pow => PointwiseMode::Pow,
            BinaryPointwiseOp::Min => PointwiseMode::Min,
            BinaryPointwiseOp::Max => PointwiseMode::Max,
            BinaryPointwiseOp::Or | BinaryPointwiseOp::And | BinaryPointwiseOp::Xor => {
                unimplemented!()
            }
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    Matmul,
    Conv2d,
}

impl Mode {
    pub fn of(subgraph: &OpSubgraph) -> Self {
        for node in subgraph.nodes() {
            if let Node::Intermediate(Intermediate { op, .. }) = subgraph.graph().get(node) {
                if let Op::Matmul(_) = op {
                    return Mode::Matmul;
                } else if let Op::Conv(_) | Op::ConvBackwardFilter(_) | Op::ConvBackwardData(_) = op
                {
                    return Mode::Conv2d;
                }
            }
        }
        unreachable!("unsupported subgraph for cuDNN")
    }

    /// Gets tensor shape and strides to pass to the cuDNN API.
    pub fn augment_tensor_shape(&self, shape: &Shape) -> (Shape, Vec<usize>) {
        match self {
            Mode::Matmul => {
                // cuDNN as of v9.6.0 appears to have a bug where
                // matmul does not work without a batch dimension.
                // Add a batch dimension of 1.
                let mut new_shape = shape.dims().to_vec();
                if new_shape.len() == 2 {
                    new_shape.insert(0, 1);
                }
                let shape = Shape::new(new_shape);
                let strides = cudnn::compute_packed_strides(&shape);
                (shape, strides)
            }
            Mode::Conv2d => {
                // Helium tensors are always in NHWC layout,
                // as this gives better performance. But the cuDNN
                // API expects the (virtual) tensor shape to be
                // specified as NCHW. To make this work,
                // we modify the shape to NCHW while computing
                // strides such that the in-memory layout matches NHWC.
                let [n, h, w, c] = shape.dims().try_into().unwrap();
                let new_shape = Shape::new([n, c, h, w]);
                let strides = vec![h * w * c, 1, w * c, c];
                (new_shape, strides)
            }
        }
    }
}

fn make_conv_descriptor(settings: &Conv2dParams) -> ConvDescriptor {
    let padding = settings
        .padding_mode
        .compute_padding_amount(settings.kernel_size);
    ConvDescriptor::new(
        DataType::F32,
        2,
        &[settings.dilation[0] as u64, settings.dilation[1] as u64],
        &[settings.stride[0] as u64, settings.stride[1] as u64],
        &[padding[0] as u64, padding[1] as u64],
    )
    .expect("failed to build conv descriptor")
}
