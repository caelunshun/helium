use crate::{
    conv::Conv2dSettings,
    data_type::DataType,
    opgraph::{Descriptor, NodeId},
    shape::Shape,
};
use slotmap::SecondaryMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    Matmul(Matmul),
    UnaryPointwise(UnaryPointwise),
    BinaryPointwise(BinaryPointwise),
    ChangeDataType(ChangeDataType),
    Reduce(Reduce),
    Broadcast(Broadcast),
    Reshape(Reshape),
    SwapDims(SwapDims),
    Compare(Compare),
    Select(Select),
    Conv(Conv),
    ConvBackwardData(ConvBackwardData),
    ConvBackwardFilter(ConvBackwardFilter),
}

impl Op {
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            Op::Matmul(op) => vec![op.input_a, op.input_b],
            Op::UnaryPointwise(op) => vec![op.input],
            Op::BinaryPointwise(op) => vec![op.lhs, op.rhs],
            Op::ChangeDataType(op) => vec![op.input],
            Op::Reduce(op) => vec![op.input],
            Op::Broadcast(op) => vec![op.input],
            Op::Reshape(op) => vec![op.input],
            Op::SwapDims(op) => vec![op.input],
            Op::Compare(op) => vec![op.lhs, op.rhs],
            Op::Select(op) => vec![op.lhs, op.rhs, op.selector],
            Op::Conv(op) => vec![op.image, op.filter],
            Op::ConvBackwardData(op) => vec![op.flow, op.filter],
            Op::ConvBackwardFilter(op) => vec![op.flow, op.image],
        }
    }

    pub fn output_descriptor(
        &self,
        get_input_descriptor: impl Fn(NodeId) -> Descriptor,
    ) -> Descriptor {
        match self {
            Op::ChangeDataType(ChangeDataType { target_type, input }) => Descriptor {
                data_type: *target_type,
                ..get_input_descriptor(*input)
            },
            Op::Matmul(op) => {
                let input_a = get_input_descriptor(op.input_a);
                let input_b = get_input_descriptor(op.input_b);

                let mut shape = input_a.shape.clone();
                shape.set_dim_size(shape.num_dims() - 2, input_a.shape.dim_at(-2));
                shape.set_dim_size(shape.num_dims() - 1, input_b.shape.dim_at(-1));

                Descriptor { shape, ..input_a }
            }
            Op::UnaryPointwise(UnaryPointwise { input, .. })
            | Op::BinaryPointwise(BinaryPointwise { lhs: input, .. }) => {
                get_input_descriptor(*input)
            }
            Op::Reduce(Reduce { input, depth, .. }) => {
                let input = get_input_descriptor(*input);

                let mut shape = input.shape.dims().to_vec();
                shape.truncate(shape.len() - *depth as usize + 1);
                *shape.last_mut().unwrap() = 1;

                Descriptor {
                    shape: Shape::new(shape),
                    data_type: DataType::F32,
                }
            }
            Op::Reshape(op) => {
                let input = get_input_descriptor(op.input);
                Descriptor {
                    shape: op.new_shape.clone(),
                    ..input
                }
            }
            Op::Broadcast(op) => {
                let input = get_input_descriptor(op.input);
                let shape = op.output_shape(&input.shape);
                Descriptor { shape, ..input }
            }
            Op::SwapDims(op) => {
                let input = get_input_descriptor(op.input);
                let mut shape = input.shape.dims().to_vec();
                shape.swap(op.axis_a, op.axis_b);
                Descriptor {
                    shape: Shape::new(shape),
                    ..input
                }
            }
            Op::Compare(op) => {
                // Assume lhs and rhs have same shape (validated at high-level API)
                let input = get_input_descriptor(op.lhs);
                Descriptor {
                    data_type: DataType::Bool,
                    ..input
                }
            }
            Op::Select(op) => {
                // Assume lhs and rhs have same data type and shape (validated at high-level API0
                get_input_descriptor(op.lhs)
            }
            Op::Conv(op) => {
                let image = get_input_descriptor(op.image);
                let filter = get_input_descriptor(op.filter);

                let input_size = [image.shape.dim_at(1), image.shape.dim_at(2)];
                let kernel_size = [filter.shape.dim_at(1), filter.shape.dim_at(2)];
                let output_size = op.settings.compute_output_size(input_size, kernel_size);

                let batch_size = image.shape.dim_at(0);

                Descriptor {
                    data_type: image.data_type,
                    shape: Shape::new([
                        batch_size,
                        output_size[0],
                        output_size[1],
                        op.settings.out_channels,
                    ]),
                }
            }
            Op::ConvBackwardData(op) => {
                let flow = get_input_descriptor(op.flow);
                let batch_size = flow.shape.dim_at(0);

                Descriptor {
                    data_type: flow.data_type,
                    shape: Shape::new([
                        batch_size,
                        op.settings.in_channels,
                        op.input_size[0],
                        op.input_size[1],
                    ]),
                }
            }
            Op::ConvBackwardFilter(op) => {
                let flow = get_input_descriptor(op.flow);

                Descriptor {
                    data_type: flow.data_type,
                    shape: Shape::new([
                        op.settings.out_channels,
                        op.settings.kernel_size[0],
                        op.settings.kernel_size[1],
                        op.settings.in_channels,
                    ]),
                }
            }
        }
    }

    pub fn kind(&self) -> OpKind {
        match self {
            Op::Matmul(_) => OpKind::Matmul,
            Op::SwapDims(_) => OpKind::SwapDims,
            Op::Reduce(_) => OpKind::Reduce,
            Op::UnaryPointwise(_) => OpKind::UnaryPointwise,
            Op::BinaryPointwise(_) => OpKind::BinaryPointwise,
            Op::ChangeDataType(_) => OpKind::ChangeDataType,
            Op::Broadcast(_) => OpKind::Broadcast,
            Op::Reshape(_) => OpKind::Reshape,
            Op::Compare(_) => OpKind::Compare,
            Op::Select(_) => OpKind::Select,
            Op::Conv(_) => OpKind::Conv,
            Op::ConvBackwardData(_) => OpKind::ConvBackwardData,
            Op::ConvBackwardFilter(_) => OpKind::ConvBackwardFilter,
        }
    }

    pub fn apply_node_mapping(&mut self, mapping: &SecondaryMap<NodeId, NodeId>) {
        match self {
            Op::Matmul(op) => {
                op.input_a = mapping[op.input_a];
                op.input_b = mapping[op.input_b];
            }
            Op::SwapDims(op) => {
                op.input = mapping[op.input];
            }
            Op::UnaryPointwise(op) => {
                op.input = mapping[op.input];
            }
            Op::BinaryPointwise(op) => {
                op.lhs = mapping[op.lhs];
                op.rhs = mapping[op.rhs];
            }
            Op::ChangeDataType(op) => {
                op.input = mapping[op.input];
            }
            Op::Reduce(op) => {
                op.input = mapping[op.input];
            }
            Op::Broadcast(op) => {
                op.input = mapping[op.input];
            }
            Op::Reshape(op) => {
                op.input = mapping[op.input];
            }
            Op::Compare(op) => {
                op.lhs = mapping[op.lhs];
                op.rhs = mapping[op.rhs];
            }
            Op::Select(op) => {
                op.lhs = mapping[op.lhs];
                op.rhs = mapping[op.rhs];
                op.selector = mapping[op.selector];
            }
            Op::Conv(op) => {
                op.image = mapping[op.image];
                op.filter = mapping[op.filter];
            }
            Op::ConvBackwardData(op) => {
                op.flow = mapping[op.flow];
                op.filter = mapping[op.filter];
            }
            Op::ConvBackwardFilter(op) => {
                op.flow = mapping[op.flow];
                op.image = mapping[op.image];
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    Matmul,
    SwapDims,
    Reduce,
    UnaryPointwise,
    BinaryPointwise,
    ChangeDataType,
    Broadcast,
    Reshape,
    Compare,
    Select,
    Conv,
    ConvBackwardData,
    ConvBackwardFilter,
}

/// Batched multiplication of column-major matrices stored in the last
/// two dimensions of the input tensors `A` and `B`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Matmul {
    pub input_a: NodeId,
    pub input_b: NodeId,
}

/// Pointwise operator with one input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnaryPointwise {
    pub input: NodeId,
    pub op: UnaryPointwiseOp,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryPointwiseOp {
    Recip,
    Neg,
    Exp,
    Sin,
    Cos,
    Tan,
    Sigmoid,
    Tanh,
    Relu,
    Sqrt,
    Log,
}

/// Pointwise operator with two inputs.
/// Both inputs must have matching numbers of elements.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinaryPointwise {
    pub lhs: NodeId,
    pub rhs: NodeId,
    pub op: BinaryPointwiseOp,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinaryPointwiseOp {
    Add,
    Mul,
    /// `lhs^rhs`
    Pow,
    Min,
    Max,
}

/// Cast tensor to a new precision.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChangeDataType {
    pub input: NodeId,
    pub target_type: DataType,
}

/// Reduction of tensor along dimension(s).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Reduce {
    pub input: NodeId,
    /// How many dimensions to reduce on.
    /// The last `depth` dimensions are replaced
    /// with a single dimension that will contain
    /// the reduced values.
    pub depth: u32,
    pub op: ReduceOp,
}

/// Type of reduction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

impl ReduceOp {
    pub fn default_value(self) -> f32 {
        match self {
            ReduceOp::Sum => 0.0,
            ReduceOp::Mean => 0.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Broadcast {
    pub input: NodeId,
    pub new_dim_count: usize,
    pub broadcast_axes: Vec<BroadcastAxis>,
}

impl Broadcast {
    pub fn output_shape(&self, input_shape: &Shape) -> Shape {
        let mut shape = input_shape.dims().to_vec();

        if shape.len() < self.new_dim_count {
            for _ in 0..self.new_dim_count - shape.len() {
                shape.insert(0, 1);
            }
        }

        for BroadcastAxis { axis, new_size } in &self.broadcast_axes {
            shape[*axis] = *new_size;
        }
        Shape::new(shape)
    }

    pub fn is_axis_broadcasted(&self, axis: usize) -> bool {
        self.broadcast_axes.iter().any(|x| x.axis == axis)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BroadcastAxis {
    pub axis: usize,
    pub new_size: usize,
}

impl BroadcastAxis {
    pub fn new(axis: usize, new_size: usize) -> Self {
        Self { axis, new_size }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Reshape {
    pub input: NodeId,
    pub new_shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SwapDims {
    pub input: NodeId,
    pub axis_a: usize,
    pub axis_b: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Compare {
    pub lhs: NodeId,
    pub rhs: NodeId,
    pub op: CompareOp,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Select {
    pub lhs: NodeId,
    pub rhs: NodeId,
    /// Boolean tensor. Selects `lhs` for `false` and `rhs` for `true`.
    pub selector: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Conv {
    pub settings: Conv2dSettings,
    pub image: NodeId,
    pub filter: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvBackwardFilter {
    pub settings: Conv2dSettings,
    pub image: NodeId,
    pub flow: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvBackwardData {
    pub input_size: [usize; 2],
    pub settings: Conv2dSettings,
    pub filter: NodeId,
    pub flow: NodeId,
}
