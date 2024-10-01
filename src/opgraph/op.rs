use crate::{
    data_type::DataType,
    opgraph::{Descriptor, NodeId, VarId},
    shape::Shape,
};
use slotmap::SecondaryMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    UploadTensor(UploadTensor),
    Matmul(Matmul),
    Transpose(Transpose),
    UnaryPointwise(UnaryPointwise),
    BinaryPointwise(BinaryPointwise),
    ChangeDataType(ChangeDataType),
    Reduce(Reduce),
    Restructure(Restructure),
    Reshape(Reshape),
}

impl Op {
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            Op::Matmul(op) => vec![op.input_a, op.input_b],
            Op::Transpose(op) => vec![op.input],
            Op::UnaryPointwise(op) => vec![op.input],
            Op::BinaryPointwise(op) => vec![op.lhs, op.rhs],
            Op::ChangeDataType(op) => vec![op.input],
            Op::Reduce(op) => vec![op.input],
            Op::UploadTensor(_) => vec![],
            Op::Restructure(op) => vec![op.input],
            Op::Reshape(op) => vec![op.input],
        }
    }

    pub fn referenced_vars(&self) -> Vec<VarId> {
        match self {
            Op::UnaryPointwise(UnaryPointwise {
                op:
                    UnaryPointwiseOp::AddScalar(var)
                    | UnaryPointwiseOp::MulScalar(var)
                    | UnaryPointwiseOp::PowScalar(var),
                ..
            }) => vec![*var],
            Op::UploadTensor(op) => vec![op.data_var],
            _ => vec![],
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
                shape.set_dim_size(shape.num_dims() - 1, input_a.shape.dim_at(-1));
                shape.set_dim_size(shape.num_dims() - 2, input_b.shape.dim_at(-2));

                Descriptor { shape, ..input_a }
            }
            Op::Transpose(op) => {
                let input = get_input_descriptor(op.input);
                let mut shape = input.shape.dims().to_vec();
                let len = shape.len();
                shape.swap(len - 1, len - 2);
                Descriptor {
                    shape: Shape::new(shape),
                    data_type: DataType::F16,
                }
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
            Op::UploadTensor(op) => op.descriptor.clone(),
            Op::Reshape(op) => {
                let input = get_input_descriptor(op.input);
                Descriptor {
                    shape: op.new_shape.clone(),
                    ..input
                }
            }
            Op::Restructure(op) => {
                let input = get_input_descriptor(op.input);
                Descriptor {
                    shape: op.op.compute_output_shape(input.shape.dims()),
                    ..input
                }
            }
        }
    }

    pub fn kind(&self) -> OpKind {
        match self {
            Op::UploadTensor(_) => OpKind::UploadTensor,
            Op::Matmul(_) => OpKind::Matmul,
            Op::Transpose(_) => OpKind::Tranpose,
            Op::Reduce(_) => OpKind::Reduce,
            Op::UnaryPointwise(_) => OpKind::UnaryPointwise,
            Op::BinaryPointwise(_) => OpKind::BinaryPointwise,
            Op::ChangeDataType(_) => OpKind::ChangeDataType,
            Op::Restructure(_) => OpKind::Restructure,
            Op::Reshape(_) => OpKind::Reshape,
        }
    }

    pub fn is_pointwise(&self) -> bool {
        matches!(
            self.kind(),
            OpKind::UnaryPointwise
                | OpKind::BinaryPointwise
                | OpKind::ChangeDataType
                | OpKind::Reshape
        )
    }

    pub fn apply_node_mapping(&mut self, mapping: &SecondaryMap<NodeId, NodeId>) {
        match self {
            Op::UploadTensor(_) => {}
            Op::Matmul(op) => {
                op.input_a = mapping[op.input_a];
                op.input_b = mapping[op.input_b];
            }
            Op::Transpose(op) => {
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
            Op::Restructure(op) => {
                op.input = mapping[op.input];
            }
            Op::Reshape(op) => {
                op.input = mapping[op.input];
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    UploadTensor,
    Matmul,
    Tranpose,
    Reduce,
    UnaryPointwise,
    BinaryPointwise,
    ChangeDataType,
    Restructure,
    Reshape,
}

/// Batched multiplication of column-major matrices stored in the last
/// two dimensions of the input tensors `A` and `B`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Matmul {
    pub input_a: NodeId,
    pub input_b: NodeId,
}

/// Batched transpose of matrices stored in the
/// last two dimensions of the input tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transpose {
    pub input: NodeId,
}

/// Pointwise operator with one input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnaryPointwise {
    pub input: NodeId,
    pub op: UnaryPointwiseOp,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnaryPointwiseOp {
    AddScalar(VarId),
    MulScalar(VarId),
    PowScalar(VarId),
    Recip,
    Neg,
    Exp,
    Sin,
    Cos,
    Tan,
    Sigmoid,
    Tanh,
    Relu,
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
pub struct UploadTensor {
    pub data_var: VarId,
    pub descriptor: Descriptor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Restructure {
    pub input: NodeId,
    pub op: RestrctureOp,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RestrctureOp {
    /// Broadcast an axis originally of length 1.
    BroadcastAxis {
        axis: BroadcastAxis,
        new_size: usize,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BroadcastAxis {
    Existing(usize),
    /// Prepend a new axis.
    Expand,
}

impl RestrctureOp {
    pub fn compute_output_shape(&self, input: &[usize]) -> Shape {
        let mut shape = input.to_vec();
        match self {
            RestrctureOp::BroadcastAxis { axis, new_size } => match *axis {
                BroadcastAxis::Existing(axis) => shape[axis] = *new_size,
                BroadcastAxis::Expand => shape.insert(0, *new_size),
            },
        }
        Shape::new(shape)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Reshape {
    pub input: NodeId,
    pub new_shape: Shape,
}
