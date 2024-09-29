use crate::{
    data_type::DataType,
    opgraph::{Descriptor, NodeId, VarId, VarMap},
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
                dimension: get_input_descriptor(*input).dimension,
                data_type: *target_type,
            },
            Op::UnaryPointwise(UnaryPointwise { input, .. })
            | Op::BinaryPointwise(BinaryPointwise { lhs: input, .. })
            | Op::Transpose(Transpose { input })
            | Op::Matmul(Matmul { input_a: input, .. }) => get_input_descriptor(*input),
            Op::Reduce(Reduce { input, depth, .. }) => {
                let input = get_input_descriptor(*input);
                Descriptor {
                    dimension: input.dimension - *depth + 1,
                    data_type: DataType::F32,
                }
            }
            Op::UploadTensor(op) => op.descriptor,
        }
    }

    pub fn output_shape(
        &self,
        get_input_shape: impl Fn(NodeId) -> Vec<usize>,
        vars: &VarMap,
    ) -> Vec<usize> {
        match self {
            Op::Matmul(config) => {
                let shape_a = get_input_shape(config.input_a);
                let shape_b = get_input_shape(config.input_b);

                let mut out = shape_a.clone();
                let len = out.len();
                out[len - 2] = shape_b[shape_b.len() - 2];
                out
            }
            Op::Transpose(config) => {
                let mut shape = get_input_shape(config.input);
                tranpose_shape(&mut shape);
                shape
            }
            Op::UnaryPointwise(UnaryPointwise { input, .. })
            | Op::ChangeDataType(ChangeDataType { input, .. })
            | Op::BinaryPointwise(BinaryPointwise { lhs: input, .. }) => get_input_shape(*input),
            Op::Reduce(Reduce { depth, input, .. }) => {
                let mut shape = get_input_shape(*input);
                shape.truncate((shape.len() - *depth as usize).max(1));

                shape
            }
            Op::UploadTensor(op) => {
                let (_, shape) = vars.get(op.data_var).expect_tensor();
                shape.to_vec()
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
        }
    }

    pub fn is_pointwise(&self) -> bool {
        matches!(
            self.kind(),
            OpKind::UnaryPointwise | OpKind::BinaryPointwise | OpKind::ChangeDataType
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
        }
    }
}

fn tranpose_shape(shape: &mut Vec<usize>) {
    let len = shape.len();
    shape.swap(len - 1, len - 2);
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
