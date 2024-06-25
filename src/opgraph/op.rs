use crate::{
    data_type::DataType,
    opgraph::{Descriptor, NodeId},
};

#[derive(Debug, Clone)]
pub enum Op {
    Matmul(Matmul),
    Transpose(Transpose),
    UnaryPointwise(UnaryPointwise),
    BinaryPointwise(BinaryPointwise),
    ChangeDataType(ChangeDataType),
}

impl Op {
    pub fn inputs(&self) -> Vec<NodeId> {
        match self {
            Op::Matmul(op) => vec![op.input_a, op.input_b],
            Op::Transpose(op) => vec![op.input],
            Op::UnaryPointwise(op) => vec![op.input],
            Op::BinaryPointwise(op) => vec![op.lhs, op.rhs],
            Op::ChangeDataType(op) => vec![op.input],
        }
    }

    pub fn output_descriptor(&self, input_descriptors: &[Descriptor]) -> Descriptor {
        input_descriptors[0]
    }

    pub fn kind(&self) -> OpKind {
        match self {
            Op::Matmul(_) => OpKind::Matmul,
            Op::Transpose(_) => OpKind::Tranpose,
            Op::UnaryPointwise(_) => OpKind::UnaryPointwise,
            Op::BinaryPointwise(_) => OpKind::BinaryPointwise,
            Op::ChangeDataType(_) => OpKind::ChangeDataType,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    Matmul,
    Tranpose,
    UnaryPointwise,
    BinaryPointwise,
    ChangeDataType,
}

/// Batched multiplication of column-major matrices stored in the last
/// two dimensions of the input tensors `A` and `B`.
#[derive(Debug, Clone)]
pub struct Matmul {
    pub input_a: NodeId,
    pub input_b: NodeId,
}

/// Batched transpose of matrices stored in the
/// last two dimensions of the input tensor.
#[derive(Debug, Clone)]
pub struct Transpose {
    pub input: NodeId,
}

/// Pointwise operator with one input.
#[derive(Debug, Clone)]
pub struct UnaryPointwise {
    pub input: NodeId,
    pub op: UnaryPointwiseOp,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnaryPointwiseOp {
    AddScalar(f64),
    MulScalar(f64),
    PowScalar(f64),
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
#[derive(Debug, Clone)]
pub struct BinaryPointwise {
    pub lhs: NodeId,
    pub rhs: NodeId,
    pub op: BinaryPointwiseOp,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BinaryPointwiseOp {
    Add,
    Mul,
    /// `lhs^rhs`
    Pow,
}

/// Cast tensor to a new precision.
#[derive(Debug, Clone)]
pub struct ChangeDataType {
    pub input: NodeId,
    pub target_type: DataType,
}
