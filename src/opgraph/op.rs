use crate::{
    data_type::DataType,
    opgraph::{Descriptor, NodeId, VarId},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
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
            _ => vec![],
        }
    }

    pub fn output_descriptor(&self, input_descriptors: &[Descriptor]) -> Descriptor {
        match self {
            Op::ChangeDataType(ChangeDataType { target_type, .. }) => Descriptor {
                dimension: input_descriptors[0].dimension,
                data_type: *target_type,
            },
            _ => input_descriptors[0],
        }
    }

    pub fn kind(&self) -> OpKind {
        match self {
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
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
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
