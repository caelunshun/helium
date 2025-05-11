use crate::builder::{Section, Symbol, cpp_data_class};
use ahash::AHashMap;
use helium_ir::{
    data_type::{DataClass, Scalar},
    opgraph::{
        Intermediate, Node, NodeId,
        op::{BinaryPointwiseOp, Broadcast, CompareOp, Op, Reshape, SwapDims, UnaryPointwiseOp},
        subgraph::OpSubgraph,
    },
};

/// Context for generating pointwise operations
/// on data in registers (local variables).
///
/// All float computations happen in `f32` precision;
/// inputs and outputs should be casted to and from
/// the `float` data type. Boolean computations use `bool`
/// and integer computations use `u32` (`uint32_t`).
#[derive(Debug, Clone, Default)]
pub struct PointwiseContext {
    node_values: AHashMap<NodeId, Symbol>,
    node_data_classes: AHashMap<NodeId, DataClass>,
}

impl PointwiseContext {
    /// Indicates that the output value of the given node
    /// is already stored in the given symbol.
    pub fn insert(&mut self, node: NodeId, symbol: Symbol) {
        self.node_values.insert(node, symbol);
    }

    /// Gets the symbol name of the given node's output.
    pub fn get(&self, node: NodeId) -> Symbol {
        self.node_values[&node]
    }

    fn data_class(&self, node: NodeId) -> DataClass {
        self.node_data_classes[&node]
    }

    /// Emits code for a single pointwise operation, returning
    /// the symbol assigned to the output.
    /// All inputs must already be present.
    /// Ignores shape operations. Panics for non-pointwise non-shape
    /// operations (matmul/conv/reduce).
    fn emit_for_op(&mut self, op: &Op, dst: &mut Section) -> Symbol {
        let output = dst.new_symbol();
        match op {
            Op::UnaryPointwise(op) => {
                let input_sym = self.get(op.input);
                match op.op {
                    UnaryPointwiseOp::Recip => {
                        dst.emit(format!("float {output} = __fdividef(1.0f, {input_sym});"))
                    }
                    UnaryPointwiseOp::Neg => dst.emit(format!("float {output} = -{input_sym};")),
                    UnaryPointwiseOp::Exp => {
                        dst.emit(format!("float {output} = __expf({input_sym});"))
                    }
                    UnaryPointwiseOp::Sin => {
                        dst.emit(format!("float {output} = __sinf({input_sym});"))
                    }
                    UnaryPointwiseOp::Cos => {
                        dst.emit(format!("float {output} = __cosf({input_sym});"))
                    }
                    UnaryPointwiseOp::Tan => {
                        dst.emit(format!("float {output} = __tanf({input_sym});"))
                    }
                    UnaryPointwiseOp::Sigmoid => dst.emit(format!(
                        "float {output} = __fdividef(1.0f, 1.0f + __expf(-{input_sym}));"
                    )),
                    UnaryPointwiseOp::Tanh => {
                        dst.emit(format!("float {output} = __tanhf({input_sym});"))
                    }
                    UnaryPointwiseOp::Relu => {
                        dst.emit(format!("float {output} = fmaxf(0.0f, {input_sym});"))
                    }
                    UnaryPointwiseOp::Sqrt => {
                        dst.emit(format!("float {output} = __fsqrt_rn({input_sym});"))
                    }
                    UnaryPointwiseOp::Log => {
                        dst.emit(format!("float {output} = __logf({input_sym});"))
                    }
                };
            }
            Op::BinaryPointwise(op) => {
                let lhs_sym = self.get(op.lhs);
                let rhs_sym = self.get(op.rhs);
                match op.op {
                    BinaryPointwiseOp::Add => {
                        dst.emit(format!("float {output} = {lhs_sym} + {rhs_sym};"));
                    }
                    BinaryPointwiseOp::Mul => {
                        dst.emit(format!("float {output} = {lhs_sym} * {rhs_sym};"));
                    }
                    BinaryPointwiseOp::Pow => {
                        dst.emit(format!("float {output} = __powf({lhs_sym}, {rhs_sym});"));
                    }
                    BinaryPointwiseOp::Min => {
                        dst.emit(format!("float {output} = fminf({lhs_sym}, {rhs_sym});"));
                    }
                    BinaryPointwiseOp::Max => {
                        dst.emit(format!("float {output} = fmaxf({lhs_sym}, {rhs_sym});"));
                    }
                    BinaryPointwiseOp::And => {
                        dst.emit(format!("bool {output} = {lhs_sym} & {rhs_sym};"));
                    }
                    BinaryPointwiseOp::Or => {
                        dst.emit(format!("bool {output} = {lhs_sym} | {rhs_sym};"));
                    }
                    BinaryPointwiseOp::Xor => {
                        dst.emit(format!("bool {output} = {lhs_sym} ^ {rhs_sym};"));
                    }
                }
            }
            Op::ChangeDataType(op) => {
                let input_sym = self.get(op.input);
                let input_class = self.data_class(op.input);
                let output_class = op.target_type.class();
                match (input_class, output_class) {
                    (DataClass::Bool, DataClass::Bool)
                    | (DataClass::Float, DataClass::Float)
                    | (DataClass::Int, DataClass::Int) => {
                        dst.emit(format!(
                            "{} {output} = {input_sym};",
                            cpp_data_class(output_class)
                        ));
                    }
                    (DataClass::Bool, DataClass::Int) => {
                        dst.emit(format!(
                            "uint32_t {output} = static_cast<uint32_t>({input_sym});"
                        ));
                    }
                    (DataClass::Bool, DataClass::Float) => {
                        dst.emit(format!("float {output} = static_cast<float>({input_sym});"));
                    }
                    (DataClass::Int, DataClass::Bool) => {
                        dst.emit(format!("bool {output} = {input_sym} != 0;"));
                    }
                    (DataClass::Int, DataClass::Float) => {
                        dst.emit(format!("float {output} = static_cast<float>({input_sym});"));
                    }
                    (DataClass::Float, DataClass::Bool) => {
                        dst.emit(format!("bool {output} = {input_sym} != 0.0f;"));
                    }
                    (DataClass::Float, DataClass::Int) => {
                        dst.emit(format!(
                            "bool {output} = static_cast<uint32_t>({input_sym});"
                        ));
                    }
                }
            }
            Op::Compare(op) => {
                let lhs = self.get(op.lhs);
                let rhs = self.get(op.rhs);
                let operator = match op.op {
                    CompareOp::Equal => "==",
                    CompareOp::NotEqual => "!=",
                    CompareOp::LessThan => "<",
                    CompareOp::LessThanOrEqual => "<=",
                    CompareOp::GreaterThan => ">",
                    CompareOp::GreaterThanOrEqual => ">=",
                };
                dst.emit(format!("bool {output} = {lhs} {operator} {rhs};"));
            }
            Op::Select(op) => {
                let lhs = self.get(op.lhs);
                let rhs = self.get(op.rhs);
                let selector = self.get(op.selector);
                dst.emit(format!(
                    "{} {output} = {selector} ? {rhs} : {lhs};",
                    cpp_data_class(self.data_class(op.lhs))
                ));
            }
            Op::Constant(c) => {
                let scalar = match c.value {
                    Scalar::F32(x) => {
                        let x = x.to_bits();
                        format!("__uint_as_float(0x{x:x}u)")
                    }
                    Scalar::Bf16(x) => {
                        let x = x.to_bits();
                        format!("__ushort_as_bfloat16(0x{x:x}u)")
                    }
                    Scalar::F16(x) => {
                        let x = x.to_bits();
                        format!("__ushort_as_half(0x{x:x}u)")
                    }
                    Scalar::U32(x) => {
                        format!("0x{x:x}u")
                    }
                    Scalar::Bool(x) => x.to_string(),
                };
                dst.emit(format!(
                    "{} {output} = {scalar};",
                    cpp_data_class(c.value.data_type().class()),
                ));
            }
            Op::Broadcast(Broadcast { input, .. })
            | Op::Reshape(Reshape { input, .. })
            | Op::SwapDims(SwapDims { input, .. }) => {
                let input_sym = self.get(*input);
                dst.emit(format!(
                    "{} {output} = {input_sym};",
                    cpp_data_class(self.data_class(*input))
                ));
            }
            _ => panic!("unsupported pointwise op: {op:#?}"),
        }
        output
    }

    /// Recursively emits code to compute up to the given node.
    pub fn emit(&mut self, op_subgraph: &OpSubgraph, node_id: NodeId, dst: &mut Section) -> Symbol {
        if let Some(symbol) = self.node_values.get(&node_id) {
            return *symbol;
        }

        for input in op_subgraph.inbound_edges(node_id) {
            if !self.node_values.contains_key(&input) {
                self.emit(op_subgraph, input, dst);
            }
        }

        let Node::Intermediate(Intermediate { op, descriptor }) = op_subgraph.get(node_id) else {
            unreachable!()
        };
        let symbol = self.emit_for_op(op, dst);
        self.node_values.insert(node_id, symbol);
        self.node_data_classes
            .insert(node_id, descriptor.data_type.class());

        symbol
    }
}
