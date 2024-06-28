//! Runtime generation of fused kernels.

use crate::{
    data_type::DataType,
    opgraph::{NodeId, VarId},
};
use ahash::AHashMap;
use std::cell::Cell;

pub mod pointwise;
pub mod reduction;

#[derive(Debug, Clone)]
pub struct Kernel {
    /// CUDA C++ code
    pub code: String,
    /// List of inputs to pass to kernel.
    pub params: Vec<KernelParam>,
}

#[derive(Debug, Clone)]
pub enum KernelParam {
    /// Tensor data pointer from a previous node's output
    Node(NodeId),
    /// Scalar float variable
    Var(VarId),
    /// Output tensor
    Output,
    /// Number of elements to operate on; used for pointwise
    /// and reduction
    Size,
    /// For reductions, specifies the stride between reduction
    /// groups
    ReductionStride,
}

/// Context for building kernel source code.
#[derive(Default)]
struct Context {
    /// Maps variables in the opgraph (i.e. dynamic scalar inputs)
    /// to the C++ variable name storing their values.
    var_map: AHashMap<VarId, String>,
    /// Maps input tensor arrays to their C++ identifiers.
    input_map: AHashMap<NodeId, String>,
    /// Maps intermediate outputs of nodes to their C++ identifiers,
    /// if they are present. This is used for fused operations.
    intermediate_map: AHashMap<NodeId, String>,
    next_identifier: Cell<u64>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_var(&mut self, var: VarId) -> String {
        let id = self.generate_identifier();
        self.var_map.insert(var, id.clone());
        id
    }

    pub fn insert_input(&mut self, input: NodeId) -> String {
        let id = self.generate_identifier();
        self.input_map.insert(input, id.clone());
        id
    }

    pub fn insert_intermediate(&mut self, intermediate: NodeId) -> String {
        let id = self.generate_identifier();
        self.intermediate_map.insert(intermediate, id.clone());
        id
    }

    pub fn var(&self, id: VarId) -> &str {
        &self.var_map[&id]
    }

    pub fn input(&self, id: NodeId) -> &str {
        &self.input_map[&id]
    }

    pub fn intermediate(&self, id: NodeId) -> &str {
        &self.intermediate_map[&id]
    }

    /// Generates a new unique identifier
    /// for use in variable declarations, function
    /// names, etc.
    pub fn generate_identifier(&self) -> String {
        let x = self.next_identifier.get();
        self.next_identifier.set(x + 1);
        format!("ident{x}")
    }
}

fn cpp_type_name(datatype: DataType) -> &'static str {
    match datatype {
        DataType::F32 => "float",
        DataType::Bf16 => "nv_bfloat16",
        DataType::F16 => "half",
    }
}
