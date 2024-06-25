use crate::{data_type::DataType, opgraph::op::Op};
use slotmap::{SecondaryMap, SlotMap};

pub mod op;
pub mod subgraph;

#[derive(Debug, Clone, Default)]
pub struct OpGraph {
    nodes: SlotMap<NodeId, Node>,
    outputs: Vec<NodeId>,
    inputs: Vec<NodeId>,
    outbound_edges: SecondaryMap<NodeId, Vec<NodeId>>,
    inbound_edges: SecondaryMap<NodeId, Vec<NodeId>>,
    num_vars: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VarId(u32);

impl OpGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new input node.
    pub fn new_input(&mut self, descriptor: Descriptor) -> NodeId {
        let id = self.nodes.insert(Node::Input(Input { descriptor }));
        self.inputs.push(id);
        id
    }

    /// Creates a new intermediate node with the given operation.
    pub fn new_op(&mut self, op: Op) -> NodeId {
        let inputs = op.inputs();
        let input_descriptors: Vec<_> = inputs
            .iter()
            .map(|id| self.nodes[*id].descriptor())
            .collect();
        let node = self.nodes.insert(Node::Intermediate(Intermediate {
            descriptor: op.output_descriptor(&input_descriptors),
            op,
        }));
        for &input in &inputs {
            push_if_absent(self.outbound_edges.entry(input).unwrap().or_default(), node);
        }
        self.inbound_edges.insert(node, inputs);
        node
    }

    /// Creates a new output from the given node.
    pub fn new_output(&mut self, from_node: NodeId) -> NodeId {
        let descriptor = self.nodes[from_node].descriptor();
        let node = self.nodes.insert(Node::Output(descriptor));
        self.outputs.push(node);
        self.inbound_edges.insert(node, vec![from_node]);
        push_if_absent(
            self.outbound_edges.entry(from_node).unwrap().or_default(),
            node,
        );
        node
    }

    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    pub fn outbound_edges(&self, id: NodeId) -> &[NodeId] {
        self.outbound_edges
            .get(id)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub fn inbound_edges(&self, id: NodeId) -> &[NodeId] {
        self.inbound_edges
            .get(id)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn is_output(&self, id: NodeId) -> bool {
        self.outputs.contains(&id)
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn is_input(&self, id: NodeId) -> bool {
        self.inputs.contains(&id)
    }

    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    pub fn vars(&self) -> impl Iterator<Item = VarId> + '_ {
        (0..self.num_vars).map(VarId)
    }

    pub fn new_var(&mut self) -> VarId {
        let id = VarId(self.num_vars);
        self.num_vars += 1;
        id
    }
}

fn push_if_absent<T: Eq>(vec: &mut Vec<T>, val: T) {
    if !vec.contains(&val) {
        vec.push(val);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Descriptor {
    /// Dimension of the tensor in this node.
    pub dimension: u32,
    /// Data type stored in the tensor.
    pub data_type: DataType,
}

slotmap::new_key_type! {
    pub struct NodeId;
}

#[derive(Debug, Clone)]
pub enum Node {
    Input(Input),
    Intermediate(Intermediate),
    Output(Descriptor),
}

impl Node {
    pub fn descriptor(&self) -> Descriptor {
        match self {
            Node::Input(n) => n.descriptor,
            Node::Intermediate(n) => n.descriptor,
            Node::Output(d) => *d,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Input {
    pub descriptor: Descriptor,
}

#[derive(Debug, Clone)]
pub struct Intermediate {
    pub descriptor: Descriptor,
    pub op: Op,
}
