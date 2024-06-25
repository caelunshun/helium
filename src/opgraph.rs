use crate::{data_type::DataType, opgraph::op::Op};
use slotmap::{SecondaryMap, SlotMap};

pub mod op;

#[derive(Debug, Clone, Default)]
pub struct OpGraph {
    nodes: SlotMap<NodeId, Node>,
    outputs: Vec<NodeId>,
    outbound_edges: SecondaryMap<NodeId, Vec<NodeId>>,
    inbound_edges: SecondaryMap<NodeId, Vec<NodeId>>,
}

impl OpGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new input node.
    pub fn new_input(&mut self, descriptor: Descriptor) -> NodeId {
        self.nodes.insert(Node::Input(Input { descriptor }))
    }

    /// Creates a new intermediate node with the given operation.
    pub fn push(&mut self, op: Op) -> NodeId {
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
            self.outbound_edges
                .entry(input)
                .unwrap()
                .or_default()
                .push(input);
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
        self.outbound_edges
            .entry(from_node)
            .unwrap()
            .or_default()
            .push(node);
        node
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
