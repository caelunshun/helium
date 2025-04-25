use crate::{data_type::DataType, opgraph::op::Op, shape::Shape};
use slotmap::{SecondaryMap, SlotMap};
use std::{
    fmt::{Debug, Formatter},
    hash::{Hash, Hasher},
};

pub mod op;
pub mod subgraph;

#[derive(Clone, Default)]
pub struct OpGraph {
    nodes: SlotMap<NodeId, Node>,
    outputs: Vec<NodeId>,
    inputs: Vec<NodeId>,
    outbound_edges: SecondaryMap<NodeId, Vec<NodeId>>,
    inbound_edges: SecondaryMap<NodeId, Vec<NodeId>>,
}

impl OpGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge_into(&self, other: &mut OpGraph) -> SecondaryMap<NodeId, NodeId> {
        let mut id_mapping = SecondaryMap::new();

        for (node_id, node) in &self.nodes {
            let new_node_id = other.nodes.insert(node.clone());
            if let Node::Intermediate(op) = &mut other.nodes[new_node_id] {
                op.op.apply_node_mapping(&id_mapping);
            }

            id_mapping.insert(node_id, new_node_id);
        }

        for node_id in self.nodes.keys() {
            let new_node_id = id_mapping[node_id];
            other.outbound_edges.insert(
                new_node_id,
                self.outbound_edges(node_id)
                    .iter()
                    .map(|&id| id_mapping[id])
                    .collect(),
            );
            other.inbound_edges.insert(
                new_node_id,
                self.inbound_edges(node_id)
                    .iter()
                    .map(|&id| id_mapping[id])
                    .collect(),
            );
        }

        for &input in &self.inputs {
            other.inputs.push(id_mapping[input]);
        }
        for &output in &self.outputs {
            other.outputs.push(id_mapping[output]);
        }

        id_mapping
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
        let node = self.nodes.insert(Node::Intermediate(Intermediate {
            descriptor: op.output_descriptor(|id| self.get(id).descriptor().clone()),
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
        let node = self.nodes.insert(Node::Output(descriptor.clone()));
        self.outputs.push(node);
        self.inbound_edges.insert(node, vec![from_node]);
        push_if_absent(
            self.outbound_edges.entry(from_node).unwrap().or_default(),
            node,
        );
        node
    }

    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys()
    }

    pub fn remove(&mut self, id: NodeId) {
        self.nodes.remove(id).unwrap();
        for pred in self
            .inbound_edges
            .get(id)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            remove_element(&mut self.outbound_edges[*pred], id);
        }
        for suc in self
            .outbound_edges
            .get(id)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            remove_element(&mut self.inbound_edges[*suc], id);
        }

        remove_element(&mut self.inputs, id);
        remove_element(&mut self.outputs, id);
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

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Removes unused operations and inputs.
    pub fn optimize(&mut self) {
        self.prune_unused();
    }

    fn prune_unused(&mut self) {
        let mut visited_nodes = SecondaryMap::<NodeId, ()>::default();

        let mut stack = self.outputs.clone();
        while let Some(node) = stack.pop() {
            visited_nodes.insert(node, ());

            for &pred in self.inbound_edges(node) {
                if !visited_nodes.contains_key(pred) {
                    stack.push(pred);
                }
            }
        }

        let to_remove: Vec<NodeId> = self
            .nodes
            .keys()
            .filter(|&id| !visited_nodes.contains_key(id))
            .collect();
        for node in to_remove {
            self.remove(node);
        }
    }
}

impl Debug for OpGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_list();

        for (id, node) in &self.nodes {
            f.entry(&(id, node));
        }

        f.finish()
    }
}

impl PartialEq for OpGraph {
    fn eq(&self, other: &Self) -> bool {
        self.nodes.len() == other.nodes.len()
            && self.nodes.iter().zip(other.nodes.iter()).all(|(a, b)| {
                a == b
                    && self.inbound_edges(a.0) == other.inbound_edges(b.0)
                    && self.outbound_edges(a.0) == other.outbound_edges(b.0)
            })
    }
}

impl Eq for OpGraph {}

impl Hash for OpGraph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (id, node) in &self.nodes {
            (id, node).hash(state);
        }
    }
}

fn push_if_absent<T: Eq>(vec: &mut Vec<T>, val: T) {
    if !vec.contains(&val) {
        vec.push(val);
    }
}

fn remove_element<T: Eq>(vec: &mut Vec<T>, val: T) {
    let pos = vec.iter().position(|x| x == &val);
    if let Some(pos) = pos {
        vec.swap_remove(pos);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Descriptor {
    /// Shape of the tensor in this node.
    pub shape: Shape,
    /// Data type stored in the tensor.
    pub data_type: DataType,
}

slotmap::new_key_type! {
    pub struct NodeId;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Node {
    Input(Input),
    Intermediate(Intermediate),
    Output(Descriptor),
}

impl Node {
    pub fn descriptor(&self) -> &Descriptor {
        match self {
            Node::Input(n) => &n.descriptor,
            Node::Intermediate(n) => &n.descriptor,
            Node::Output(d) => d,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Input {
    pub descriptor: Descriptor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Intermediate {
    pub descriptor: Descriptor,
    pub op: Op,
}
