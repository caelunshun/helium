use crate::opgraph::{NodeId, OpGraph};
use std::{
    fmt::{Debug, Formatter},
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Subset of an `OpGraph`.
#[derive(Clone)]
pub struct OpSubgraph {
    graph: Arc<OpGraph>,
    nodes: Vec<NodeId>,
    inputs: Vec<NodeId>,
}

impl OpSubgraph {
    /// Creates a subgraph from a list of nodes
    /// in the subgraph.
    ///
    /// # Panics
    /// Panics if the conditions in the type-level docs
    /// are not met.
    pub fn from_nodes(graph: &Arc<OpGraph>, mut nodes: Vec<NodeId>) -> Self {
        let mut inputs = Vec::new();
        for node in &nodes {
            for input in graph.inbound_edges(*node) {
                if !nodes.contains(input) && !inputs.contains(input) {
                    inputs.push(*input);
                }
            }
        }
        inputs.sort_unstable();
        nodes.sort_unstable();
        Self {
            nodes,
            graph: Arc::clone(graph),
            inputs,
        }
    }

    pub fn merge_with(&self, other: &Self) -> Self {
        assert!(Arc::ptr_eq(&self.graph, &other.graph));
        let mut nodes = self
            .nodes
            .iter()
            .copied()
            .chain(other.nodes.iter().copied())
            .collect::<Vec<_>>();
        nodes.sort_unstable();
        nodes.dedup();
        Self::from_nodes(&self.graph, nodes)
    }

    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().copied()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn contains_node(&self, id: NodeId) -> bool {
        self.nodes.contains(&id)
    }

    pub fn inputs(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.inputs.iter().copied()
    }

    pub fn leafs(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().copied().filter(|&id| {
            self.graph
                .outbound_edges(id)
                .iter()
                .any(|output| !self.nodes.contains(output))
        })
    }

    pub fn internal_indegree(&self, id: NodeId) -> usize {
        self.graph
            .inbound_edges(id)
            .iter()
            .filter(|&&prev| self.contains_node(prev))
            .count()
    }

    pub fn graph(&self) -> &Arc<OpGraph> {
        &self.graph
    }
}

impl PartialEq for OpSubgraph {
    fn eq(&self, other: &Self) -> bool {
        self.nodes.len() == other.nodes.len()
            && self
                .nodes
                .iter()
                .zip(other.nodes.iter())
                .all(|(a, b)| self.graph.get(*a) == other.graph.get(*b) && *a == *b)
            && self.inputs.len() == other.inputs.len()
            && self
                .inputs
                .iter()
                .zip(other.inputs.iter())
                .all(|(a, b)| self.graph.get(*a).descriptor() == other.graph.get(*b).descriptor())
    }
}

impl Eq for OpSubgraph {}

impl Hash for OpSubgraph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.nodes.len().hash(state);
        for node in &self.nodes {
            self.graph.get(*node).hash(state);
        }
        self.inputs.len().hash(state);
        for node in &self.nodes {
            self.graph.get(*node).descriptor().hash(state);
        }
    }
}

impl Debug for OpSubgraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_list();

        for &node_id in &self.nodes {
            let node = self.graph.get(node_id);
            f.entry(&(node_id, node));
        }

        f.finish()
    }
}
