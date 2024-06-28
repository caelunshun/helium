use crate::opgraph::{Intermediate, Node, NodeId, OpGraph, VarId};

/// Subset of an `OpGraph` with the guarantee
/// that exactly one node (the leaf node) has an outgoing edge to a node
/// not in this subset.
pub struct OpSubgraph<'a> {
    graph: &'a OpGraph,
    nodes: Vec<NodeId>,
    inputs: Vec<NodeId>,
}

impl<'a> OpSubgraph<'a> {
    /// Creates a subgraph from a list of nodes
    /// in the subgraph.
    ///
    /// # Panics
    /// Panics if the conditions in the type-level docs
    /// are not met.
    pub fn from_nodes(graph: &'a OpGraph, nodes: Vec<NodeId>) -> Self {
        assert_eq!(
            nodes
                .iter()
                .copied()
                .filter(|&id| graph
                    .outbound_edges(id)
                    .iter()
                    .any(|input| !nodes.contains(input)))
                .count(),
            1,
            "number of leaf nodes must be exactly 1"
        );
        let mut inputs = Vec::new();
        for node in &nodes {
            for input in graph.inbound_edges(*node) {
                if !nodes.contains(input) && !inputs.contains(input) {
                    inputs.push(*input);
                }
            }
        }
        Self {
            nodes,
            graph,
            inputs,
        }
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

    pub fn leaf(&self) -> NodeId {
        self.nodes
            .iter()
            .copied()
            .find(|&id| {
                self.graph
                    .outbound_edges(id)
                    .iter()
                    .any(|output| !self.nodes.contains(output))
            })
            .expect("no leaf node in subgraph")
    }

    pub fn internal_indegree(&self, id: NodeId) -> usize {
        self.graph
            .inbound_edges(id)
            .iter()
            .filter(|&&prev| self.contains_node(prev))
            .count()
    }

    pub fn referenced_vars(&self) -> impl Iterator<Item = VarId> + '_ {
        let mut vars = Vec::new();
        for node in self.nodes() {
            if let Node::Intermediate(Intermediate { op, .. }) = self.graph.get(node) {
                for var in op.referenced_vars() {
                    if !vars.contains(&var) {
                        vars.push(var);
                    }
                }
            }
        }
        vars.into_iter()
    }

    pub fn graph(&self) -> &'a OpGraph {
        self.graph
    }
}
