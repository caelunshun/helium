use crate::opgraph::{Intermediate, Node, NodeId, OpGraph};
use indexmap::IndexSet;
use slotmap::SecondaryMap;
use std::{
    fmt::{Debug, Formatter},
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Subset of an `OpGraph`.
#[derive(Clone)]
pub struct OpSubgraph {
    graph: Arc<OpGraph>,
    nodes: IndexSet<NodeId, ahash::RandomState>,
    inputs: IndexSet<NodeId, ahash::RandomState>,
    roots: IndexSet<NodeId, ahash::RandomState>,
    leafs: IndexSet<NodeId, ahash::RandomState>,
}

impl OpSubgraph {
    /// Creates a subgraph from a list of nodes
    /// in the subgraph.
    ///
    /// # Panics
    /// Panics if the conditions in the type-level docs
    /// are not met.
    pub fn from_nodes(graph: &Arc<OpGraph>, mut nodes: Vec<NodeId>) -> Self {
        let mut inputs = IndexSet::default();
        for node in &nodes {
            assert!(matches!(graph.get(*node), Node::Intermediate(_)));
            for input in graph.inbound_edges(*node) {
                if !nodes.contains(input) {
                    inputs.insert(*input);
                }
            }
        }
        inputs.sort_unstable();
        nodes.sort_unstable();
        let nodes = IndexSet::<_, ahash::RandomState>::from_iter(nodes);

        let roots = nodes
            .iter()
            .copied()
            .filter(|node| {
                graph
                    .inbound_edges(*node)
                    .iter()
                    .any(|n| !nodes.contains(n))
            })
            .collect();

        let leafs = nodes
            .iter()
            .copied()
            .filter(|&id| {
                graph
                    .outbound_edges(id)
                    .iter()
                    .any(|output| !nodes.contains(output))
            })
            .collect();

        Self {
            nodes: nodes.into_iter().collect(),
            graph: Arc::clone(graph),
            roots,
            leafs,
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

    pub fn get(&self, id: NodeId) -> &Node {
        assert!(self.nodes.contains(&id));
        self.graph.get(id)
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

    pub fn roots(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.roots.iter().copied()
    }

    pub fn leafs(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.leafs.iter().copied()
    }

    pub fn is_input(&self, id: NodeId) -> bool {
        self.inputs.contains(&id)
    }

    pub fn is_root(&self, id: NodeId) -> bool {
        self.roots.contains(&id)
    }

    pub fn is_leaf(&self, id: NodeId) -> bool {
        self.leafs.contains(&id)
    }

    pub fn inbound_edges(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.graph
            .inbound_edges(id)
            .iter()
            .copied()
            .filter(|&pred| self.contains_node(pred) || self.is_input(pred))
    }

    pub fn outbound_edges(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.graph
            .outbound_edges(id)
            .iter()
            .copied()
            .filter(|&succ| self.contains_node(succ))
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

    /// Creates a new `OpGraph` containing only this subgraph.
    ///
    /// This operation is "lossy" because _NodeIds are not preserved_.
    /// However, the ordering of the inputs and outputs (as returned
    /// by `self.inputs()` and `self.leafs()`) is preserved.
    ///
    /// Returns the new graph and the forward mapping of node IDs.
    pub fn to_owned_lossy(&self) -> (OpGraph, SecondaryMap<NodeId, NodeId>) {
        let mut id_mapping = SecondaryMap::default();

        let mut graph = OpGraph::new();

        for input in self.inputs() {
            let new_id = graph.new_input(self.graph().get(input).descriptor().clone());
            id_mapping.insert(input, new_id);
        }

        let mut stack = self.roots().collect::<Vec<_>>();
        while let Some(current) = stack.pop() {
            let Node::Intermediate(Intermediate { op, .. }) = self.graph.get(current) else {
                unreachable!()
            };
            let mut new_op = op.clone();
            new_op.apply_node_mapping(&id_mapping);

            let new_id = graph.new_op(new_op);
            id_mapping.insert(current, new_id);

            for successor in self.graph.outbound_edges(current) {
                if self.contains_node(*successor)
                    && self
                        .graph
                        .inbound_edges(*successor)
                        .iter()
                        .all(|&id| id_mapping.contains_key(id))
                {
                    stack.push(*successor);
                }
            }
        }

        for leaf in self.leafs() {
            graph.new_output(id_mapping[leaf]);
        }

        (graph, id_mapping)
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
