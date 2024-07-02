use crate::{
    cuda::plan::{Instr, Plan, Step},
    opgraph::{op::Op, Node, NodeId, OpGraph},
};
use ahash::AHashSet;
use slotmap::{SecondaryMap, SlotMap};
use std::{collections::VecDeque, mem};

/// Compiles an `OpGraph` into a `Plan` for CUDA execution.
struct Planner<'a> {
    graph: &'a OpGraph,

    /// Nodes whose inputs are already in the plan
    roots: Vec<NodeId>,

    /// Nodes that have been covered in the plan
    covered: AHashSet<NodeId>,

    instr_graph: InstrGraph,
}

impl<'a> Planner<'a> {
    pub fn new(graph: &'a OpGraph) -> Self {
        let mut roots = Vec::new();
        for &input in graph.inputs() {
            for &outbound in graph.outbound_edges(input) {
                if !roots.contains(&outbound) {
                    roots.push(outbound);
                }
            }
        }
        Self {
            graph,
            roots,
            covered: AHashSet::new(),
            instr_graph: InstrGraph::new(),
        }
    }

    /// Given a root, attempts to greedily
    /// consume more nodes that can be fused with the
    /// root. Generates steps in the plan
    /// for those nodes.
    fn cover_from_node(&mut self, node_id: NodeId) {
        let node = self.graph.get(node_id);
        let node = match node {
            Node::Intermediate(int) => int,
            Node::Output(_) => return,
            Node::Input(_) => unreachable!(),
        };

        match &node.op {
            Op::Matmul(_) => {}
            Op::Transpose(_) => todo!(),
            Op::UnaryPointwise(_) | Op::BinaryPointwise(_) | Op::ChangeDataType(_) => {}
            Op::Reduce(_) => {}
        }
    }

    fn cover_greedy_pointwise(&mut self, root: NodeId) -> Vec<NodeId> {
        let mut nodes = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while let Some(node) = queue.pop_front() {
            nodes.push(node);
            self.covered.insert(node);
            for &next_node_id in self.graph.outbound_edges(node) {
                // To fuse with the next node, it must
                // 1) be a pointwise operator
                // 2) all its dependencies must already be covered

                let next_node = self.graph.get(outbound_edge);
            }
        }
        nodes
    }
}

slotmap::new_key_type! {
    struct InstrId;
}

#[derive(Default)]
struct InstrGraph {
    roots: Vec<InstrId>,
    node_to_instr: SecondaryMap<NodeId, InstrId>,
    instrs: SlotMap<InstrId, Instr>,
    inbound_edges: SecondaryMap<InstrId, Vec<InstrId>>,
    outbound_edges: SecondaryMap<InstrId, Vec<InstrId>>,
}

impl InstrGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, id: InstrId) -> &Instr {
        &self.instrs[id]
    }

    pub fn insert(&mut self, instr: Instr) -> InstrId {
        let dependencies = instr.dependencies();
        let output = instr.output();
        let id = self.instrs.insert(instr);
        if let Some(output) = output {
            self.node_to_instr.insert(output, id);
        }

        for &dep in &dependencies {
            if let Some(dep) = self.node_to_instr.get(dep).copied() {
                self.outbound_edges
                    .entry(dep)
                    .unwrap()
                    .or_default()
                    .push(id);
            }
        }
        self.inbound_edges.insert(
            id,
            dependencies
                .into_iter()
                .map(|d| self.node_to_instr[d])
                .collect(),
        );
        id
    }

    /// Generates a `Plan` that aims to maximize parallelism
    /// based on this graph.
    pub fn into_plan(mut self) -> Plan {
        let mut plan = Plan::new();

        let mut indegrees = SecondaryMap::default();
        let mut ready_to_run = Vec::new();
        for instr in self.instrs.keys() {
            let indegree = self
                .inbound_edges
                .get(instr)
                .map(Vec::as_slice)
                .unwrap_or_default()
                .len();
            indegrees.insert(instr, indegree);
            if indegree == 0 {
                ready_to_run.push(instr);
            }
        }

        while !ready_to_run.is_empty() {
            // All nodes in `ready_to_run` have all their dependencies
            // satisfied, so they can run in parallel in a single step.
            let step = Step::new(
                ready_to_run
                    .iter()
                    .map(|&id| self.instrs.remove(id).unwrap()),
            );
            plan.push_step(step);

            for instr in mem::take(&mut ready_to_run) {
                for &outbound in self
                    .outbound_edges
                    .get(instr)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                {
                    let indegree = &mut indegrees[outbound];
                    *indegree = indegree.checked_sub(1).unwrap();
                    if *indegree == 0 {
                        ready_to_run.push(outbound);
                    }
                }
            }
        }

        plan
    }
}
