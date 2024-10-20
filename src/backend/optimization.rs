use crate::{
    backend::{Backend, Instruction},
    opgraph::{Node, NodeId, OpGraph},
};
use slotmap::{SecondaryMap, SlotMap};
use std::{collections::BTreeSet, mem, sync::Arc};

pub fn generate_plan<B: Backend>(op_graph: &Arc<OpGraph>, backend: &B) -> Plan<B> {
    let mut graph = make_instr_graph(op_graph, backend);
    do_fusions(&mut graph, op_graph);
    generate_plan_from_graph(graph, op_graph)
}

/// Output of the optimization engine.
///
/// This is a list of steps, where each step
/// is a list of instructions that may execute
/// concurrently. A step begins executing only
/// after all instructions in the previous step
/// have completed.
#[derive(Debug)]
pub struct Plan<B: Backend> {
    steps: Vec<Step<B>>,
}

impl<B: Backend> Plan<B> {
    pub fn steps(&self) -> impl Iterator<Item = &Step<B>> {
        self.steps.iter()
    }
}

#[derive(Debug)]
pub struct Step<B: Backend> {
    instrs: Vec<B::Instr>,
    tensors_to_release: Vec<NodeId>,
}

impl<B: Backend> Step<B> {
    pub fn instrs(&self) -> &[B::Instr] {
        &self.instrs
    }

    pub(super) fn tensors_to_release(&self) -> &[NodeId] {
        &self.tensors_to_release
    }
}

/// Graph of backend instructions for optimization.
struct InstrGraph<B: Backend> {
    instrs: SlotMap<InstrNodeId, B::Instr>,
}

impl<B: Backend> InstrGraph<B> {
    pub fn new() -> Self {
        Self {
            instrs: SlotMap::default(),
        }
    }

    pub fn insert(&mut self, instr: B::Instr) -> InstrNodeId {
        self.instrs.insert(instr)
    }

    pub fn remove(&mut self, id: InstrNodeId) -> B::Instr {
        self.instrs.remove(id).unwrap()
    }

    pub fn get(&self, instr: InstrNodeId) -> &B::Instr {
        &self.instrs[instr]
    }

    pub fn instr_dependencies(&self, instr: InstrNodeId) -> impl Iterator<Item = InstrNodeId> + '_ {
        // PERF: can be optimized by storing edge lists
        let inputs = self.get(instr).inputs();
        self.instrs.iter().filter_map(move |(id, instr)| {
            if instr.outputs().iter().any(|o| inputs.contains(o)) {
                Some(id)
            } else {
                None
            }
        })
    }

    pub fn instr_dependents(&self, instr: InstrNodeId) -> impl Iterator<Item = InstrNodeId> + '_ {
        // PERF: can be optimized by storing edge lists
        let outputs = self.get(instr).outputs();
        self.instrs.iter().filter_map(move |(id, instr)| {
            if instr.inputs().iter().any(|i| outputs.contains(i)) {
                Some(id)
            } else {
                None
            }
        })
    }

    #[expect(unused)]
    pub fn roots(&self) -> impl Iterator<Item = InstrNodeId> + '_ {
        // PERF: can be optimized by caching
        self.instrs.iter().filter_map(|(id, instr)| {
            if instr.inputs().is_empty() {
                Some(id)
            } else {
                None
            }
        })
    }

    pub fn can_fuse_instrs(&self, a: InstrNodeId, b: InstrNodeId, op_graph: &Arc<OpGraph>) -> bool {
        self.instrs[a].can_fuse_with(&self.instrs[b], op_graph) && self.num_paths(a, b) == 1
    }

    fn num_paths(&self, a: InstrNodeId, b: InstrNodeId) -> usize {
        let mut count = 0;
        let mut stack = vec![a];
        while let Some(current) = stack.pop() {
            if current == b {
                count += 1;
            } else {
                stack.extend(self.instr_dependents(current));
            }
        }
        count
    }

    /// Fuses two instructions. `b` must depend on `a`.
    ///
    /// If no other instruction depends on `a`, then `a`
    /// is removed from the graph and fully merged into `b`.
    /// Otherwise, a duplicate of `a` is kept in the graph.
    pub fn fuse_instrs(&mut self, a: InstrNodeId, b: InstrNodeId, op_graph: &Arc<OpGraph>) {
        debug_assert!(self.instr_dependents(a).any(|x| x == b));
        debug_assert!(self.instr_dependencies(b).any(|x| x == a));
        debug_assert!(self.get(a).can_fuse_with(self.get(b), op_graph));

        let new_instr = self.get(a).fuse_with(self.get(b), op_graph);
        self.remove(a);
        self.instrs[b] = new_instr;
    }
}

slotmap::new_key_type! {
     struct InstrNodeId;
}

fn do_fusions<B: Backend>(graph: &mut InstrGraph<B>, op_graph: &Arc<OpGraph>) {
    let mut working_set: BTreeSet<InstrNodeId> = graph.instrs.keys().collect();

    while let Some(current) = working_set.first().copied() {
        let mut did_fuse = false;
        for next in graph.instr_dependents(current).collect::<Vec<_>>() {
            if graph.can_fuse_instrs(current, next, op_graph) {
                graph.fuse_instrs(current, next, op_graph);
                if !graph.instrs.contains_key(current) {
                    working_set.remove(&current);
                }
                did_fuse = true;
                break;
            }
        }

        if !did_fuse {
            working_set.remove(&current);
        }
    }
}

fn make_instr_graph<B: Backend>(op_graph: &Arc<OpGraph>, backend: &B) -> InstrGraph<B> {
    let mut graph = InstrGraph::new();

    for node_id in op_graph.nodes() {
        if let Node::Intermediate(node) = op_graph.get(node_id) {
            let instr = backend.make_instr_for_op(&node.op, op_graph, node_id);
            graph.insert(instr);
        }
    }

    graph
}

/// Performs a topological sort of the graph
/// to generate a plan that maximizes concurrency.
fn generate_plan_from_graph<B: Backend>(mut graph: InstrGraph<B>, op_graph: &OpGraph) -> Plan<B> {
    let num_instrs = graph.instrs.len();

    let mut indegrees = SecondaryMap::<InstrNodeId, usize>::default();
    let mut stack = Vec::new();
    for instr_id in graph.instrs.keys() {
        let indegree = graph.instr_dependencies(instr_id).count();
        indegrees.insert(instr_id, indegree);
        if indegree == 0 {
            stack.push(instr_id);
        }
    }

    let mut steps: Vec<Step<B>> = Vec::new();

    while !stack.is_empty() {
        let step = mem::take(&mut stack);

        for &instr_id in &step {
            for next in graph.instr_dependents(instr_id) {
                indegrees[next] = indegrees[next].checked_sub(1).unwrap();
                if indegrees[next] == 0 {
                    stack.push(next);
                }
            }
        }

        steps.push(Step {
            instrs: step
                .iter()
                .map(|id| graph.instrs.remove(*id).unwrap())
                .collect(),
            tensors_to_release: Vec::new(), // computed later in analyze_tensor_lifetimes
        });
    }

    assert_eq!(
        steps.iter().map(|step| step.instrs.len()).sum::<usize>(),
        num_instrs,
    );

    let mut plan = Plan { steps };
    analyze_tensor_lifetimes(&mut plan, op_graph);
    plan
}

fn analyze_tensor_lifetimes<B: Backend>(plan: &mut Plan<B>, op_graph: &OpGraph) {
    let mut tensor_release_steps = Vec::<Vec<NodeId>>::new();
    tensor_release_steps.push(Vec::new()); // no frees on first step

    for (i, step) in plan.steps().enumerate() {
        // Determine which tensors can be freed
        // at the next step, which is the subset
        // of used_tensors that are not used in any future
        // step.
        let mut used_tensors: Vec<NodeId> = step
            .instrs()
            .iter()
            .flat_map(|instr| instr.inputs())
            .collect();
        used_tensors.sort_unstable();
        used_tensors.dedup();

        let mut tensors_to_release = Vec::new();

        for used_tensor in used_tensors {
            if op_graph
                .outbound_edges(used_tensor)
                .iter()
                .any(|id| matches!(op_graph.get(*id), Node::Output(_)))
            {
                continue;
            }

            let mut used = false;
            for next_step in &plan.steps[i + 1..] {
                if next_step
                    .instrs()
                    .iter()
                    .any(|i| i.inputs().contains(&used_tensor))
                {
                    used = true;
                    break;
                }
            }
            if !used {
                tensors_to_release.push(used_tensor);
            }
        }

        tensor_release_steps.push(tensors_to_release);
    }

    // Add last step to free final tensors
    plan.steps.push(Step {
        instrs: Vec::new(),
        tensors_to_release: Vec::new(),
    });

    plan.steps
        .iter_mut()
        .zip(tensor_release_steps)
        .for_each(|(step, tensors)| step.tensors_to_release = tensors);
}
