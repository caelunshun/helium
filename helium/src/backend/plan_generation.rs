use crate::{
    backend::{Backend, Instruction},
    cache::Cache,
    opgraph::{Node, NodeId, OpGraph},
};
use ahash::{AHashMap, AHashSet};
use slotmap::{SecondaryMap, SlotMap};
use std::{
    any::{Any, TypeId},
    collections::BTreeSet,
    mem,
    sync::Arc,
};

#[profiling::function]
pub fn generate_cached_plan<B: Backend>(op_graph: &Arc<OpGraph>, backend: &B) -> Arc<Plan<B>> {
    static CACHE: PlanCache = PlanCache::new();
    CACHE.get_or_insert(op_graph, backend)
}

#[profiling::function]
fn generate_plan<B: Backend>(op_graph: &Arc<OpGraph>, backend: &B) -> Plan<B> {
    let mut graph = make_instr_graph(op_graph, backend);
    do_fusions(&mut graph, op_graph);
    generate_plan_from_graph(graph, op_graph)
}

/// Caches plans by backend type ID + op graph.
struct PlanCache {
    cache: Cache<(Arc<OpGraph>, TypeId), Arc<dyn Any + Send + Sync>>,
}

impl PlanCache {
    pub const fn new() -> Self {
        Self {
            cache: Cache::with_capacity(256),
        }
    }

    pub fn get_or_insert<B: Backend>(&self, graph: &Arc<OpGraph>, backend: &B) -> Arc<Plan<B>> {
        Arc::downcast(
            self.cache
                .get_or_insert(&(graph.clone(), TypeId::of::<B>()), || {
                    Arc::new(generate_plan(graph, backend))
                }),
        )
        .ok()
        .unwrap()
    }
}

/// Output of the optimization engine.
///
/// This is a list of steps, where each step
/// is a list of instructions that may execute
/// concurrently. A step begins executing only
/// after all instructions in the previous step
/// have completed.
#[derive(Debug, Clone)]
pub struct Plan<B: Backend> {
    steps: Vec<Step<B>>,
}

impl<B: Backend> Plan<B> {
    pub fn steps(&self) -> &[Step<B>] {
        &self.steps
    }
}

#[derive(Debug, Clone)]
pub struct Step<B: Backend> {
    instrs: Vec<B::Instr>,
    tensors_to_release: Vec<NodeId>,
}

impl<B: Backend> Step<B> {
    pub fn instrs(&self) -> &[B::Instr] {
        &self.instrs
    }

    pub fn tensors_to_release(&self) -> &[NodeId] {
        &self.tensors_to_release
    }
}

/// Graph of backend instructions for optimization.
struct InstrGraph<B: Backend> {
    instrs: SlotMap<InstrNodeId, B::Instr>,
    producers: AHashMap<NodeId, InstrNodeId>,
    consumers: AHashMap<NodeId, BTreeSet<InstrNodeId>>,

    dependencies: SecondaryMap<InstrNodeId, BTreeSet<InstrNodeId>>,
    dependents: SecondaryMap<InstrNodeId, BTreeSet<InstrNodeId>>,
}

impl<B: Backend> InstrGraph<B> {
    pub fn new() -> Self {
        Self {
            instrs: SlotMap::default(),
            producers: AHashMap::default(),
            consumers: AHashMap::default(),
            dependencies: SecondaryMap::default(),
            dependents: SecondaryMap::default(),
        }
    }

    pub fn insert(&mut self, instr: B::Instr) -> InstrNodeId {
        let inputs = instr.inputs();
        let outputs = instr.outputs();

        let id = self.instrs.insert(instr);

        for input in inputs {
            self.consumers.entry(input).or_default().insert(id);
            if let Some(producer) = self.producers.get(&input) {
                self.dependents[*producer].insert(id);
            }
        }
        for output in outputs {
            assert!(self.producers.insert(output, id).is_none());
            for consumer in self.consumers.get(&output).into_iter().flatten().copied() {
                self.dependencies[consumer].insert(id);
            }
        }

        let dependents = self.compute_instr_dependents(id);
        self.dependents.insert(id, dependents.collect());
        let dependencies = self.compute_instr_dependencies(id);
        self.dependencies.insert(id, dependencies.collect());

        id
    }

    pub fn remove(&mut self, id: InstrNodeId) -> B::Instr {
        let instr = self.instrs.remove(id).unwrap();
        for input in instr.inputs() {
            self.consumers.get_mut(&input).unwrap().remove(&id);
            if let Some(producer) = self.producers.get(&input).copied() {
                self.dependents[producer].remove(&id);
            }
        }
        for output in instr.outputs() {
            self.producers.remove(&output).unwrap();
            for consumer in self.consumers.get(&output).into_iter().flatten().copied() {
                self.dependencies[consumer].remove(&id);
            }
        }
        instr
    }

    pub fn get(&self, instr: InstrNodeId) -> &B::Instr {
        &self.instrs[instr]
    }

    pub fn instr_dependencies(&self, instr: InstrNodeId) -> impl Iterator<Item = InstrNodeId> + '_ {
        self.dependencies[instr].iter().copied()
    }

    pub fn instr_dependents(&self, instr: InstrNodeId) -> impl Iterator<Item = InstrNodeId> + '_ {
        self.dependents[instr].iter().copied()
    }

    fn compute_instr_dependencies(
        &self,
        instr: InstrNodeId,
    ) -> impl Iterator<Item = InstrNodeId> + '_ {
        self.get(instr)
            .inputs()
            .into_iter()
            .filter_map(|id| self.producers.get(&id).copied())
    }

    fn compute_instr_dependents(
        &self,
        instr: InstrNodeId,
    ) -> impl Iterator<Item = InstrNodeId> + '_ {
        self.get(instr)
            .outputs()
            .into_iter()
            .flat_map(|id| self.consumers.get(&id).into_iter().flatten().copied())
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
        self.instrs[a].can_fuse_with(&self.instrs[b], op_graph) && !self.exist_multiple_paths(a, b)
    }

    #[profiling::function]
    fn exist_multiple_paths(&self, a: InstrNodeId, b: InstrNodeId) -> bool {
        let mut stack = self
            .instr_dependents(a)
            .filter(|d| *d != b)
            .collect::<Vec<_>>();
        let mut visited = AHashSet::new();
        visited.extend(stack.iter().copied());
        while let Some(current) = stack.pop() {
            if current == b {
                return true;
            }
            stack.extend(
                self.instr_dependents(current)
                    .filter(|next| visited.insert(*next)),
            );
        }
        false
    }

    /// Fuses two instructions. `b` must depend on `a`.
    ///
    /// `a` and `b` are removed from the graph, producing a new instruction.
    pub fn fuse_instrs(
        &mut self,
        a: InstrNodeId,
        b: InstrNodeId,
        op_graph: &Arc<OpGraph>,
    ) -> InstrNodeId {
        debug_assert!(self.get(a).can_fuse_with(self.get(b), op_graph));

        let a_instr = self.remove(a);
        let b_instr = self.remove(b);
        let new_instr = a_instr.fuse_with(&b_instr, op_graph);

        self.insert(new_instr)
    }
}

slotmap::new_key_type! {
     struct InstrNodeId;
}

#[profiling::function]
fn do_fusions<B: Backend>(graph: &mut InstrGraph<B>, op_graph: &Arc<OpGraph>) {
    let mut working_set: BTreeSet<InstrNodeId> = graph.instrs.keys().collect();

    'outer: while let Some(current) = working_set.pop_first() {
        for next in graph.instr_dependents(current).collect::<Vec<_>>() {
            if graph.can_fuse_instrs(current, next, op_graph) {
                let new = graph.fuse_instrs(current, next, op_graph);
                working_set.remove(&next);
                working_set.insert(new);
                continue 'outer;
            }
        }
        /*for prev in graph.instr_dependencies(current).collect::<Vec<_>>() {
            for sibling in graph
                .instr_dependents(prev)
                .filter(|id| *id != current)
                .collect::<Vec<_>>()
            {
                if graph.can_fuse_instrs(current, sibling, op_graph) {
                    let new = graph.fuse_instrs(current, sibling, op_graph);
                    working_set.remove(&sibling);
                    working_set.insert(new);
                    continue 'outer;
                }
            }
        }*/
    }
}

#[profiling::function]
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
#[profiling::function]
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
        "cycle detected in instruction graph?"
    );

    let mut plan = Plan { steps };
    delay_computations(&mut plan, op_graph);
    analyze_tensor_lifetimes(&mut plan, op_graph);
    plan
}

/// Delays instructions in a plan as long as possible
/// to reduce memory use.
fn delay_computations<B: Backend>(plan: &mut Plan<B>, _op_graph: &OpGraph) {
    for i in (0..plan.steps.len()).rev() {
        let mut instrs = mem::take(&mut plan.steps[i].instrs);

        instrs.retain(|instr| {
            let outputs = instr.outputs();
            let mut target_step = None;
            for j in (i + 1)..plan.steps.len() {
                if plan.steps[j]
                    .instrs
                    .iter()
                    .any(|instr2| has_intersection(&instr2.inputs(), &outputs))
                {
                    target_step = Some(j - 1);
                    break;
                }
            }
            match target_step {
                Some(s) if s != i => {
                    plan.steps[s].instrs.push(instr.clone());
                    false
                }
                _ => true,
            }
        });

        plan.steps[i].instrs = instrs;
    }
}

fn has_intersection<T: Eq>(a: &[T], b: &[T]) -> bool {
    a.iter().any(|ai| b.iter().any(|bi| ai == bi))
}

#[profiling::function]
fn analyze_tensor_lifetimes<B: Backend>(plan: &mut Plan<B>, op_graph: &OpGraph) {
    let mut tensor_release_steps = Vec::<Vec<NodeId>>::new();
    tensor_release_steps.push(Vec::new()); // no frees on first step

    for (i, step) in plan.steps().iter().enumerate() {
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

    #[cfg(debug_assertions)]
    {
        let freed = plan
            .steps
            .iter()
            .flat_map(|s| s.tensors_to_release.iter().copied())
            .collect::<AHashSet<_>>();
        assert!(freed.iter().all(|f| {
            !matches!(op_graph.get(*f), Node::Output(_))
                && !op_graph
                    .outbound_edges(*f)
                    .iter()
                    .any(|&b| matches!(op_graph.get(b), Node::Output(_)))
        }));
        let not_freed = plan
            .steps
            .iter()
            .flat_map(|step| step.instrs.iter().flat_map(|instr| instr.outputs()))
            .chain(op_graph.inputs().iter().copied())
            .filter(|node| !freed.contains(node))
            .collect::<AHashSet<_>>();
        assert!(not_freed.iter().all(|id| {
            op_graph
                .outbound_edges(*id)
                .iter()
                .any(|&b| matches!(op_graph.get(b), Node::Output(_)))
        }))
    }
}
