use crate::{
    opgraph::{Op, OpGraph, OpNode},
    subdivision::NodePortion,
    util::order_independent_hash,
};
use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use itertools::{Either, Itertools};
use std::{
    collections::{BTreeMap, BTreeSet},
    hash::{BuildHasher, Hash, Hasher},
    iter::once,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Sidecar representation indicating fusion choices.
///
/// Invariant: every node appears in at least one fusion (even
/// if it is an "identity fusion," i.e. a kernel of only one node).
///
/// For each core node, we store the set of auxiliary nodes that were
/// mainloop fused to it, and the set of nodes that were epilogue
/// fused to it. Note that a node may be fused to multiple core nodes,
/// indicating recomputation.
///
/// We additionally store the set of auxiliary fusions,
/// i.e. fusions that lack a core node. Again, note that
/// a node may appear in multiple fusions, indicating recomputation.
#[derive(Debug, Clone)]
pub struct FusedGraph {
    opgraph: Arc<OpGraph>,
    fusions: BTreeMap<FusionId, Fusion>,
    /// Maps nodes to the set of fusions they are contained in.
    /// (For core nodes, they are always mapped to exactly
    /// one fusion).
    pub node_fusions: HashMap<OpNode, Vec<FusionId>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FusionId(pub u64);

impl FusionId {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::SeqCst))
    }
}

impl Hash for FusedGraph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        order_independent_hash(self.fusions.iter().map(|(_, fusion)| fusion), state)
    }
}

impl FusedGraph {
    /// Returns the identity fusion for the given op graph.
    /// In this fused graph, no fusions are performed
    /// (in our representation, each node appears only in
    /// its own "fusion").
    pub fn identity(opgraph: &Arc<OpGraph>) -> Self {
        let mut fusions = BTreeMap::new();
        let mut node_fusions = HashMap::new();

        for node in opgraph.nodes() {
            let fusion = FusionId::new();
            fusions.insert(
                fusion,
                Fusion {
                    kind: if opgraph.get(node).is_anchor() {
                        FusionKind::Core(CoreFusion {
                            mainloop_fused: vec![],
                            anchor: node,
                            epilogue_fused: vec![],
                        })
                    } else {
                        FusionKind::Aux(AuxFusion {
                            fused_nodes: vec![node],
                        })
                    },
                    dependencies: BTreeMap::new(),
                    dependents: BTreeSet::new(),
                },
            );

            node_fusions.insert(node, vec![fusion]);
        }

        for (&node, fusion) in &node_fusions {
            let fusion = fusion[0];
            fusions.get_mut(&fusion).unwrap().dependencies.extend(
                opgraph
                    .inbound_edges(node)
                    .map(|node| (node, node_fusions[&node][0])),
            );
            fusions.get_mut(&fusion).unwrap().dependents.extend(
                opgraph
                    .outbound_edges(node)
                    .map(|node2| (node, node_fusions[&node2][0])),
            );
        }

        let graph = Self {
            fusions,
            node_fusions,
            opgraph: opgraph.clone(),
        };
        graph.assert_invariants();
        graph
    }

    pub fn produce_maximal_fusions(&mut self) {
        'outer: loop {
            let possible_actions = self.enumerate_possible_actions();

            'action_loop: for action in &possible_actions {
                if let Action::CoreFuseIntoEpilogue { node, fusion, .. } = action {
                    let old_fusions = &self.node_fusions[node];
                    if old_fusions.len() != 1 {
                        continue;
                    }
                    let old_fusion = old_fusions[0];
                    if self.fusions[&old_fusion].nodes().count() > 1 {
                        continue;
                    }
                    let dependents = self.fusions[&old_fusion]
                        .dependents
                        .iter()
                        .map(|(_, f)| *f)
                        .collect::<BTreeSet<_>>();

                    // attempt to transition all edges
                    let mut new_graph = self.clone();
                    new_graph.apply_action(action);
                    for dependent in dependents {
                        let possible_actions = new_graph.enumerate_possible_actions();
                        let desired_action = Action::TransitionEdge {
                            fusion: dependent,
                            new_provider: *fusion,
                            input: *node,
                        };
                        if !possible_actions.contains(&desired_action) {
                            continue 'action_loop;
                        }
                        new_graph.apply_action(&desired_action);
                    }

                    // Success
                    *self = new_graph;
                    continue 'outer;
                }
            }
            //
            // 'action_loop: for action in &possible_actions {
            //     if let Action::CoreFuseIntoMainloop { node, fusion, .. } = action {
            //         let old_fusions = &self.node_fusions[node];
            //         if old_fusions.len() != 1 {
            //             continue;
            //         }
            //         let old_fusion = old_fusions[0];
            //         if self.fusions[&old_fusion].nodes().count() > 1 {
            //             continue;
            //         }
            //         let dependents = self.fusions[&old_fusion]
            //             .dependents
            //             .iter()
            //             .map(|(_, f)| *f)
            //             .collect::<BTreeSet<_>>();
            //         if dependents.len() > 1 {
            //             continue 'action_loop;
            //         }
            //         assert_eq!(dependents.iter().next().unwrap(), fusion);
            //
            //         // Success
            //         self.apply_action(action);
            //         continue 'outer;
            //     }
            // }

            // Aux
            'action_loop: for action in &possible_actions {
                if let Action::AuxFuse { node, fusion, .. } = action {
                    let old_fusions = &self.node_fusions[node];
                    if old_fusions.len() != 1 {
                        continue;
                    }
                    let old_fusion = old_fusions[0];
                    if self.fusions[&old_fusion].nodes().count() > 1 {
                        continue;
                    }
                    let dependents = self.fusions[&old_fusion]
                        .dependents
                        .iter()
                        .map(|(_, f)| *f)
                        .collect::<BTreeSet<_>>();

                    // attempt to transition all edges
                    let mut new_graph = self.clone();
                    new_graph.apply_action(action);
                    for dependent in dependents {
                        let possible_actions = new_graph.enumerate_possible_actions();
                        let desired_action = Action::TransitionEdge {
                            fusion: dependent,
                            new_provider: *fusion,
                            input: *node,
                        };
                        if !possible_actions.contains(&desired_action) {
                            continue 'action_loop;
                        }
                        new_graph.apply_action(&desired_action);
                    }

                    // Success
                    *self = new_graph;
                    continue 'outer;
                }
            }

            break;
        }
        self.assert_invariants();
    }

    pub fn iter(&self) -> impl Iterator<Item = FusionId> {
        self.fusions.iter().map(|(i, _)| *i)
    }

    pub fn get(&self, fusion: FusionId) -> &Fusion {
        &self.fusions[&fusion]
    }

    pub fn opgraph(&self) -> &Arc<OpGraph> {
        &self.opgraph
    }

    pub fn clean(&mut self) {
        self.assert_invariants();
        let mut stack = self
            .fusions
            .iter()
            .filter(|(_, f)| {
                f.is_input_output(&self.opgraph)
                    && matches!(
                        self.opgraph.get(f.nodes().next().unwrap()),
                        Op::Consumer { .. }
                    )
            })
            .map(|(id, _)| *id)
            .collect_vec();
        let mut visited: HashSet<_> = stack.iter().copied().collect();

        while let Some(current) = stack.pop() {
            for inbound in self.fusions[&current].dependencies.values() {
                if visited.insert(*inbound) {
                    stack.push(*inbound);
                }
            }
        }

        let remove = self
            .fusions
            .keys()
            .copied()
            .filter(|id| !visited.contains(id))
            .collect_vec();

        for fusion_id in remove {
            self.fusions.remove(&fusion_id);
            self.node_fusions
                .values_mut()
                .for_each(|v| maybe_vec_remove_item(v, &fusion_id));
            self.fusions.values_mut().for_each(|f| {
                f.dependencies.retain(|_, f| *f != fusion_id);
                f.dependents.retain(|(_, f)| *f != fusion_id);
            });
        }

        for (id, fusion) in &mut self.fusions {
            fusion.clean(&self.opgraph, *id, &mut self.node_fusions);
        }
    }

    /// Computes the set of legal `Action`s on the current graph.
    pub fn enumerate_possible_actions(&self) -> Vec<Action> {
        let mut possible_actions = Vec::new();
        let mut cycle_cache = HashMap::new();

        for (fusion_id, fusion) in &self.fusions {
            let fusion_id = *fusion_id;

            // ----
            // Edge transitions
            // ----
            for (dependency_node, dependency_provider) in &fusion.dependencies {
                for &alt_provider in &self.node_fusions[dependency_node] {
                    if alt_provider == *dependency_provider {
                        continue;
                    }

                    // Mainloop can't provide dependencies
                    if let FusionKind::Core(core) = &self.fusions[&alt_provider].kind
                        && core.mainloop_fused.contains(dependency_node)
                    {
                        continue;
                    }

                    if !*cycle_cache
                        .entry((alt_provider, fusion_id))
                        .or_insert_with(|| self.would_create_cycle(alt_provider, fusion_id))
                    {
                        possible_actions.push(Action::TransitionEdge {
                            fusion: fusion_id,
                            input: *dependency_node,
                            new_provider: alt_provider,
                        });
                    }
                }
            }

            if let FusionKind::Aux(fusion) = &fusion.kind
                && fusion.fused_nodes.len() == 1
                && self.opgraph.get(fusion.fused_nodes[0]).is_input_output()
            {
                continue;
            }

            // ----
            // Fusion expansions
            // ----
            match &fusion.kind {
                FusionKind::Core(core_fusion) => {
                    self.enumerate_epilogue_expansions(
                        fusion_id,
                        Some(core_fusion.anchor),
                        &core_fusion.epilogue_fused,
                        &core_fusion.mainloop_fused,
                        &mut possible_actions,
                        &mut cycle_cache,
                    );
                    self.enumerate_mainloop_expansions(
                        fusion_id,
                        core_fusion.anchor,
                        &core_fusion.mainloop_fused,
                        &mut possible_actions,
                        &mut cycle_cache,
                    );
                }
                FusionKind::Aux(aux_fusion) => {
                    self.enumerate_epilogue_expansions(
                        fusion_id,
                        None,
                        &aux_fusion.fused_nodes,
                        &[],
                        &mut possible_actions,
                        &mut cycle_cache,
                    );
                }
            }
        }

        possible_actions
    }

    /// Adds the set of legal epilogue fusion expansions for the given fusion
    /// (whether core or aux).
    fn enumerate_epilogue_expansions(
        &self,
        fusion_id: FusionId,
        core_node: Option<OpNode>,
        existing_epilogue: &[OpNode],
        existing_mainloop: &[OpNode],
        possible_actions: &mut Vec<Action>,
        cycle_cache: &mut HashMap<(FusionId, FusionId), bool>,
    ) {
        let existing_nodes = match core_node {
            Some(core_node) => {
                Either::Left(existing_epilogue.iter().copied().chain(once(core_node)))
            }
            None => Either::Right(existing_epilogue.iter().copied()),
        };

        let mut visited = HashSet::new();
        for existing_node_id in existing_nodes {
            let existing_node = self.opgraph.get(existing_node_id);
            if existing_node.is_fusion_terminator() {
                continue;
            }

            for (candidate_id, is_predecessor) in self
                .opgraph
                .outbound_edges(existing_node_id)
                .map(|node| (node, false))
                .chain(
                    self.opgraph
                        .inbound_edges(existing_node_id)
                        .map(|node| (node, true)),
                )
            {
                if !visited.insert(candidate_id) {
                    continue;
                }

                if is_predecessor && Some(existing_node_id) == core_node {
                    continue;
                }

                if existing_epilogue.contains(&candidate_id)
                    || existing_mainloop.contains(&candidate_id)
                    || Some(candidate_id) == core_node
                {
                    continue;
                }

                let candidate = self.opgraph.get(candidate_id);
                if !candidate.is_epilogue_fusible() {
                    continue;
                }
                if is_predecessor && candidate.is_fusion_terminator() {
                    continue;
                }

                const HEURISTIC_MAX_NODE_RECOMPUTATIONS: usize = 3;
                if self.node_fusions[&candidate_id].len() >= HEURISTIC_MAX_NODE_RECOMPUTATIONS {
                    continue;
                }

                // Cycle condition:
                // For a node to be epilogue fusible into fusion F_1,
                // all its dependencies must either be part of F_1
                // or be part of a fusion F_2 where adding
                // an edge from F_2 to F_1 would not create a cycle.
                let mut unmet_dependencies = Vec::new();
                for successor_dependency_id in self.opgraph.inbound_edges(candidate_id) {
                    if core_node != Some(successor_dependency_id)
                        && !existing_epilogue.contains(&successor_dependency_id)
                    {
                        unmet_dependencies.push(successor_dependency_id);
                    }
                }

                for dependency_providers in unmet_dependencies
                    .iter()
                    .copied()
                    .map(|dep| {
                        self.enumerate_dependency_providers(dep, fusion_id, cycle_cache)
                            .into_iter()
                            .map(move |provider| (dep, provider))
                    })
                    .multi_cartesian_product()
                {
                    if dependency_providers.is_empty() && !unmet_dependencies.is_empty() {
                        continue;
                    }
                    possible_actions.push(if core_node.is_some() {
                        Action::CoreFuseIntoEpilogue {
                            node: candidate_id,
                            fusion: fusion_id,
                            dependency_providers: dependency_providers.into_iter().collect(),
                        }
                    } else {
                        Action::AuxFuse {
                            node: candidate_id,
                            fusion: fusion_id,
                            dependency_providers: dependency_providers.into_iter().collect(),
                        }
                    });
                }
            }
        }
    }

    fn enumerate_mainloop_expansions(
        &self,
        fusion_id: FusionId,
        core_node: OpNode,
        existing_mainloop: &[OpNode],
        possible_actions: &mut Vec<Action>,
        cycle_cache: &mut HashMap<(FusionId, FusionId), bool>,
    ) {
        let existing_nodes = existing_mainloop.iter().copied().chain(once(core_node));

        let mut visited = HashSet::new();
        for existing_node_id in existing_nodes {
            for predecessor_id in self.opgraph.inbound_edges(existing_node_id) {
                if !visited.insert(predecessor_id) {
                    continue;
                }

                if existing_mainloop.contains(&predecessor_id) {
                    continue;
                }

                if self.node_fusions[&predecessor_id].len() >= 3 {
                    continue;
                }

                let predecessor = self.opgraph.get(predecessor_id);
                if !predecessor.is_mainloop_fusible() {
                    continue;
                }

                // Cycle condition:
                // For a node to be fusible into fusion F_1,
                // all its dependencies must either be part of F_1
                // or be part of a fusion F_2 where adding
                // an edge from F_2 to F_1 would not create a cycle.
                let mut unmet_dependencies = Vec::new();
                for successor_dependency_id in self.opgraph.inbound_edges(predecessor_id) {
                    if !existing_mainloop.contains(&successor_dependency_id) {
                        unmet_dependencies.push(successor_dependency_id);
                    }
                }

                for dependency_providers in unmet_dependencies
                    .iter()
                    .copied()
                    .map(|dep| {
                        self.enumerate_dependency_providers(dep, fusion_id, cycle_cache)
                            .into_iter()
                            .map(move |provider| (dep, provider))
                    })
                    .multi_cartesian_product()
                {
                    if dependency_providers.is_empty() && !unmet_dependencies.is_empty() {
                        continue;
                    }

                    possible_actions.push(Action::CoreFuseIntoMainloop {
                        node: predecessor_id,
                        fusion: fusion_id,
                        dependency_providers: dependency_providers.into_iter().collect(),
                    });
                }
            }
        }
    }

    fn enumerate_dependency_providers(
        &self,
        dependency: OpNode,
        fusion_id: FusionId,
        cycle_cache: &mut HashMap<(FusionId, FusionId), bool>,
    ) -> Vec<FusionId> {
        // If this fusion already has this dependency,
        // then use that fusion - zero reason to ever
        // use multiple providers for the same dependency
        if let Some(existing_provider) = self.fusions[&fusion_id].dependencies.get(&dependency) {
            return vec![*existing_provider];
        }

        let mut providers = Vec::new();

        for &provider_fusion_id in &self.node_fusions[&dependency] {
            if let FusionKind::Core(core) = &self.fusions[&provider_fusion_id].kind
                && core.mainloop_fused.contains(&dependency)
            {
                // Mainloop can't provide dependencies
                continue;
            }

            let would_create_cycle = *cycle_cache
                .entry((provider_fusion_id, fusion_id))
                .or_insert_with(|| self.would_create_cycle(provider_fusion_id, fusion_id));
            if !would_create_cycle {
                providers.push(provider_fusion_id);
            }
        }

        providers
    }

    /// Returns whether introducing an edge from `fusion1`
    /// to `fusion2` would create a cycle.
    fn would_create_cycle(&self, fusion1: FusionId, fusion2: FusionId) -> bool {
        // PERF - can use a cached topological ordering to pre-check

        let mut stack = vec![fusion2];
        let mut visited = HashSet::new();
        while let Some(current) = stack.pop() {
            if current == fusion1 {
                return true;
            }

            for &(_, next) in &self.fusions[&current].dependents {
                if visited.insert(next) {
                    stack.push(next);
                }
            }
        }

        false
    }

    /// Applies an `Action`.
    pub fn apply_action(&mut self, action: &Action) {
        debug_assert!(
            self.enumerate_possible_actions().contains(action),
            "missing action {action:#?} for graph {self:#?}: legal are {:#?}",
            self.enumerate_possible_actions()
        );
        match action {
            Action::CoreFuseIntoEpilogue {
                node,
                fusion: fusion_id,
                dependency_providers,
            } => {
                let previous_provider = self.fusions[fusion_id].dependencies.get(node).copied();

                for (dependency, provider) in dependency_providers {
                    self.fusions
                        .get_mut(&fusion_id)
                        .unwrap()
                        .dependencies
                        .insert(*dependency, *provider);
                    self.fusions
                        .get_mut(&provider)
                        .unwrap()
                        .dependents
                        .insert((*dependency, *fusion_id));
                }

                if let Some(previous_provider) = previous_provider {
                    self.remove_edge(*node, previous_provider, *fusion_id);
                }

                self.node_fusions.get_mut(&node).unwrap().push(*fusion_id);
                let FusionKind::Core(fusion) = &mut self.fusions.get_mut(fusion_id).unwrap().kind
                else {
                    panic!()
                };
                let pos = fusion.epilogue_fused.binary_search(&node).unwrap_err();
                fusion.epilogue_fused.insert(pos, *node);

                for other_fusion in self.node_fusions[node].clone() {
                    if other_fusion != *fusion_id {
                        self.maybe_remove_collapsed_fusion(other_fusion);
                    }
                }
            }
            Action::CoreFuseIntoMainloop {
                node,
                fusion: fusion_id,
                dependency_providers,
            } => {
                let previous_provider = self.fusions[fusion_id].dependencies.get(node).copied();

                for (dependency, provider) in dependency_providers {
                    self.fusions
                        .get_mut(fusion_id)
                        .unwrap()
                        .dependencies
                        .insert(*dependency, *provider);
                    self.fusions
                        .get_mut(provider)
                        .unwrap()
                        .dependents
                        .insert((*dependency, *fusion_id));
                }

                if let Some(previous_provider) = previous_provider {
                    self.remove_edge(*node, previous_provider, *fusion_id);
                }

                self.node_fusions.get_mut(&node).unwrap().push(*fusion_id);
                let FusionKind::Core(fusion) = &mut self.fusions.get_mut(&fusion_id).unwrap().kind
                else {
                    panic!()
                };
                let pos = fusion.mainloop_fused.binary_search(&node).unwrap_err();
                fusion.mainloop_fused.insert(pos, *node);

                for other_fusion in self.node_fusions[node].clone() {
                    if other_fusion != *fusion_id {
                        self.maybe_remove_collapsed_fusion(other_fusion);
                    }
                }
            }
            Action::AuxFuse {
                node,
                fusion: fusion_id,
                dependency_providers,
            } => {
                let previous_provider = self
                    .fusions
                    .get_mut(&fusion_id)
                    .unwrap()
                    .dependencies
                    .get(node)
                    .copied();

                for (dependency, provider) in dependency_providers {
                    self.fusions
                        .get_mut(fusion_id)
                        .unwrap()
                        .dependencies
                        .insert(*dependency, *provider);
                    self.fusions
                        .get_mut(provider)
                        .unwrap()
                        .dependents
                        .insert((*dependency, *fusion_id));
                }

                if let Some(previous_provider) = previous_provider {
                    self.remove_edge(*node, previous_provider, *fusion_id);
                }

                self.node_fusions.get_mut(&node).unwrap().push(*fusion_id);
                let FusionKind::Aux(fusion) = &mut self.fusions.get_mut(fusion_id).unwrap().kind
                else {
                    panic!()
                };
                let pos = fusion.fused_nodes.binary_search(&node).unwrap_err();
                fusion.fused_nodes.insert(pos, *node);

                for other_fusion in self.node_fusions[node].clone() {
                    if other_fusion != *fusion_id {
                        self.maybe_remove_collapsed_fusion(other_fusion);
                    }
                }
            }
            Action::TransitionEdge {
                fusion,
                input,
                new_provider,
            } => {
                let previous_provider = self
                    .fusions
                    .get_mut(fusion)
                    .unwrap()
                    .dependencies
                    .remove(&input)
                    .unwrap();
                self.fusions
                    .get_mut(&previous_provider)
                    .unwrap()
                    .dependents
                    .remove(&(*input, *fusion));
                self.fusions
                    .get_mut(fusion)
                    .unwrap()
                    .dependencies
                    .insert(*input, *new_provider);
                self.fusions
                    .get_mut(new_provider)
                    .unwrap()
                    .dependents
                    .insert((*input, *fusion));

                self.maybe_remove_collapsed_fusion(previous_provider);
            }
        }
    }

    fn remove_edge(&mut self, node: OpNode, provider: FusionId, consumer: FusionId) {
        self.fusions
            .get_mut(&provider)
            .unwrap()
            .dependents
            .remove(&(node, consumer));
        self.fusions
            .get_mut(&consumer)
            .unwrap()
            .dependencies
            .remove(&node);

        // if the provider no longer provides this node to _anyone_,
        // and is not the only provider for it, then we can
        // recalculate to remove unused portion of the provider's
        // graph and propagate the changes
        if self.node_fusions[&node].len() > 1
            && !self.fusions[&provider]
                .dependents
                .iter()
                .any(|(n, _)| *n == node)
        {
            // recalculate which dependencies this fusion has
            // and remove unneeded nodes
            let mut stack = self.fusions[&provider]
                .unique_dependent_nodes()
                .collect::<Vec<_>>();
            let mut visited = stack.iter().copied().collect::<HashSet<_>>();

            while let Some(current) = stack.pop() {
                for prev in self.opgraph.inbound_edges(current) {
                    if visited.insert(prev) && self.fusions[&provider].contains_node(prev) {
                        stack.push(prev);
                    }
                }
            }

            for node in self.fusions[&provider].nodes().collect::<Vec<_>>() {
                if !visited.contains(&node) && !self.opgraph.get(node).is_anchor() {
                    self.fusions.get_mut(&provider).unwrap().remove_node(node);
                    vec_remove_item(self.node_fusions.get_mut(&node).unwrap(), &provider);
                }
            }

            for (dep, provider2) in self.fusions[&provider].dependencies.clone() {
                if !visited.contains(&dep) {
                    self.remove_edge(dep, provider2, provider);
                }
            }
        }

        self.maybe_remove_collapsed_fusion(provider);
    }

    fn maybe_remove_collapsed_fusion(&mut self, fusion_id: FusionId) {
        if self.fusions[&fusion_id].dependents.is_empty() {
            for node in self.fusions[&fusion_id].nodes() {
                if self.node_fusions[&node].len() == 1 {
                    return;
                }
            }

            let fusion = self.fusions.remove(&fusion_id).unwrap();
            for node in fusion.nodes() {
                vec_remove_item(self.node_fusions.get_mut(&node).unwrap(), &fusion_id);
            }

            for (dependency, provider) in fusion.dependencies {
                self.fusions
                    .get_mut(&provider)
                    .unwrap()
                    .dependents
                    .remove(&(dependency, fusion_id));
            }
        }
    }

    #[allow(unused)]
    pub fn assert_invariants(&self) {
        for (id, fusion) in &self.fusions {
            let id = *id;
            for node in fusion.nodes() {
                assert!(self.node_fusions[&node].contains(&id));
                let dependencies = self.opgraph.inbound_edges(node);
                for dependency in dependencies {
                    let contains_dependency = fusion.contains_node(dependency);
                    if !contains_dependency {
                        let provider = &self.fusions[&fusion
                            .dependencies
                            .get(&dependency)
                            .unwrap_or_else(|| {
                                dbg!(
                                    fusion,
                                    node,
                                    dependency,
                                    self.opgraph.get(node),
                                    self.opgraph.get(dependency)
                                );
                                panic!();
                            })];
                        assert!(provider.contains_node(dependency));
                        assert!(
                            provider.dependents.contains(&(dependency, id)),
                            "p={:?} d={dependency:?} u={id:?}",
                            fusion.dependencies[&dependency]
                        );
                    }
                }
            }
        }

        for (node, fusions) in &self.node_fusions {
            for fusion in fusions {
                assert!(self.fusions[fusion].contains_node(*node));
            }
        }

        for node in self.opgraph.nodes() {
            assert!(!self.node_fusions[&node].is_empty());
        }
    }

    pub fn topo_sort(&self) -> Vec<FusionId> {
        self.assert_invariants();
        let mut result = Vec::new();

        let mut stack = self
            .fusions
            .iter()
            .filter(|(_, f)| f.dependencies.is_empty())
            .map(|(id, _)| *id)
            .collect_vec();

        let mut visited = stack.iter().copied().collect::<HashSet<_>>();

        while let Some(current) = stack.pop() {
            result.push(current);
            for dependent in self.get(current).dependents.iter().map(|(_, f)| *f) {
                if self
                    .get(dependent)
                    .dependencies
                    .values()
                    .all(|&f| result.contains(&f))
                {
                    if visited.insert(dependent) {
                        stack.push(dependent);
                    }
                }
            }
        }

        result
    }

    pub fn merge_from_subdivisions(
        subdivisions: &[BTreeSet<(OpNode, NodePortion)>],
        subdivision_fusions: &[FusedGraph],
        original_graph: &Arc<OpGraph>,
    ) -> Self {
        subdivision_fusions
            .iter()
            .for_each(|f| f.assert_invariants());

        let mut fusions = BTreeMap::<FusionId, Fusion>::new();

        let mut merged = HashMap::<OpNode, FusionId>::new();

        let mut collapsed = HashSet::new();

        // first pass: merge anchor nodes that appear in two subdivisions
        for (subdivision, subdivision_fusions) in subdivisions.iter().zip(subdivision_fusions) {
            for (node, portion) in subdivision {
                if matches!(portion, NodePortion::Epilogue | NodePortion::Mainloop) {
                    let corresponding_fusion_id = subdivision_fusions.node_fusions[node][0];
                    let corresponding_fusion =
                        &subdivision_fusions.fusions[&corresponding_fusion_id];
                    let FusionKind::Core(corresponding_fusion_core) = &corresponding_fusion.kind
                    else {
                        unreachable!()
                    };

                    if let Some(existing_fusion_id) = merged.get(node).copied() {
                        let existing_fusion = fusions.get_mut(&existing_fusion_id).unwrap();
                        let FusionKind::Core(core) = &mut existing_fusion.kind else {
                            unreachable!()
                        };
                        match portion {
                            NodePortion::Epilogue => {
                                core.epilogue_fused =
                                    corresponding_fusion_core.epilogue_fused.clone();
                                existing_fusion
                                    .dependents
                                    .extend(corresponding_fusion.dependents.clone());
                            }
                            NodePortion::Mainloop => {
                                core.mainloop_fused =
                                    corresponding_fusion_core.mainloop_fused.clone();
                                existing_fusion
                                    .dependencies
                                    .extend(corresponding_fusion.dependencies.clone());
                            }
                            _ => unreachable!(),
                        }

                        if existing_fusion_id != corresponding_fusion_id {
                            collapsed.insert(corresponding_fusion_id);
                        }
                    } else {
                        fusions.insert(corresponding_fusion_id, corresponding_fusion.clone());
                        merged.insert(*node, corresponding_fusion_id);
                    }
                }
            }
        }

        // second pass: remaining nodes
        for subdivision_fusions in subdivision_fusions {
            for (fusion_id, fusion) in &subdivision_fusions.fusions {
                if let FusionKind::Aux(aux) = &fusion.kind {
                    if aux.fused_nodes.len() == 1
                        && subdivision_fusions
                            .opgraph
                            .get(aux.fused_nodes[0])
                            .is_input_output()
                    {
                        if !original_graph.get(aux.fused_nodes[0]).is_input_output() {
                            continue;
                        } else if fusions
                            .values()
                            .any(|f| f.contains_node(aux.fused_nodes[0]))
                        {
                            continue;
                        }
                    }
                }

                if !collapsed.contains(fusion_id) && !fusions.contains_key(fusion_id) {
                    fusions.insert(*fusion_id, fusion.clone());
                }
            }
        }

        let fusions_clone = fusions.clone();
        // update edges
        let mut updates = Vec::new();
        for (fusion_id, fusion) in &mut fusions {
            for (node, provider) in &mut fusion.dependencies {
                if !fusions_clone.contains_key(provider) {
                    let new_provider = fusions_clone
                        .iter()
                        .find(|(_, f)| f.contains_node(*node))
                        .unwrap()
                        .0;
                    *provider = *new_provider;
                    updates.push((*new_provider, *node, *fusion_id));
                }
            }
            fusion.dependents = fusion
                .dependents
                .clone()
                .into_iter()
                .filter(|(_, consumer)| fusions_clone.contains_key(consumer))
                .collect();
        }

        for (new_provider, node, fusion) in updates {
            fusions
                .get_mut(&new_provider)
                .unwrap()
                .dependents
                .insert((node, fusion));
        }

        // clean
        for fusion in fusions.values_mut() {
            fusion
                .dependents
                .retain(|(_, f)| fusions_clone.contains_key(f));
        }

        // build node_fusions
        let node_fusions: HashMap<OpNode, Vec<_>> = original_graph
            .nodes()
            .map(|node| {
                let fusions = fusions
                    .iter()
                    .filter(|(_, f)| f.contains_node(node))
                    .map(|(id, _)| *id)
                    .collect::<Vec<_>>();
                (node, fusions)
            })
            .collect();

        assert!(original_graph.nodes().all(|n| !node_fusions[&n].is_empty()));

        let graph = FusedGraph {
            opgraph: original_graph.clone(),
            fusions,
            node_fusions,
        };
        graph.assert_invariants();
        graph
    }

    pub fn print(&self) {
        for id in self.topo_sort() {
            if !self.fusions[&id].is_input_output(&self.opgraph) {
                self.fusions[&id].print(id, &self.opgraph, self);
                println!();
            }
        }
    }
}

pub fn vec_remove_item<T: Eq>(vec: &mut Vec<T>, item: &T) {
    let pos = vec.iter().position(|x| *x == *item).unwrap();
    vec.remove(pos);
}

pub fn maybe_vec_remove_item<T: Eq>(vec: &mut Vec<T>, item: &T) {
    if let Some(pos) = vec.iter().position(|x| x == item) {
        vec.remove(pos);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fusion {
    pub dependencies: BTreeMap<OpNode, FusionId>,
    pub dependents: BTreeSet<(OpNode, FusionId)>,
    pub kind: FusionKind,
}

impl Fusion {
    pub fn clean(
        &mut self,
        opgraph: &OpGraph,
        id: FusionId,
        node_fusions: &mut HashMap<OpNode, Vec<FusionId>>,
    ) {
        let mut stack = self
            .dependents
            .iter()
            .map(|(n, _)| *n)
            .unique()
            .collect_vec();
        let mut visited: HashSet<_> = stack.iter().copied().collect();
        while let Some(current) = stack.pop() {
            for pred in opgraph.inbound_edges(current) {
                if visited.insert(pred) && self.contains_node(pred) {
                    stack.push(pred);
                }
            }
        }

        match &mut self.kind {
            FusionKind::Core(core) => {
                core.epilogue_fused.retain(|n| {
                    if !visited.contains(n) && node_fusions[n].len() > 1 {
                        vec_remove_item(node_fusions.get_mut(n).unwrap(), &id);
                        false
                    } else {
                        true
                    }
                });
            }
            FusionKind::Aux(aux) => {
                aux.fused_nodes.retain(|n| {
                    if !visited.contains(n) && node_fusions[n].len() > 1 {
                        vec_remove_item(node_fusions.get_mut(n).unwrap(), &id);
                        false
                    } else {
                        true
                    }
                });
            }
        }
    }

    pub fn make_subgraph(&self, opgraph: &OpGraph) -> OpGraph {
        opgraph.subgraph(self.nodes())
    }

    pub fn nodes(&self) -> impl Iterator<Item = OpNode> {
        match &self.kind {
            FusionKind::Core(fusion) => Either::Left(
                fusion
                    .mainloop_fused
                    .iter()
                    .copied()
                    .chain(fusion.epilogue_fused.iter().copied())
                    .chain(once(fusion.anchor)),
            ),
            FusionKind::Aux(fusion) => Either::Right(fusion.fused_nodes.iter().copied()),
        }
    }

    pub fn print(&self, id: FusionId, opgraph: &OpGraph, _fused_graph: &FusedGraph) {
        if self.is_input_output(opgraph) {
            return;
        }

        println!("{id:?}: ");
        for (node, dependent) in self.dependents.iter() {
            println!("Dependent: {dependent:?} uses {node:?}");
        }
        match &self.kind {
            FusionKind::Aux(aux) => {
                println!("---Aux---");
                for node in &aux.fused_nodes {
                    println!("\t{:?} = {:?}", *node, opgraph.get(*node));
                }
            }
            FusionKind::Core(core) => {
                let Op::Matmul { a, b, .. } = opgraph.get(core.anchor) else {
                    unreachable!()
                };
                println!("---Core({:?} = matmul({:?}, {:?}))---", core.anchor, *a, *b);
                println!("\tMainloop:");
                for mainloop in &core.mainloop_fused {
                    println!("\t\t{:?} = {:?}", *mainloop, opgraph.get(*mainloop));
                }
                println!("\tEpilogue:");
                for epilogue in &core.epilogue_fused {
                    println!("\t\t{:?} = {:?}", *epilogue, opgraph.get(*epilogue));
                }
            }
        }
    }

    pub fn contains_node(&self, node: OpNode) -> bool {
        self.nodes().any(|n| n == node)
    }

    pub fn remove_node(&mut self, node: OpNode) {
        match &mut self.kind {
            FusionKind::Core(fusion) => {
                maybe_vec_remove_item(&mut fusion.mainloop_fused, &node);
                maybe_vec_remove_item(&mut fusion.epilogue_fused, &node);
            }
            FusionKind::Aux(fusion) => {
                maybe_vec_remove_item(&mut fusion.fused_nodes, &node);
            }
        }
    }

    pub fn is_input_output(&self, opgraph: &OpGraph) -> bool {
        self.nodes().all(|node| opgraph.get(node).is_input_output())
    }

    pub fn unique_dependent_nodes(&self) -> impl Iterator<Item = OpNode> {
        self.dependents
            .iter()
            .map(|(node, _)| *node)
            .collect::<BTreeSet<_>>()
            .into_iter()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionKind {
    Core(CoreFusion),
    Aux(AuxFusion),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CoreFusion {
    /// Sorted.
    pub mainloop_fused: Vec<OpNode>,
    pub anchor: OpNode,
    /// Sorted.
    pub epilogue_fused: Vec<OpNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AuxFusion {
    /// Sorted.
    fused_nodes: Vec<OpNode>,
}

/// Action modifying a `FusedGraph`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Action {
    /// Fuse a node into the epilogue of an existing core fusion.
    CoreFuseIntoEpilogue {
        node: OpNode,
        fusion: FusionId,
        dependency_providers: BTreeMap<OpNode, FusionId>,
    },
    /// Fuse a node into the mainloop of an existing core fusion.
    CoreFuseIntoMainloop {
        node: OpNode,
        fusion: FusionId,
        dependency_providers: BTreeMap<OpNode, FusionId>,
    },
    /// Fuse a node into an auxiliary fusion.
    AuxFuse {
        node: OpNode,
        fusion: FusionId,
        dependency_providers: BTreeMap<OpNode, FusionId>,
    },
    /// Transitions fusion `F_0` to take its input `N_0` from
    /// fusion `F_1` instead of the existing input fusion.
    TransitionEdge {
        fusion: FusionId,
        input: OpNode,
        new_provider: FusionId,
    },
}

pub fn count_state_tree_size(opgraph: &Arc<OpGraph>) -> usize {
    let fused_graph = FusedGraph::identity(opgraph);
    let mut visited_hashes: HashSet<u64> = HashSet::new();

    fn inner(fused_graph: &FusedGraph, visited_hashes: &mut HashSet<u64>) {
        for action in fused_graph.enumerate_possible_actions() {
            let mut modified_graph = fused_graph.clone();
            modified_graph.apply_action(&action);
            if visited_hashes.insert(
                foldhash::quality::FixedState::with_seed(13642023059672996567)
                    .hash_one(&modified_graph),
            ) {
                inner(&modified_graph, visited_hashes);
            }
        }
    }

    inner(&fused_graph, &mut visited_hashes);
    visited_hashes.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cost_model::{minimize_cost_function, naive_cost_function},
        opgraph::{BinaryPointwiseOp, Op, Shape, UnaryPointwiseOp},
        subdivision::subdivide,
    };
    use std::thread;

    #[test]
    fn simple_graph_actions() {
        let mut opgraph = OpGraph::new();
        let a = opgraph.insert(Op::Producer {
            shape: Shape::new([1, 256, 256]),
        });
        let b = opgraph.insert(Op::Producer {
            shape: Shape::new([1, 256, 256]),
        });
        let c = opgraph.insert(Op::BinaryPointwise {
            lhs: a,
            rhs: b,
            op: BinaryPointwiseOp::Add,
        });
        let d = opgraph.insert(Op::Matmul { a: c, b });
        let e = opgraph.insert(Op::UnaryPointwise {
            x: d,
            op: UnaryPointwiseOp::Exp,
        });
        let _f = opgraph.insert(Op::Matmul { a: d, b: e });
        let opgraph = Arc::new(opgraph);

        let mut fused_graph = FusedGraph::identity(&opgraph);

        // assert_eq!(
        //     fused_graph.enumerate_possible_actions(),
        //     vec![
        //         Action::CoreFuseIntoEpilogue {
        //             node: e,
        //             fusion: fused_graph.node_fusions[&d][0],
        //             dependency_providers: Default::default(),
        //         },
        //         Action::CoreFuseIntoMainloop {
        //             node: c,
        //             fusion: fused_graph.node_fusions[&d][0],
        //             dependency_providers: btreemap! {
        //                 a => fused_graph.node_fusions[&a][0],
        //                 b => fused_graph.node_fusions[&b][0],
        //             }
        //         },
        //         Action::CoreFuseIntoMainloop {
        //             node: e,
        //             fusion: fused_graph.node_fusions[&f][0],
        //             dependency_providers: btreemap! {
        //                 d => fused_graph.node_fusions[&d][0],
        //             }
        //         }
        //     ]
        // );

        // fused_graph.apply_action(&Action::CoreFuseIntoEpilogue {
        //     node: e,
        //     fusion: fused_graph.node_fusions[&d][0],
        //     dependency_providers: Default::default(),
        // });
        // fused_graph.apply_action(&Action::TransitionEdge {
        //     fusion: fused_graph.node_fusions[&f][0],
        //     input: e,
        //     new_provider: fused_graph.node_fusions[&d][0],
        // });
        // fused_graph.apply_action(&Action::CoreFuseIntoMainloop {
        //     fusion: fused_graph.node_fusions[&d][0],
        //     node: c,
        //     dependency_providers: btreemap! {
        //         a => fused_graph.node_fusions[&a][0],
        //         b => fused_graph.node_fusions[&b][0],
        //     },
        // });

        fused_graph.produce_maximal_fusions();
        dbg!(&fused_graph);
    }

    #[test]
    fn complex_graph() {
        thread::scope(|s| {
            thread::Builder::new()
                .stack_size(256 * 1024 * 1024)
                .spawn_scoped(s, || {
                    let mut opgraph = OpGraph::new();
                    let n0 = opgraph.insert(Op::Producer {
                        shape: Shape::new([1, 1024, 1024]),
                    });
                    let n1 = opgraph.insert(Op::Producer {
                        shape: Shape::new([1, 1024, 1024]),
                    });
                    let n2 = opgraph.insert(Op::Matmul { a: n0, b: n1 });
                    let n3 = opgraph.insert(Op::UnaryPointwise {
                        x: n2,
                        op: UnaryPointwiseOp::Exp,
                    });
                    let n4 = opgraph.insert(Op::BinaryPointwise {
                        lhs: n3,
                        rhs: n1,
                        op: BinaryPointwiseOp::Add,
                    });
                    let n5 = opgraph.insert(Op::Matmul { a: n0, b: n4 });

                    // backwards
                    let n6 = opgraph.insert(Op::Matmul { a: n5, b: n4 }); // weight update
                    let n7 = opgraph.insert(Op::Matmul { a: n5, b: n0 }); // flow
                    let n8 = opgraph.insert(Op::BinaryPointwise {
                        lhs: n3,
                        rhs: n7,
                        op: BinaryPointwiseOp::Add,
                    });
                    let n9 = opgraph.insert(Op::Matmul { a: n8, b: n2 }); // weight update

                    opgraph.insert(Op::Consumer { x: n6 });
                    opgraph.insert(Op::Consumer { x: n9 });

                    let opgraph = Arc::new(opgraph);
                    dbg!(minimize_cost_function(&opgraph, naive_cost_function));
                })
                .unwrap();
        });
    }

    #[test]
    fn merge_from_subdivisions() {
        let mut graph = OpGraph::new();

        let input = graph.insert(Op::Producer {
            shape: Shape::new([1, 256, 256]),
        });

        let mm1 = graph.insert(Op::Matmul { a: input, b: input });
        let relu1 = graph.insert(Op::UnaryPointwise {
            x: mm1,
            op: UnaryPointwiseOp::Relu,
        });
        let add1 = graph.insert(Op::BinaryPointwise {
            lhs: relu1,
            rhs: input,
            op: BinaryPointwiseOp::Add,
        });
        graph.insert(Op::Consumer { x: add1 });
        let graph = Arc::new(graph);

        let subdivisions = subdivide(&graph);
        let mut subdivision_fusions = Vec::new();
        for subdivision in &subdivisions {
            let subgraph = Arc::new(graph.subgraph(subdivision.iter().map(|(n, _)| *n)));
            subdivision_fusions.push(minimize_cost_function(&subgraph, naive_cost_function).0);
        }

        let merged =
            FusedGraph::merge_from_subdivisions(&subdivisions, &subdivision_fusions, &graph);
        dbg!(merged);
    }
}
