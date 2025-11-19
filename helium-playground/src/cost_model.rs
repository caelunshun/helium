use crate::{
    fused_graph::{FusedGraph, Fusion, FusionKind},
    opgraph::{BinaryPointwiseOp, Op, OpGraph, OpNode, UnaryPointwiseOp},
    subdivision::subdivide,
};
use foldhash::HashSet;
use std::{hash::BuildHasher, sync::Arc};

pub fn naive_cost_function(fused_graph: &FusedGraph) -> f64 {
    const MEMORY_BANDWIDTH: f64 = 2.4e11; // elements/s
    const TENSOR_FLOPS: f64 = 950.0 * 1e12;
    const REGULAR_FLOPS: f64 = 67.0 * 1e12;
    const MEMORY_TRANSFER_COST: f64 = 1.0 / MEMORY_BANDWIDTH;
    const TENSOR_FLOP_COST: f64 = 1.0 / TENSOR_FLOPS;
    const REGULAR_FLOP_COST: f64 = 1.0 / REGULAR_FLOPS;

    let mut cost = 0.0f64;
    for fusion_id in fused_graph.iter() {
        let fusion = fused_graph.get(fusion_id);
        if fusion.is_input_output(fused_graph.opgraph()) {
            continue;
        }

        let mut stack = fusion
            .dependents
            .iter()
            .map(|(node, _)| *node)
            .collect::<Vec<_>>();
        stack.sort_unstable();
        stack.dedup();

        let mut visited: HashSet<OpNode> = stack.iter().copied().collect();

        let mut tensor_flops = 0;
        let mut regular_flops = 0;

        while let Some(current) = stack.pop() {
            if fusion.dependents.iter().any(|(node, _)| *node == current) {
                // save
                let num_elements = fused_graph.opgraph().shape(current).element_count();
                cost += MEMORY_TRANSFER_COST * num_elements as f64;
            }

            // ----
            // computations (ignoring overlap etc.,
            // better model would be statistical from empirical data)
            // ----
            match fused_graph.opgraph().get(current) {
                Op::Producer { .. } => {}
                Op::Matmul { a, b } => {
                    tensor_flops += 2
                        * fused_graph.opgraph().shape(*a).element_count() as u64
                        * fused_graph.opgraph().shape(*b).dim(1) as u64;
                }
                Op::UnaryPointwise { x, .. } => {
                    regular_flops += 2 * fused_graph.opgraph().shape(*x).element_count() as u64;
                }
                Op::Reduction { x, .. } => {
                    regular_flops += 2 * fused_graph.opgraph().shape(*x).element_count() as u64;
                }
                Op::BinaryPointwise { lhs, .. } => {
                    regular_flops += 3 * fused_graph.opgraph().shape(*lhs).element_count() as u64;
                }
                Op::Constant { .. } => {}
                Op::Broadcast { .. } => {}
                Op::Consumer { .. } => {}
                Op::Transpose { .. } => {}
            }

            for prev in fused_graph.opgraph().inbound_edges(current) {
                if !visited.insert(prev) {
                    continue;
                }
                if fusion.nodes().any(|n| n == prev) {
                    stack.push(prev);
                } else if fusion.dependencies.contains_key(&prev) {
                    // dependency load
                    let num_elements = fused_graph.opgraph().shape(prev).element_count();
                    cost += MEMORY_TRANSFER_COST * num_elements as f64;
                }
            }
        }

        cost += tensor_flops as f64 * TENSOR_FLOP_COST;
        cost += regular_flops as f64 * REGULAR_FLOP_COST;
    }
    cost
}

pub fn analytical_cost_function(fused_graph: &FusedGraph) -> f64 {
    let mut cost = 0.0f64;
    for fusion_id in fused_graph.iter() {
        let fusion = fused_graph.get(fusion_id);
        if fusion.is_input_output(fused_graph.opgraph()) {
            continue;
        }

        let mut stack = fusion
            .dependents
            .iter()
            .map(|(node, _)| *node)
            .collect::<Vec<_>>();
        stack.sort_unstable();
        stack.dedup();

        let mut visited: HashSet<OpNode> = stack.iter().copied().collect();

        let mut e_c = 0;
        let mut e_m = 0;

        let mut m_ca = 0;
        let mut m_cb = 0;

        let (mainloop_a, mainloop_b, r_a, r_b) = if let FusionKind::Core(core) = &fusion.kind {
            let Op::Matmul { a, b } = fused_graph.opgraph().get(core.anchor) else {
                unreachable!()
            };
            let m = fused_graph.opgraph().shape(*a).dim(1);
            let n = fused_graph.opgraph().shape(*b).dim(2);
            (
                enumerate_ancestors_in_fusion(fusion, fused_graph.opgraph(), *a),
                enumerate_ancestors_in_fusion(fusion, fused_graph.opgraph(), *b),
                m.div_ceil(256),
                n.div_ceil(128),
            )
        } else {
            // (if unanchored fusion then no mainloop nodes, no recomputation)
            (vec![], vec![], 0, 0)
        };

        while let Some(current) = stack.pop() {
            if fusion.dependents.iter().any(|(node, _)| *node == current) {
                // save
                e_m += 2;
            }

            // ----
            // computations
            // ----
            let compute_target = if mainloop_a.contains(&current) {
                &mut m_ca
            } else if mainloop_b.contains(&current) {
                &mut m_cb
            } else {
                &mut e_c
            };
            match fused_graph.opgraph().get(current) {
                Op::Producer { .. } => {}
                Op::Matmul { .. } => {} // anchor node, not relevant for cost model (constant overhead)
                Op::UnaryPointwise { x, op } => {
                    *compute_target += match op {
                        UnaryPointwiseOp::Neg
                        | UnaryPointwiseOp::Relu
                        | UnaryPointwiseOp::AddConstant(_)
                        | UnaryPointwiseOp::MulConstant(_) => 1,
                        UnaryPointwiseOp::PowConstant(x) if *x == 2.0 => 1,
                        UnaryPointwiseOp::PowConstant(_) => 10,
                        UnaryPointwiseOp::Exp => 15,
                        UnaryPointwiseOp::Gelu => 8, // approx
                        UnaryPointwiseOp::Ln => 12,
                        UnaryPointwiseOp::Sqrt => 13,
                        UnaryPointwiseOp::Tanh => 23,
                    };
                }
                Op::Reduction { .. } => {
                    // per-warp reduction (log2(32) = 5)
                    *compute_target += 10;
                }
                Op::BinaryPointwise { op, .. } => {
                    *compute_target += match *op {
                        BinaryPointwiseOp::Add => 1,
                        BinaryPointwiseOp::Sub => 1,
                        BinaryPointwiseOp::Mul => 1,
                        BinaryPointwiseOp::Div => 5,
                        BinaryPointwiseOp::Pow => 24, // gross exp and ln combo
                        BinaryPointwiseOp::Drelu => 2,
                    };
                }
                Op::Constant { .. } => {}  // free (one-time register set)
                Op::Broadcast { .. } => {} // free (modifies view)
                Op::Consumer { .. } => {}  // free (not a computation)
                Op::Transpose { .. } => {} // free (modifies view)
            }

            for prev in fused_graph.opgraph().inbound_edges(current) {
                if !visited.insert(prev) {
                    continue;
                }
                if fusion.nodes().any(|n| n == prev) {
                    stack.push(prev);
                } else if fusion.dependencies.contains_key(&prev) {
                    // dependency load
                    e_m += 2;
                }
            }

            let m_c = r_a as u64 * m_ca + r_b as u64 * m_cb;

            // statistical model to map to execution time - Ampere
            let alpha_c = 1.32e-6;
            let alpha_m = 2.30e-5;
            let beta_0 = 0.0;
            let beta_1 = 6.55e-5;
            let alpha_1 = 3.37e-6;
            let gamma = 4.0;
            let alpha_2 = 6.60e-6;
            let beta_2 = beta_1 + gamma * alpha_1 - alpha_2 * gamma; // not a degree of freedom

            let c_epilogue = alpha_c * e_c as f64 + alpha_m * e_m as f64;
            let c_mainloop = if m_c == 0 {
                beta_0
            } else if (m_c as f64) < gamma {
                beta_1 + alpha_1 * m_c as f64
            } else {
                beta_2 + alpha_2 * m_c as f64
            };

            cost += c_epilogue + c_mainloop;
        }
    }
    cost
}

fn enumerate_ancestors_in_fusion(fusion: &Fusion, opgraph: &OpGraph, node: OpNode) -> Vec<OpNode> {
    let mut ancestors = vec![node];
    let mut visited: HashSet<_> = ancestors.iter().copied().collect();
    let mut stack = vec![node];

    while let Some(current) = stack.pop() {
        for pred in opgraph.inbound_edges(current) {
            if fusion.contains_node(pred) && visited.insert(pred) {
                ancestors.push(pred);
                stack.push(pred);
            }
        }
    }

    ancestors
}

pub fn minimize_cost_function_with_subdivision(
    opgraph: &Arc<OpGraph>,
    cost_function: impl Fn(&FusedGraph) -> f64,
) -> (FusedGraph, f64) {
    let subdivisions = subdivide(opgraph);

    let mut fused_subdivisions = Vec::new();
    for subdivision in &subdivisions {
        let subgraph = opgraph.subgraph(subdivision.iter().map(|(node, _)| *node));
        fused_subdivisions.push(minimize_cost_function(&Arc::new(subgraph), &cost_function).0);
    }

    let final_graph =
        FusedGraph::merge_from_subdivisions(&subdivisions, &fused_subdivisions, opgraph);

    let cost = cost_function(&final_graph);

    (final_graph, cost)
}

pub fn minimize_cost_function(
    opgraph: &Arc<OpGraph>,
    cost_function: impl Fn(&FusedGraph) -> f64,
) -> (FusedGraph, f64) {
    let mut best: Option<(FusedGraph, f64)> = None;

    #[derive(Default)]
    struct State {
        visited_hashes_current: HashSet<u64>,
        best_updated: bool,
    }

    fn visit(
        current: &FusedGraph,
        best: &mut Option<(FusedGraph, f64)>,
        state: &mut State,
        cost_function: &impl Fn(&FusedGraph) -> f64,
        depth: u64,
    ) {
        let cost = cost_function(current);
        match &best {
            None => *best = Some((current.clone(), cost)),
            Some((_, best_score)) => {
                if cost < *best_score {
                    *best = Some((current.clone(), cost));
                    state.best_updated = true;
                }
            }
        }

        if depth == 7 {
            return;
        }
        for action in current.enumerate_possible_actions() {
            let mut modified_graph = current.clone();
            modified_graph.apply_action(&action);
            let hash = foldhash::quality::FixedState::with_seed(10055928159175561594)
                .hash_one(&modified_graph);
            if state.visited_hashes_current.insert(hash) {
                visit(&modified_graph, best, state, cost_function, depth + 1);
            }
        }
    }

    let mut initial = FusedGraph::identity(opgraph);
    initial.produce_maximal_fusions();

    let mut state = State::default();
    loop {
        visit(&initial, &mut best, &mut state, &cost_function, 0);
        initial = best
            .clone()
            .map(|(mut f, _)| {
                f.clean();
                f
            })
            .unwrap();

        if !state.best_updated {
            // let mut best = best.as_ref().unwrap().0.clone();
            // dbg!(cost_function(&best));
            // best.assert_invariants();
            // best.apply_action(&Action::CoreFuseIntoEpilogue {
            //     node: OpNode(108),
            //     fusion: FusionId(66),
            //     dependency_providers: btreemap! { OpNode(3) => FusionId(3 )},
            // });
            // dbg!(&best.node_fusions[&OpNode(108)]);
            // best.enumerate_possible_actions2();
            // best.print();
            // panic!();
            break;
        }
        state.best_updated = false;

        state.visited_hashes_current.clear();
    }
    best.map(|(mut f, score)| {
        f.clean();
        (f, score)
    })
    .unwrap()
}
