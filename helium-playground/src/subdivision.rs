use crate::opgraph::{OpGraph, OpNode};
use foldhash::{HashSet, HashSetExt};
use itertools::Itertools;
use std::collections::BTreeSet;

/// special handling of mainloop vs. epilogue neighbors
/// (pseudocode makes this cleaner by constructing
/// G_2 that splits anchor nodes in two parts)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NodePortion {
    Mainloop,
    Epilogue,
    Full,
}

type Subdivision = BTreeSet<(OpNode, NodePortion)>;

/// Problem size reduction step. Subdivides a large opgraph
/// into several smaller, independently optimizable subgraphs
/// under the weak connections rule given in the paper.
/// All non-anchor nodes appear in exactly one subgraph.
/// All anchor nodes appear in two subgraphs (one for the mainloop, one
/// for the epilogue).
pub fn subdivide(graph: &OpGraph) -> Vec<Subdivision> {
    let mut subdivisions = Vec::new();

    let mut nonanchor_visited = HashSet::<OpNode>::new();
    let mut anchor_mainloop_visited = HashSet::<OpNode>::new();
    let mut anchor_epilogue_visited = HashSet::<OpNode>::new();

    loop {
        // Select arbitrary node not yet visited
        let node = graph
            .nodes()
            .filter(|n| !graph.get(*n).is_input_output())
            .find(|n| {
                if graph.get(*n).is_anchor() {
                    !anchor_epilogue_visited.contains(n) || !anchor_mainloop_visited.contains(n)
                } else {
                    !nonanchor_visited.contains(n)
                }
            });
        let node = match node {
            Some(node) => node,
            None => break, // done
        };

        let node_portion = if graph.get(node).is_anchor() {
            if !anchor_epilogue_visited.contains(&node) {
                NodePortion::Epilogue
            } else {
                NodePortion::Mainloop
            }
        } else {
            NodePortion::Full // non-anchor
        };

        // ConnectionsMatmulTerminated
        let mut subdivision = BTreeSet::<(OpNode, NodePortion)>::new();
        subdivision.insert((node, node_portion));
        let mut stack = vec![(node, node_portion)];
        while let Some((current_node, current_node_portion)) = stack.pop() {
            let neighbors = match current_node_portion {
                NodePortion::Epilogue => graph.outbound_edges(current_node).collect_vec(),
                NodePortion::Mainloop => graph.inbound_edges(current_node).collect_vec(),
                NodePortion::Full => graph
                    .inbound_edges(current_node)
                    .chain(graph.outbound_edges(current_node))
                    .collect_vec(),
            };

            for neighbor in neighbors {
                if graph.get(neighbor).is_input_output() {
                    continue;
                }

                let neighbor_portion = if graph.get(neighbor).is_anchor() {
                    if graph.outbound_edges(current_node).contains(&neighbor) {
                        NodePortion::Mainloop
                    } else {
                        NodePortion::Epilogue
                    }
                } else {
                    NodePortion::Full
                };

                if subdivision.insert((neighbor, neighbor_portion)) {
                    stack.push((neighbor, neighbor_portion));
                }
            }
        }

        for &(node, node_portion) in &subdivision {
            match node_portion {
                NodePortion::Mainloop => {
                    anchor_mainloop_visited.insert(node);
                }
                NodePortion::Epilogue => {
                    anchor_epilogue_visited.insert(node);
                }
                NodePortion::Full => {
                    anchor_epilogue_visited.insert(node);
                    anchor_mainloop_visited.insert(node);
                    nonanchor_visited.insert(node);
                }
            }
        }

        subdivisions.push(subdivision);
    }

    subdivisions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opgraph::{BinaryPointwiseOp, Op, Shape, UnaryPointwiseOp};
    use maplit::btreeset;

    #[test]
    fn example_subdivision() {
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

        let mm2 = graph.insert(Op::Matmul { a: mm1, b: input });
        let relu2 = graph.insert(Op::UnaryPointwise {
            x: mm2,
            op: UnaryPointwiseOp::Relu,
        });
        let add2 = graph.insert(Op::BinaryPointwise {
            lhs: relu2,
            rhs: input,
            op: BinaryPointwiseOp::Add,
        });

        let mut subdivisions = subdivide(&graph);
        subdivisions.sort();

        let mut expected_subdivisions = vec![
            btreeset![(mm1, NodePortion::Mainloop)],
            btreeset![
                (mm1, NodePortion::Epilogue),
                (relu1, NodePortion::Full),
                (add1, NodePortion::Full),
                (mm2, NodePortion::Mainloop)
            ],
            btreeset![
                (mm2, NodePortion::Epilogue),
                (relu2, NodePortion::Full),
                (add2, NodePortion::Full)
            ],
        ];
        expected_subdivisions.sort();

        assert_eq!(subdivisions, expected_subdivisions);
    }
}
