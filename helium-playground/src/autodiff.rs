use crate::opgraph::{
    BinaryPointwiseOp, Op, OpGraph, OpNode, ReductionDepth, ReductionOp, UnaryPointwiseOp,
};
use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use std::{collections::hash_map::Entry, sync::Arc};

pub fn autodiff(op_graph: &Arc<OpGraph>, loss: OpNode) -> AutodiffOutput {
    Autodiffer::new(op_graph).backprop(loss)
}

pub struct Autodiffer {
    op_graph: Arc<OpGraph>,
}

impl Autodiffer {
    pub fn new(op_graph: &Arc<OpGraph>) -> Self {
        Self {
            op_graph: op_graph.clone(),
        }
    }

    pub fn backprop(self, loss: OpNode) -> AutodiffOutput {
        let mut flows: HashMap<OpNode, OpNode> = HashMap::new();
        let mut visited = HashSet::new();
        let mut new_graph = (*self.op_graph).clone();

        struct StackEntry {
            node: OpNode,
        }

        let mut stack = vec![StackEntry { node: loss }];
        while let Some(StackEntry { node }) = stack.pop() {
            visited.insert(node);
            let flow = flows.get(&node).copied().unwrap_or_else(|| {
                new_graph.insert(Op::Constant {
                    shape: self.op_graph.shape(node).clone(),
                    value: 1.0,
                })
            });
            let mut computed_gradients = Vec::new();

            match self.op_graph.get(node) {
                Op::Matmul { a, b } => {
                    let bt = new_graph.insert(Op::Transpose { x: *b });
                    let gradient_a = new_graph.insert(Op::Matmul { a: flow, b: bt });
                    let at = new_graph.insert(Op::Transpose { x: *a });
                    let gradient_b = new_graph.insert(Op::Matmul { a: at, b: flow });

                    computed_gradients.push((*a, gradient_a));
                    computed_gradients.push((*b, gradient_b));
                }
                Op::UnaryPointwise { x, op } => {
                    let gradient_x = match *op {
                        UnaryPointwiseOp::Neg => new_graph.insert(Op::UnaryPointwise {
                            x: flow,
                            op: UnaryPointwiseOp::Neg,
                        }),
                        UnaryPointwiseOp::Relu => new_graph.insert(Op::BinaryPointwise {
                            lhs: flow,
                            rhs: *x,
                            op: BinaryPointwiseOp::Drelu,
                        }),
                        UnaryPointwiseOp::Gelu => {
                            todo!()
                        }
                        UnaryPointwiseOp::Tanh => {
                            let squared = new_graph.insert(Op::UnaryPointwise {
                                x: node,
                                op: UnaryPointwiseOp::PowConstant(2.0),
                            });
                            let neg = new_graph.insert(Op::UnaryPointwise {
                                x: squared,
                                op: UnaryPointwiseOp::Neg,
                            });
                            let plus_one = new_graph.insert(Op::UnaryPointwise {
                                x: neg,
                                op: UnaryPointwiseOp::AddConstant(1.0),
                            });
                            new_graph.insert(Op::BinaryPointwise {
                                lhs: flow,
                                rhs: plus_one,
                                op: BinaryPointwiseOp::Mul,
                            })
                        }
                        UnaryPointwiseOp::Exp => new_graph.insert(Op::BinaryPointwise {
                            lhs: flow,
                            rhs: node,
                            op: BinaryPointwiseOp::Mul,
                        }),
                        UnaryPointwiseOp::Ln => todo!(),
                        UnaryPointwiseOp::Sqrt => new_graph.insert(Op::BinaryPointwise {
                            lhs: flow,
                            rhs: node,
                            op: BinaryPointwiseOp::Div,
                        }),
                        UnaryPointwiseOp::AddConstant(_) => flow,
                        UnaryPointwiseOp::MulConstant(c) => new_graph.insert(Op::UnaryPointwise {
                            x: flow,
                            op: UnaryPointwiseOp::MulConstant(c),
                        }),
                        UnaryPointwiseOp::PowConstant(c) => {
                            let c1 = new_graph.insert(Op::Constant {
                                shape: self.op_graph.shape(*x).clone(),
                                value: c,
                            });
                            let c2 = new_graph.insert(Op::Constant {
                                shape: self.op_graph.shape(*x).clone(),
                                value: c - 1.0,
                            });
                            let imm1 = new_graph.insert(Op::BinaryPointwise {
                                lhs: *x,
                                rhs: c2,
                                op: BinaryPointwiseOp::Pow,
                            });
                            let imm2 = new_graph.insert(Op::BinaryPointwise {
                                lhs: imm1,
                                rhs: c1,
                                op: BinaryPointwiseOp::Mul,
                            });
                            new_graph.insert(Op::BinaryPointwise {
                                lhs: imm2,
                                rhs: flow,
                                op: BinaryPointwiseOp::Mul,
                            })
                        }
                    };
                    computed_gradients.push((*x, gradient_x));
                }
                Op::Reduction { x, depth, .. } => {
                    let gradient_x = match *depth {
                        ReductionDepth::Rows => new_graph.insert(Op::Broadcast {
                            x: flow,
                            axis: 2,
                            amount: self.op_graph.shape(*x).dim(2),
                        }),
                        ReductionDepth::Columns => new_graph.insert(Op::Broadcast {
                            x: flow,
                            axis: 1,
                            amount: self.op_graph.shape(*x).dim(1),
                        }),
                        ReductionDepth::Item => {
                            let imm1 = new_graph.insert(Op::Broadcast {
                                x: flow,
                                axis: 1,
                                amount: self.op_graph.shape(*x).dim(1),
                            });
                            new_graph.insert(Op::Broadcast {
                                x: imm1,
                                axis: 2,
                                amount: self.op_graph.shape(*x).dim(2),
                            })
                        }
                        ReductionDepth::Batch => {
                            let imm1 = new_graph.insert(Op::Broadcast {
                                x: flow,
                                axis: 1,
                                amount: self.op_graph.shape(*x).dim(1),
                            });
                            let imm2 = new_graph.insert(Op::Broadcast {
                                x: imm1,
                                axis: 2,
                                amount: self.op_graph.shape(*x).dim(2),
                            });
                            new_graph.insert(Op::Broadcast {
                                x: imm2,
                                axis: 0,
                                amount: self.op_graph.shape(*x).dim(0),
                            })
                        }
                    };
                    computed_gradients.push((*x, gradient_x));
                }
                Op::BinaryPointwise { lhs, rhs, op } => {
                    let gradient_lhs = match *op {
                        BinaryPointwiseOp::Add => flow,
                        BinaryPointwiseOp::Sub => flow,
                        BinaryPointwiseOp::Mul => new_graph.insert(Op::BinaryPointwise {
                            lhs: flow,
                            rhs: *rhs,
                            op: BinaryPointwiseOp::Mul,
                        }),
                        BinaryPointwiseOp::Div => new_graph.insert(Op::BinaryPointwise {
                            lhs: flow,
                            rhs: *rhs,
                            op: BinaryPointwiseOp::Div,
                        }),
                        BinaryPointwiseOp::Pow => {
                            todo!()
                        }
                        BinaryPointwiseOp::Drelu => todo!(),
                    };
                    let gradient_rhs = match *op {
                        BinaryPointwiseOp::Add => flow,
                        BinaryPointwiseOp::Sub => new_graph.insert(Op::UnaryPointwise {
                            x: flow,
                            op: UnaryPointwiseOp::Neg,
                        }),
                        BinaryPointwiseOp::Mul => new_graph.insert(Op::BinaryPointwise {
                            lhs: *lhs,
                            rhs: flow,
                            op: BinaryPointwiseOp::Mul,
                        }),
                        BinaryPointwiseOp::Div => new_graph.insert(Op::BinaryPointwise {
                            lhs: flow,
                            rhs: *lhs,
                            op: BinaryPointwiseOp::Div,
                        }),
                        BinaryPointwiseOp::Pow => todo!(),
                        BinaryPointwiseOp::Drelu => todo!(),
                    };

                    computed_gradients.push((*lhs, gradient_lhs));
                    computed_gradients.push((*rhs, gradient_rhs));
                }
                Op::Broadcast { x, axis, .. } => {
                    let gradient_x = match *axis {
                        2 => new_graph.insert(Op::Reduction {
                            x: flow,
                            op: ReductionOp::Sum,
                            depth: ReductionDepth::Rows,
                        }),
                        1 => new_graph.insert(Op::Reduction {
                            x: flow,
                            op: ReductionOp::Sum,
                            depth: ReductionDepth::Columns,
                        }),
                        _ => todo!(),
                    };
                    computed_gradients.push((*x, gradient_x));
                }
                Op::Transpose { x } => {
                    computed_gradients.push((*x, new_graph.insert(Op::Transpose { x: flow })));
                }
                Op::Consumer { .. } | Op::Producer { .. } | Op::Constant { .. } => continue,
            };

            for (node, gradient) in computed_gradients {
                match flows.entry(node) {
                    Entry::Occupied(mut entry) => {
                        // total gradient
                        let existing_flow = *entry.get();
                        *entry.get_mut() = new_graph.insert(Op::BinaryPointwise {
                            lhs: existing_flow,
                            rhs: gradient,
                            op: BinaryPointwiseOp::Add,
                        });
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(gradient);
                    }
                }

                // check if all of this node's predecessors have been visited => we can
                // push it to stack
                if self
                    .op_graph
                    .outbound_edges(node)
                    .all(|n| visited.contains(&n))
                    && visited.insert(node)
                {
                    stack.push(StackEntry { node });
                }
            }
        }

        AutodiffOutput {
            new_graph: Arc::new(new_graph),
            param_gradients: flows,
        }
    }
}

pub struct AutodiffOutput {
    new_graph: Arc<OpGraph>,
    param_gradients: HashMap<OpNode, OpNode>,
}

impl AutodiffOutput {
    pub fn new_graph(&self) -> &Arc<OpGraph> {
        &self.new_graph
    }

    pub fn param_gradient(&self, node: OpNode) -> OpNode {
        self.param_gradients[&node]
    }
}
