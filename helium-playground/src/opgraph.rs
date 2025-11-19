use crate::fused_graph::maybe_vec_remove_item;
use foldhash::{HashMap, HashSet};
use itertools::Itertools;
use std::{
    collections::{BTreeMap, BTreeSet},
    hash::Hash,
    sync::atomic::AtomicU64,
};

#[derive(Debug, Clone, Default)]
pub struct OpGraph {
    nodes: BTreeMap<OpNode, Op>,
    shapes: HashMap<OpNode, Shape>,
    node_inbound: HashMap<OpNode, Vec<OpNode>>,
    node_outbound: HashMap<OpNode, Vec<OpNode>>,
    roots: Vec<OpNode>,
    core_nodes: Vec<OpNode>,
}

impl PartialEq for OpGraph {
    fn eq(&self, other: &Self) -> bool {
        self.nodes.iter().eq(other.nodes.iter())
    }
}

impl Eq for OpGraph {}

impl OpGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, node: Op) -> OpNode {
        let inputs = node.inputs();
        let is_core = node.is_anchor();
        let shape = node.shape_given_inputs(|input| &self.shapes[&input]);
        let node_id = OpNode::new();
        self.nodes.insert(node_id, node);

        for &input in &inputs {
            self.node_outbound.get_mut(&input).unwrap().push(node_id);
        }

        if inputs.is_empty() {
            self.roots.push(node_id);
        }

        if is_core {
            self.core_nodes.push(node_id);
        }

        self.node_inbound.insert(node_id, inputs);
        self.node_outbound.insert(node_id, Vec::new());
        self.shapes.insert(node_id, shape);

        node_id
    }

    pub fn get(&self, node: OpNode) -> &Op {
        &self.nodes[&node]
    }

    pub fn shape(&self, node: OpNode) -> &Shape {
        &self.shapes[&node]
    }

    pub fn inbound_edges(&self, node: OpNode) -> impl Iterator<Item = OpNode> {
        self.node_inbound[&node].iter().copied()
    }

    pub fn outbound_edges(&self, node: OpNode) -> impl Iterator<Item = OpNode> {
        self.node_outbound[&node].iter().copied()
    }

    pub fn roots(&self) -> impl Iterator<Item = OpNode> {
        self.roots.iter().copied()
    }

    pub fn core_nodes(&self) -> impl Iterator<Item = OpNode> {
        self.core_nodes.iter().copied()
    }

    pub fn nodes(&self) -> impl Iterator<Item = OpNode> {
        self.nodes.keys().copied()
    }

    pub fn aux_nodes(&self) -> impl Iterator<Item = OpNode> {
        self.nodes().filter(|n| !self.core_nodes.contains(n))
    }

    pub fn leaves(&self) -> impl Iterator<Item = OpNode> {
        self.nodes()
            .filter(|n| self.outbound_edges(*n).count() == 0)
    }

    pub fn subgraph(&self, nodes: impl IntoIterator<Item = OpNode>) -> OpGraph {
        let nodes: BTreeSet<OpNode> = nodes.into_iter().collect();

        let roots = nodes
            .iter()
            .copied()
            .filter(|n| self.inbound_edges(*n).all(|n2| !nodes.contains(&n2)))
            .collect::<Vec<_>>();

        let mut new_graph = OpGraph {
            nodes: nodes
                .iter()
                .copied()
                .map(|n| (n, self.nodes[&n].clone()))
                .collect(),
            shapes: nodes
                .iter()
                .copied()
                .map(|n| (n, self.shapes[&n].clone()))
                .collect(),
            node_inbound: nodes
                .iter()
                .copied()
                .map(|n| (n, Default::default()))
                .collect(),
            node_outbound: nodes
                .iter()
                .copied()
                .map(|n| (n, Default::default()))
                .collect(),
            roots: roots.clone(),
            core_nodes: nodes
                .iter()
                .copied()
                .filter(|n| self.nodes[n].is_anchor())
                .collect(),
        };

        // add producers for inputs
        for &node in &nodes {
            for dep in self.inbound_edges(node) {
                if nodes.contains(&dep) {
                    continue;
                }

                maybe_vec_remove_item(&mut new_graph.roots, &node);
                new_graph.nodes.insert(
                    dep,
                    Op::Producer {
                        shape: self.shape(dep).clone(),
                    },
                );
                new_graph.shapes.insert(dep, self.shape(dep).clone());
                new_graph.roots.push(dep);
            }
        }

        // add consumers for leaves
        for &node in &nodes {
            for user in self.outbound_edges(node) {
                if nodes.contains(&user) {
                    continue;
                }

                new_graph.nodes.insert(user, Op::Consumer { x: node });
                new_graph.shapes.insert(user, new_graph.shape(node).clone());

                break;
            }
        }

        // populate edges
        for (&id, node) in &new_graph.nodes {
            new_graph.node_inbound.entry(id).or_default();
            new_graph.node_outbound.entry(id).or_default();
            let inbound = node.inputs();
            for inbound in inbound {
                assert!(new_graph.nodes.contains_key(&inbound));
                new_graph.node_inbound.entry(id).or_default().push(inbound);
                new_graph.node_outbound.entry(inbound).or_default().push(id);
            }
        }

        new_graph
    }

    pub fn remove_dead_code(&mut self) {
        let mut stack: Vec<_> = self
            .nodes()
            .filter(|n| matches!(self.get(*n), Op::Consumer { .. }))
            .collect();
        let mut visited: HashSet<_> = stack.iter().copied().collect();

        while let Some(current) = stack.pop() {
            for inbound in self.inbound_edges(current) {
                if visited.insert(inbound) {
                    stack.push(inbound);
                }
            }
        }

        for node in self.nodes().collect_vec() {
            if !visited.contains(&node) {
                self.nodes.remove(&node);
                self.shapes.remove(&node);
                self.node_inbound.remove(&node);
                self.node_outbound.remove(&node);

                self.node_inbound
                    .values_mut()
                    .for_each(|v| maybe_vec_remove_item(v, &node));
                self.node_outbound
                    .values_mut()
                    .for_each(|v| maybe_vec_remove_item(v, &node));
            }
        }
    }

    pub fn topo_sort(&self) -> Vec<OpNode> {
        let mut stack = self.roots.clone();
        let mut visited: HashSet<_> = stack.iter().copied().collect();
        let mut result = Vec::new();

        while let Some(current) = stack.pop() {
            result.push(current);
            for next in self.outbound_edges(current) {
                if self.inbound_edges(next).all(|n| result.contains(&n)) && visited.insert(next) {
                    stack.push(next);
                }
            }
        }

        result
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OpNode(pub u64);

impl OpNode {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape([u32; 3]);

impl Shape {
    pub fn new(shape: impl Into<[u32; 3]>) -> Self {
        Self(shape.into())
    }

    pub fn dim(&self, n: usize) -> u32 {
        self.0[n]
    }

    pub fn dims(&self) -> [u32; 3] {
        self.0
    }

    pub fn element_count(&self) -> u64 {
        self.0.iter().copied().map(u64::from).product()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Producer {
        shape: Shape,
    },
    Constant {
        shape: Shape,
        value: f32,
    },
    Matmul {
        a: OpNode,
        b: OpNode,
    },
    UnaryPointwise {
        x: OpNode,
        op: UnaryPointwiseOp,
    },
    Reduction {
        x: OpNode,
        op: ReductionOp,
        depth: ReductionDepth,
    },
    BinaryPointwise {
        lhs: OpNode,
        rhs: OpNode,
        op: BinaryPointwiseOp,
    },
    Broadcast {
        x: OpNode,
        axis: u32,
        amount: u32,
    },
    Transpose {
        x: OpNode,
    },
    Consumer {
        x: OpNode,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    Sum,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReductionDepth {
    Rows,
    Columns,
    Item,
    Batch,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum UnaryPointwiseOp {
    Neg,
    Relu,
    Gelu,
    Tanh,
    Exp,
    Ln,
    Sqrt,
    AddConstant(f32),
    MulConstant(f32),
    PowConstant(f32),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BinaryPointwiseOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Drelu,
}

impl Op {
    pub fn replace_input(&mut self, prev: OpNode, new: OpNode) {
        match self {
            Op::Producer { .. } | Op::Constant { .. } => {}
            Op::Matmul { a, b } => {
                if *a == prev {
                    *a = new;
                }
                if *b == prev {
                    *b = new;
                }
            }
            Op::UnaryPointwise { x, .. } => {
                if *x == prev {
                    *x = new;
                }
            }
            Op::Reduction { x, .. } => {
                if *x == prev {
                    *x = new;
                }
            }
            Op::BinaryPointwise { lhs, rhs, .. } => {
                if *lhs == prev {
                    *lhs = new;
                }
                if *rhs == prev {
                    *rhs = new;
                }
            }
            Op::Broadcast { x, .. } => {
                if *x == prev {
                    *x = new;
                }
            }
            Op::Consumer { x, .. } => {
                if *x == prev {
                    *x = new;
                }
            }
            Op::Transpose { x } => {
                if *x == prev {
                    *x = new;
                }
            }
        }
    }

    pub fn shape_given_inputs<'a>(&self, input_shape: impl Fn(OpNode) -> &'a Shape) -> Shape {
        match self {
            Op::Producer { shape } | Op::Constant { shape, .. } => shape.clone(),
            Op::Matmul { a, b } => Shape::new([
                input_shape(*a).dim(0),
                input_shape(*a).dim(1),
                input_shape(*b).dim(2),
            ]),
            Op::UnaryPointwise { x, .. } => input_shape(*x).clone(),
            Op::Reduction { x, depth, .. } => {
                let mut input = input_shape(*x).0;
                match *depth {
                    ReductionDepth::Rows => input[2] = 1,
                    ReductionDepth::Columns => input[1] = 1,
                    ReductionDepth::Item => {
                        input[2] = 1;
                        input[1] = 1;
                    }
                    ReductionDepth::Batch => {
                        input[2] = 1;
                        input[1] = 1;
                        input[0] = 1;
                    }
                }
                Shape::new(input)
            }
            Op::BinaryPointwise { lhs, .. } => input_shape(*lhs).clone(),
            Op::Consumer { x } => input_shape(*x).clone(),
            Op::Broadcast { x, axis, amount } => {
                let mut dims = input_shape(*x).0;
                dims[*axis as usize] = *amount;
                Shape::new(dims)
            }
            Op::Transpose { x } => {
                let mut shape = input_shape(*x).0;
                shape.swap(1, 2);
                Shape::new(shape)
            }
        }
    }

    pub fn is_anchor(&self) -> bool {
        matches!(self, Op::Matmul { .. })
    }

    pub fn is_pointwise(&self) -> bool {
        matches!(
            self,
            Op::UnaryPointwise { .. } | Op::BinaryPointwise { .. } | Op::Constant { .. }
        )
    }

    pub fn is_reduction(&self) -> bool {
        matches!(self, Op::Reduction { .. })
    }

    pub fn is_transpose(&self) -> bool {
        matches!(self, Op::Transpose { .. })
    }

    pub fn is_broadcast(&self) -> bool {
        matches!(self, Op::Broadcast { .. })
    }

    pub fn is_epilogue_fusible(&self) -> bool {
        self.is_pointwise() || self.is_reduction() || self.is_transpose() || self.is_broadcast()
    }

    pub fn is_fusion_terminator(&self) -> bool {
        self.is_reduction() || self.is_transpose()
    }

    pub fn is_mainloop_fusible(&self) -> bool {
        self.is_pointwise() || self.is_transpose() || self.is_broadcast()
    }

    pub fn is_aux_fusible(&self) -> bool {
        self.is_pointwise() || self.is_reduction() || self.is_transpose()
    }

    pub fn is_input_output(&self) -> bool {
        matches!(self, Op::Producer { .. } | Op::Consumer { .. })
    }

    pub fn inputs(&self) -> Vec<OpNode> {
        match self {
            Op::Producer { .. } => vec![],
            Op::Matmul { a, b } => vec![*a, *b],
            Op::UnaryPointwise { x, .. } => vec![*x],
            Op::Reduction { x, .. } => vec![*x],
            Op::BinaryPointwise { lhs, rhs, .. } => vec![*lhs, *rhs],
            Op::Broadcast { x, .. } => vec![*x],
            Op::Consumer { x } => vec![*x],
            Op::Transpose { x } => vec![*x],
            Op::Constant { .. } => vec![],
        }
    }
}
