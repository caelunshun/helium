use crate::opgraph::{BinaryPointwiseOp, Op, OpGraph, ReductionDepth, UnaryPointwiseOp};
use indoc::formatdoc;
use std::{fs, path::Path};

pub fn synthesize_torch_graph(opgraph: &OpGraph, file: impl AsRef<Path>) {
    let mut tensor_inits: Vec<String> = Vec::new();
    let mut params: Vec<String> = Vec::new();
    let mut forward: Vec<String> = Vec::new();
    let mut returns: Vec<String> = Vec::new();

    for node in opgraph.roots() {
        if let Op::Producer { shape } = opgraph.get(node) {
            tensor_inits.push(format!(
                "    tensor{} = torch.rand(({}, {}, {}), device = \"cuda:0\", dtype = torch.bfloat16)",
                node.0,
                shape.dim(0),
                shape.dim(1),
                shape.dim(2)
            ));
            params.push(format!("tensor{}", node.0));
        }
    }

    for node in opgraph.leaves() {
        if let Op::Consumer { x } = opgraph.get(node) {
            returns.push(format!("tensor{}", x.0));
        }
    }

    for node in opgraph.topo_sort() {
        let expr = match opgraph.get(node) {
            Op::Constant { shape, value } => {
                format!(
                    "torch.full(({}, {}, {}), {value}, dtype=torch.bfloat16, device=\"cuda:0\")",
                    shape.dim(0),
                    shape.dim(1),
                    shape.dim(2)
                )
            }
            Op::Matmul { a, b } => {
                format!("torch.matmul(tensor{}, tensor{})", a.0, b.0)
            }
            Op::UnaryPointwise { x, op } => {
                let x = format!("tensor{}", x.0);
                match op {
                    UnaryPointwiseOp::Neg => format!("torch.neg({x})"),
                    UnaryPointwiseOp::Relu => format!("F.relu({x})"),
                    UnaryPointwiseOp::Gelu => format!("F.gelu({x})"),
                    UnaryPointwiseOp::Tanh => format!("torch.tanh({x})"),
                    UnaryPointwiseOp::Exp => format!("torch.exp({x})"),
                    UnaryPointwiseOp::Ln => format!("torch.log({x})"),
                    UnaryPointwiseOp::Sqrt => format!("torch.sqrt({x})"),
                    UnaryPointwiseOp::AddConstant(c) => format!("{x} + {c}"),
                    UnaryPointwiseOp::MulConstant(c) => format!("{x} * {c}"),
                    UnaryPointwiseOp::PowConstant(c) => format!("{x} ** {c}"),
                }
            }
            Op::Reduction { x, depth, .. } => {
                let dim = match depth {
                    ReductionDepth::Rows => "2",
                    ReductionDepth::Columns => "1",
                    ReductionDepth::Item => "(1, 2)",
                    ReductionDepth::Batch => "(0, 1, 2)",
                };
                format!("tensor{}.sum(dim={dim}, keepdim=True)", x.0)
            }
            Op::BinaryPointwise { lhs, rhs, op } => {
                let lhs = format!("tensor{}", lhs.0);
                let rhs = format!("tensor{}", rhs.0);
                match op {
                    BinaryPointwiseOp::Add => format!("{lhs} + {rhs}"),
                    BinaryPointwiseOp::Sub => format!("{lhs} - {rhs}"),
                    BinaryPointwiseOp::Mul => format!("{lhs} * {rhs}"),
                    BinaryPointwiseOp::Div => format!("{lhs} / {rhs}"),
                    BinaryPointwiseOp::Pow => format!("{lhs} ** {rhs}"),
                    BinaryPointwiseOp::Drelu => {
                        format!("torch.where({rhs} > 0, {lhs}, torch.tensor(0.0))")
                    }
                }
            }
            Op::Broadcast { x, .. } => {
                // implicit broadcasts in torch
                format!("tensor{}", x.0)
            }
            Op::Transpose { x } => format!("torch.transpose(tensor{}, 1, 2)", x.0),
            Op::Producer { .. } | Op::Consumer { .. } => continue,
        };

        forward.push(format!("        tensor{} = {expr}\n", node.0));
    }

    let tensor_inits = tensor_inits.join("\n");
    let params = params.join(", ");
    let forward = forward.join("\n");
    let returns = returns.join(", ");

    let source = formatdoc! {"
    import torch
    import torch.nn.functional as F
    import time
    import nvtx

    def run_graph({params}):
        with torch.no_grad():
    {forward}
            return ({returns})

    with torch.no_grad():
        run_graph_compiled = torch.compile(run_graph)
    {tensor_inits}

        warmup_runs = 25
        sampling_runs = 100

        for i in range(warmup_runs):
            run_graph_compiled({params})
        torch.cuda.synchronize()

        start = time.perf_counter()
        #for i in range(sampling_runs):
        #    run_graph_compiled({params})
        with nvtx.annotate(\"step\", color=\"red\"):
            run_graph_compiled({params})
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = (end - start) / sampling_runs * 1000
        print(f\"{{elapsed}} ms\")

        run_graph_compiled({params})
    "};
    fs::write(file.as_ref(), source).unwrap();
}
