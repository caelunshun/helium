# helium

Deep learning on top of CUDA, cuDNN, and CUTLASS.

Some notable features include:

- Lazy evaluation. Rather than eagerly applying tensor operations, the `Tensor` API internally
builds an operation graph, which is compiled and evaluated only when requested. This enables
fusion optimizations. The graph tracking is in `helium/src/raw_tensor.rs` and the fusion
compilation is in `helium/src/backend/plan_generation.rs`.
- On-the-fly generation of fused kernels, including arbitrary tiled shape-permute
kernels (workaround for [this limitation of cuDNN](https://github.com/NVIDIA/cudnn-frontend/issues/119)).
The shape-permute kernels have little to no shared memory bank conflicts by using a swizzled layout.
The kernel generation is in `helium/src/cuda/instr/permute_dims` and `helium/src/cuda/instr/pointwise`.
- For GEMMs, there is a CUTLASS-based fused kernel generator in the `helium-kernel-generator`.
It implements various heuristics (in `helium-kernel-generator/src/generators/matmul.rs`)
to minimize shared memory bank conflicts and maximize load/store vectorization. Specifically,
it uses the CuTe layout algebra to evaluate many possible thread partitioning patterns and
selects the best one under a cost function.
- Where possible, the graph execution engine will execute multiple kernels in parallel using the CUDA
streams API. This is beneficial when the grid size of a single kernel is too small to fill the GPU.
- Data movement between the host and the GPU is carefully pipelined, enabling close to 100%
SM utilization for realistic training workloads (ResNet).
- Various example training runs are in `helium/examples`. A non-toy example, ResNet training on ImageNet
classification, is available in `resnet-train`.

The `helium-playground` directory contains an experimental new graph compiler that supports
the mainloop fusion and recomputation patterns from the paper. The graph fusion algorithm is
in `helium-playground/src/fused_graph.rs`. For each fused kernel, it generates an IR like the following:

```

FusionId(26): 
Core = Matmul { a: OpNode(1), b: OpNode(2) } // indicates the "anchor" node is a GEMM
    // set of operators fused into the mainloop (i.e. during tile loading pre-MMA)
	Mainloop:
		OpNode(23) = UnaryPointwise { x: OpNode(22), op: Relu }
		OpNode(25) = BinaryPointwise { lhs: OpNode(23), rhs: OpNode(9), op: Mul }
    // set of operators fused into the epilogue
	Epilogue:
		OpNode(14) = BinaryPointwise { lhs: OpNode(12), rhs: OpNode(13), op: Add }
		OpNode(15) = Broadcast { x: OpNode(6), axis: 1, amount: 4096 }
		OpNode(16) = BinaryPointwise { lhs: OpNode(14), rhs: OpNode(15), op: Add }
		OpNode(17) = UnaryPointwise { x: OpNode(16), op: Relu }
		OpNode(27) = BinaryPointwise { lhs: OpNode(24), rhs: OpNode(26), op: Add }
		OpNode(28) = Broadcast { x: OpNode(8), axis: 1, amount: 4096 }
		OpNode(29) = BinaryPointwise { lhs: OpNode(27), rhs: OpNode(28), op: Add }
		OpNode(30) = UnaryPointwise { x: OpNode(29), op: Tanh }
		OpNode(31) = UnaryPointwise { x: OpNode(17), op: Neg }
		OpNode(32) = UnaryPointwise { x: OpNode(31), op: AddConstant(1.0) }
		OpNode(33) = BinaryPointwise { lhs: OpNode(32), rhs: OpNode(9), op: Mul }
		OpNode(34) = BinaryPointwise { lhs: OpNode(17), rhs: OpNode(30), op: Mul }
		OpNode(35) = BinaryPointwise { lhs: OpNode(33), rhs: OpNode(34), op: Add }
		OpNode(36) = BinaryPointwise { lhs: OpNode(35), rhs: OpNode(11), op: Sub }
		OpNode(37) = UnaryPointwise { x: OpNode(36), op: PowConstant(2.0) }
		OpNode(38) = UnaryPointwise { x: OpNode(37), op: MulConstant(0.00024414063) }
		OpNode(39) = Reduction { x: OpNode(38), op: Sum, depth: Batch }
		OpNode(41) = Constant { shape: Shape([1, 1, 1]), value: 1.0 }
		OpNode(42) = Broadcast { x: OpNode(41), axis: 1, amount: 4096 }
		OpNode(43) = Broadcast { x: OpNode(42), axis: 2, amount: 4096 }
		OpNode(44) = Broadcast { x: OpNode(43), axis: 0, amount: 1 }
		OpNode(45) = UnaryPointwise { x: OpNode(44), op: MulConstant(0.00024414063) }
		OpNode(46) = Constant { shape: Shape([1, 4096, 4096]), value: 2.0 }
		OpNode(47) = Constant { shape: Shape([1, 4096, 4096]), value: 1.0 }
		OpNode(48) = BinaryPointwise { lhs: OpNode(36), rhs: OpNode(47), op: Pow }
		OpNode(49) = BinaryPointwise { lhs: OpNode(48), rhs: OpNode(46), op: Mul }
		OpNode(50) = BinaryPointwise { lhs: OpNode(49), rhs: OpNode(45), op: Mul }
		OpNode(52) = BinaryPointwise { lhs: OpNode(50), rhs: OpNode(30), op: Mul }
		OpNode(53) = BinaryPointwise { lhs: OpNode(17), rhs: OpNode(50), op: Mul }
		OpNode(54) = UnaryPointwise { x: OpNode(30), op: PowConstant(2.0) }
		OpNode(55) = UnaryPointwise { x: OpNode(54), op: Neg }
		OpNode(56) = UnaryPointwise { x: OpNode(55), op: AddConstant(1.0) }
		OpNode(57) = BinaryPointwise { lhs: OpNode(53), rhs: OpNode(56), op: Mul }
		OpNode(58) = Reduction { x: OpNode(57), op: Sum, depth: Columns }
		OpNode(81) = BinaryPointwise { lhs: OpNode(50), rhs: OpNode(9), op: Mul }
		OpNode(84) = UnaryPointwise { x: OpNode(81), op: Neg }
		OpNode(85) = BinaryPointwise { lhs: OpNode(52), rhs: OpNode(84), op: Add }
		OpNode(86) = BinaryPointwise { lhs: OpNode(85), rhs: OpNode(16), op: Drelu }
		OpNode(87) = Reduction { x: OpNode(86), op: Sum, depth: Columns }
```

which it then converts to CUTLASS-templated kernels using `helium-playground/src/kernel_generator.rs`.
