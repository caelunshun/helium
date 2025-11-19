use crate::{
    fused_graph::{FusedGraph, FusionId},
    kernel_generator::KernelGenerator,
    opgraph::{Op, OpGraph, OpNode},
};
use foldhash::{HashMap, HashMapExt};
use indoc::formatdoc;
use itertools::Itertools;
use std::{fs, path::PathBuf, sync::Arc};

#[derive(Default)]
struct MakefileBuilder {
    source_files: Vec<String>,
}

impl MakefileBuilder {
    pub fn add_source_file(&mut self, name: &str) {
        self.source_files.push(name.to_string());
    }

    pub fn build_makefile(&self) -> String {
        let object_file_list = self
            .source_files
            .iter()
            .map(|s| format!("{s}.o"))
            .collect_vec()
            .join(" ");

        let mut object_file_builders = Vec::new();

        for source_name in &self.source_files {
            object_file_builders.push(formatdoc! {"
            {source_name}.o: {source_name}.cu
            \tnvcc $(CCFLAGS) -o {source_name}.o -c {source_name}.cu
            "});
        }

        let object_file_builders = object_file_builders.join("\n\n");

        formatdoc! {"
        CCFLAGS=-O3 -std=c++20 -lineinfo -I../../cutlass/include -I../../cutlass/tools/util/include --expt-relaxed-constexpr -arch=sm_90a

        {object_file_builders}

        main: main.cu {object_file_list}
        \tnvcc $(CCFLAGS) -o main.o -c main.cu
        \tnvcc $(CCFLAGS) -o main main.o {object_file_list}

        run: main
        \t./main

        profile: main
        \tncu --set=full -o profile-hopper -f ./main
        "}
    }
}

pub struct GraphRunnerBuilder {
    dst_dir: PathBuf,
    fused_graph: Arc<FusedGraph>,
    opgraph: Arc<OpGraph>,
    fusion_kernel_names: HashMap<FusionId, String>,
    fusion_kernels: HashMap<FusionId, KernelGenerator>,
}

impl GraphRunnerBuilder {
    pub fn new(dst_dir: impl Into<PathBuf>, fused_graph: &Arc<FusedGraph>) -> Self {
        let dst_dir = dst_dir.into();

        fs::create_dir_all(&dst_dir).ok();

        let mut makefile = MakefileBuilder::default();
        let mut fusion_kernel_names = HashMap::new();
        let mut fusion_kernels = HashMap::new();
        for (kernel_name_counter, fusion) in fused_graph.iter().enumerate() {
            let subgraph = fused_graph.get(fusion).make_subgraph(fused_graph.opgraph());
            let mut kernel_generator = KernelGenerator::new(&Arc::new(subgraph));
            let kernel_id = format!("kernel{kernel_name_counter}");

            let kernel = kernel_generator.generate(&kernel_id);
            if !dst_dir.join(format!("{kernel_id}.cu")).exists()
                || fs::read(dst_dir.join(format!("{kernel_id}.cu")))
                    .unwrap()
                    .as_slice()
                    != kernel.as_bytes()
            {
                fs::write(dst_dir.join(format!("{kernel_id}.cu")), kernel).unwrap();
            }
            fs::write(
                dst_dir.join(format!("{kernel_id}.h")),
                kernel_generator.generate_header(&kernel_id),
            )
            .unwrap();

            makefile.add_source_file(&kernel_id);
            fusion_kernel_names.insert(fusion, kernel_id);
            fusion_kernels.insert(fusion, kernel_generator);
        }
        fs::write(dst_dir.join("Makefile"), makefile.build_makefile()).unwrap();

        Self {
            dst_dir,
            fused_graph: fused_graph.clone(),
            opgraph: fused_graph.opgraph().clone(),
            fusion_kernel_names,
            fusion_kernels,
        }
    }

    pub fn finish(self) {
        let includes = self
            .fusion_kernel_names
            .values()
            .map(|name| format!("#include \"{name}.h\""))
            .collect_vec()
            .join("\n");

        let mut init = Vec::new();
        let mut invocations = Vec::new();
        let mut tensors: HashMap<(FusionId, OpNode), String> = HashMap::new();

        // (1) Randomize inputs
        for fusion_id in self.fused_graph.iter() {
            let fusion = self.fused_graph.get(fusion_id);
            let node = fusion.nodes().next().unwrap();
            if fusion.is_input_output(&self.opgraph)
                && let Op::Producer { shape, .. } = self.opgraph.get(node)
            {
                let size = shape.element_count();
                let tensor_id = format!("tensor{}", tensors.len());
                tensors.insert((fusion_id, node), tensor_id.clone());
                init.push(formatdoc! {"
                    thrust::host_vector<bfloat16_t> {tensor_id}_host({size});
                    fill_random({tensor_id}_host);
                    thrust::device_vector<bfloat16_t> {tensor_id}({tensor_id}_host);
                    CHECK_CUDA(cudaDeviceSynchronize());
                "});
            }
        }

        // (2) Execute operation fusions
        for fusion_id in self.fused_graph.topo_sort() {
            let fusion = self.fused_graph.get(fusion_id);
            if fusion.is_input_output(&self.opgraph) {
                continue;
            }

            let kernel_name = &self.fusion_kernel_names[&fusion_id];
            let kernel = &self.fusion_kernels[&fusion_id];

            let mut arguments = Vec::new();
            for arg_node in kernel.args_passed() {
                if fusion.contains_node(arg_node) {
                    // Output argument
                    let tensor_id = format!("tensor{}", tensors.len());
                    tensors.insert((fusion_id, arg_node), tensor_id.clone());
                    let size = self.opgraph.shape(arg_node).element_count();
                    init.push(format!(
                        "thrust::device_vector<bfloat16_t> {tensor_id}({size});"
                    ));
                    arguments.push(format!("{tensor_id}.data().get()"));
                } else {
                    // Input argument
                    let provider = fusion.dependencies[&arg_node];
                    let tensor_id = tensors.get(&(provider, arg_node)).unwrap();
                    arguments.push(format!("{tensor_id}.data().get()"));
                }
            }
            let arguments = arguments.join(", ");

            invocations.push(formatdoc! {"
                {kernel_name}({arguments});
            "});
        }

        let init = init.join("\n\n");
        let invocations = invocations.join("\n\n");

        let main_file = formatdoc! {"
            {includes}
            #include <cassert>
            #include <cstdio>
            #include <cstdlib>
            #include <random>
            #include <thrust/device_vector.h>
            #include <thrust/host_vector.h>
            #include <chrono>

            using namespace cutlass;

            template <typename T>
            static inline void _check_cuda(T result, const char *expr, const char *file,
                                           int line) {{
              if constexpr (std::is_same_v<T, cudaError_t>) {{
                if (result != cudaSuccess) {{
                  std::fprintf(stderr, \"CUDA runtime failure  %s (%d)\\n  at %s:%d : %s\\n\",
                               cudaGetErrorString(result), static_cast<int>(result), file,
                               line, expr);
                  std::abort();
                }}
              }} else {{
                static_assert(std::is_same_v<T, void>,
                              \"CHECK_CUDA used with unsupported return type\");
              }}
            }}
            
            #define CHECK_CUDA(expr) _check_cuda((expr), #expr, __FILE__, __LINE__)

            void fill_random(thrust::host_vector<bfloat16_t> &v, float lo = 0.0f,
                             float hi = 1.0f, uint32_t seed = std::random_device{{}}()) {{
              if (lo > hi)
                std::swap(lo, hi);

              std::mt19937 rng(seed);
              std::uniform_real_distribution<float> dist(lo, hi);

              const std::size_t n = v.size();
              for (std::size_t i = 0; i < n; ++i) {{
                float f = dist(rng);
                v[i] = static_cast<bfloat16_t>(f);
              }}
            }}

            int main() {{
                {init}

                int warmup_runs = 1;
                int timing_runs = 0;

                for (int i = 0; i < warmup_runs; ++i) {{
                    {invocations}
                }}
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaGetLastError());

                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < timing_runs; ++i) {{
                    {invocations}
                }}
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaGetLastError());

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << static_cast<double>(duration.count()) / timing_runs / 1000 << \" ms\" << std::endl;

                std::cout << \"Passed.\";
                return 0;
            }}
        "};
        fs::write(self.dst_dir.join("main.cu"), main_file.as_bytes()).unwrap();
    }
}
