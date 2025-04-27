use std::{env, fs::File, path::PathBuf};

fn main() {
    let cutlass_path = format!("{}/cutlass/include", env!("CARGO_MANIFEST_DIR"));
    let cuda_include_path = "/usr/local/cuda/include"; // TODO - cross-platform
    println!("cargo:rerun-if-changed={cutlass_path}");
    println!("cargo:rerun-if-changed={cuda_include_path}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut cuda_tarball = tar::Builder::new(
        zstd::Encoder::new(File::create(out_dir.join("cuda.tar.zst")).unwrap(), 10).unwrap(),
    );
    cuda_tarball.append_dir_all(".", cuda_include_path).unwrap();
    cuda_tarball.into_inner().unwrap().finish().unwrap();

    let mut cutlass_tarball = tar::Builder::new(
        zstd::Encoder::new(File::create(out_dir.join("cutlass.tar.zst")).unwrap(), 10).unwrap(),
    );
    cutlass_tarball.append_dir_all(".", cutlass_path).unwrap();
    cutlass_tarball.into_inner().unwrap().finish().unwrap();
}
