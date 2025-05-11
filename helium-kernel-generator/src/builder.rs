use crate::architecture::Architecture;
use cudarc::{
    nvrtc,
    nvrtc::{CompileOptions, Ptx},
};
use helium_ir::data_type::{DataClass, DataType};
use std::{
    cell::{Cell, RefCell},
    fmt::{Display, Formatter},
    io::Cursor,
    rc::Rc,
    sync::OnceLock,
};
use tempfile::TempDir;

/// Utility to build a kernel.
#[derive(Default)]
pub struct KernelBuilder {
    name: String,
    sections: Vec<(String, Rc<RefCell<String>>)>,
    symbol_count: Rc<Cell<u32>>,
}

impl KernelBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn new_symbol(&mut self) -> Symbol {
        let s = self.symbol_count.get();
        self.symbol_count.set(s.checked_add(1).unwrap());
        Symbol(s)
    }

    pub fn add_section(&mut self, name: impl AsRef<str>) -> Section {
        self.sections.push((
            name.as_ref().to_string(),
            Rc::new(RefCell::new(String::new())),
        ));
        self.section(name.as_ref())
    }

    pub fn dangling_section(&mut self) -> Section {
        Section {
            src: Rc::new(RefCell::new(String::new())),
            symbol_count: self.symbol_count.clone(),
        }
    }

    pub fn section(&mut self, name: impl AsRef<str>) -> Section {
        let source = self
            .sections
            .iter_mut()
            .find(|(sec, _)| sec == name.as_ref())
            .expect("missing section");
        Section {
            src: source.1.clone(),
            symbol_count: self.symbol_count.clone(),
        }
    }

    pub fn build_source(&self) -> String {
        let mut s = String::new();
        for (_, section) in &self.sections {
            s.push_str(&section.borrow());
            s.push('\n');
        }
        s
    }

    pub fn compile(&self, target_arch: Architecture) -> Result<Ptx, crate::Error> {
        let source = self.build_source();
        let bundled_headers = BundledHeaders::get()?;

        nvrtc::compile_ptx_with_opts(
            &source,
            CompileOptions {
                ftz: None,
                prec_sqrt: None,
                prec_div: None,
                fmad: None,
                options: vec![
                    target_arch.nvrtc_flag(),
                    "-pch".into(),
                    "-dopt=on".into(),
                    "-G".into(),
                    "-default-device".into(),
                ],
                use_fast_math: None,
                maxrregcount: None,
                include_paths: vec![bundled_headers.dir.path().to_str().unwrap().to_string()],
                arch: None,
                name: Some(self.name.clone()),
            },
        )
        .map_err(crate::Error::from)
    }
}

/// We bundle necessary header files into the binary
/// and extract them to a temporary directory at runtime.
struct BundledHeaders {
    dir: TempDir,
}

impl BundledHeaders {
    pub fn get() -> Result<&'static Self, crate::Error> {
        static HEADERS: OnceLock<BundledHeaders> = OnceLock::new();

        HEADERS.get_or_try_init(|| {
            static TARBALL_CUDA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda.tar.zst"));
            static TARBALL_CUTLASS: &[u8] =
                include_bytes!(concat!(env!("OUT_DIR"), "/cutlass.tar.zst"));

            let dir = TempDir::new()?;

            tar::Archive::new(zstd::Decoder::new(Cursor::new(TARBALL_CUDA))?).unpack(dir.path())?;
            tar::Archive::new(zstd::Decoder::new(Cursor::new(TARBALL_CUTLASS))?)
                .unpack(dir.path())?;

            Ok::<_, crate::Error>(BundledHeaders { dir })
        })
    }
}

pub struct Section {
    src: Rc<RefCell<String>>,
    symbol_count: Rc<Cell<u32>>,
}

impl Display for Section {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.src.borrow().as_str().fmt(f)
    }
}

impl Section {
    pub fn new_symbol(&mut self) -> Symbol {
        let s = self.symbol_count.get();
        self.symbol_count.set(s.checked_add(1).unwrap());
        Symbol(s)
    }

    pub fn emit(&mut self, source: impl AsRef<str>) -> &mut Self {
        self.src.borrow_mut().push_str(source.as_ref());
        self
    }
}

/// A unique identifier within a kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Symbol(u32);

impl Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "local{}", self.0)
    }
}

pub fn cpp_data_class(data_class: DataClass) -> &'static str {
    match data_class {
        DataClass::Float => "float",
        DataClass::Int => "uint32_t",
        DataClass::Bool => "bool",
    }
}

pub fn cpp_data_type(data_type: DataType) -> &'static str {
    match data_type {
        DataType::F16 => "half_t",
        DataType::Bf16 => "bfloat16_t",
        DataType::F32 => "float",
        DataType::U32 => "uint32_t",
        DataType::Bool => "bool",
    }
}

pub fn cpp_sizeof(data_type: DataType) -> u32 {
    match data_type {
        DataType::F16 => 2,
        DataType::Bf16 => 2,
        DataType::F32 => 4,
        DataType::U32 => 4,
        DataType::Bool => 1,
    }
}

pub fn cpp_raw_byte_type(bytes: u32) -> &'static str {
    match bytes {
        1 => "uint8_t",
        2 => "uint16_t",
        4 => "uint32_t",
        8 => "uint64_t",
        16 => "uint128_t",
        _ => panic!("unsupported raw byte type for {bytes} bytes"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    #[test]
    fn trivial_compile() {
        let source = indoc! {r#"
        #include <cuda_bf16.h>
        #include <cute/int_tuple.hpp>
        
        extern "C"  __global__ __launch_bounds__(256) void kernel() {}
        "#};

        let mut kernel = KernelBuilder::new("trivial_kernel");
        kernel.add_section("main");
        kernel.section("main").emit(source);
        kernel.compile(Architecture::Sm120a).unwrap();
    }
}
