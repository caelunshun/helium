use std::{
    cell::{Cell, RefCell},
    fmt::{Display, Formatter},
    rc::Rc,
};

/// Utility to build a kernel.
#[derive(Default)]
pub struct KernelBuilder {
    #[allow(unused)]
    name: String,
    sections: Vec<(String, Rc<RefCell<String>>)>,
    symbol_count: Rc<Cell<u32>>,
    dynamic_smem_amount: u32,
}

impl KernelBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    pub fn add_dynamic_smem(&mut self, bytes: u32) {
        self.dynamic_smem_amount += bytes;
    }

    pub fn dynamic_smem_bytes(&self) -> u32 {
        self.dynamic_smem_amount
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
