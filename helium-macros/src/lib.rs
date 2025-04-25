use syn::DeriveInput;

mod derive_module;

#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as DeriveInput);
    derive_module::derive_module(input)
        .unwrap_or_else(|err| err.into_compile_error())
        .into()
}
