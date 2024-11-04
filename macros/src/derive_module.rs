use darling::FromField;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Error, Field, Fields};

pub fn derive_module(input: DeriveInput) -> Result<TokenStream, Error> {
    let struct_ident = &input.ident;

    let (impl_visit_params, impl_visit_params_mut, impl_record, impl_load_config, impl_load_params) =
        match input.data {
            Data::Struct(struct_) => {
                let Statements {
                    visit_params,
                    visit_params_mut,
                    record,
                    load_config,
                    load_params,
                } = derive_on_fields(&struct_.fields)?;

                let field_idents = struct_
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, field)| field_ident(field, i))
                    .collect::<Vec<_>>();

                let bind = match &struct_.fields {
                    Fields::Named(_) => quote! { let Self {#(#field_idents,)*} = self; },
                    Fields::Unnamed(_) => quote! { let Self(#(#field_idents,)*) = self; },
                    Fields::Unit => quote! {},
                };
                let build = match &struct_.fields {
                    Fields::Named(_) => quote! { Self { #(#field_idents,)* } },
                    Fields::Unnamed(_) => quote! { Self(#(#field_idents,)*)},
                    Fields::Unit => quote! { #struct_ident },
                };

                (
                    quote! {
                        #bind
                        #(#visit_params)*
                    },
                    quote! {
                        #bind
                        #(#visit_params_mut)*
                    },
                    quote! {
                        #bind
                        #(#record)*
                        Ok(())
                    },
                    quote! {
                        #(#load_config)*
                        Ok(#build)
                    },
                    quote! {
                        #bind
                        #(#load_params)*
                        Ok(())
                    },
                )
            }
            Data::Enum(_enum_) => todo!(),
            Data::Union(union_) => {
                return Err(Error::new_spanned(
                    union_.union_token,
                    "unions not supported",
                ))
            }
        };

    let result = quote! {
        impl ::helium::module::Module for #struct_ident {
            fn visit_params(&self, visitor: &mut impl ::helium::module::ParamVisitor) {
                #impl_visit_params
            }

            fn visit_params_mut(&mut self, visitor: &mut impl ::helium::module::ParamMutVisitor) {
                #impl_visit_params_mut
            }

            fn record(&self, recorder: &mut impl ::helium::module::record::Recorder) -> Result<(), ::helium::module::record::RecordError> {
                #impl_record
            }

            fn load_config(loader: &mut impl ::helium::module::record::ConfigLoader, device: ::helium::Device) -> Result<Self, ::helium::module::record::RecordError> {
                #impl_load_config
            }

            fn load_params(&mut self, loader: &mut impl ::helium::module::record::ParamLoader) -> Result<(), ::helium::module::record::RecordError> {
                #impl_load_params
            }
        }
    };
    Ok(result)
}

#[derive(Default)]
struct Statements {
    visit_params: Vec<TokenStream>,
    visit_params_mut: Vec<TokenStream>,
    record: Vec<TokenStream>,
    load_config: Vec<TokenStream>,
    load_params: Vec<TokenStream>,
}

#[derive(Debug, FromField)]
#[darling(attributes(module), forward_attrs(allow, doc, cfg))]
struct FieldOptions {
    /// Field is a configuration field rather than a submodule/parameter.
    #[darling(default)]
    config: bool,
}

fn derive_on_fields(fields: &Fields) -> Result<Statements, Error> {
    let mut statements = Statements::default();

    for (i, field) in fields.iter().enumerate() {
        let options = FieldOptions::from_field(field)?;
        let field_ident = field_ident(field, i);
        let ident_str = field_ident.to_string();
        let field_ty = &field.ty;

        if !options.config {
            statements.visit_params.push(quote! {
                <#field_ty as ::helium::module::Module>::visit_params(#field_ident, visitor);
            });
            statements.visit_params_mut.push(quote! {
                <#field_ty as ::helium::module::Module>::visit_params_mut(#field_ident, visitor);
            });
        }

        if options.config {
            statements.record.push(quote! {
                 recorder.record_config(#ident_str, #field_ident)?;
            });
            statements.load_config.push(quote! {
                let #field_ident = loader.load_config(#ident_str)?;
            });
        } else {
            statements.record.push(quote! {
                recorder.record_submodule(#ident_str, #field_ident)?;
            });
            statements.load_config.push(quote! {
                let #field_ident = loader.load_submodule(#ident_str, device)?;
            });
            statements.load_params.push(quote! {
                 <#field_ty as ::helium::module::Module>::load_params(#field_ident, loader)?;
            });
        }
    }

    Ok(statements)
}

fn field_ident(field: &Field, field_index: usize) -> Ident {
    field.ident.clone().unwrap_or_else(|| {
        // Tuple field
        format_ident!("_{field_index}")
    })
}
