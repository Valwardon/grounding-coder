//! GroundingType derive: engine-level type descriptions.
//!
//! `#[derive(GroundingType)]` on a struct generates `grounding_schema()`
//! returning field names, Rust types (as strings), doc comments, and
//! morph flags (`#[morph]` fields). The deterministic engine inspects
//! people through schemas, never through struct internals or stringly
//! field access.
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Marker attribute: this field is a morph target (0..1 deformer).
/// Parsed and recorded; generates no code beyond the flag.
#[proc_macro_derive(GroundingType, attributes(morph))]
pub fn derive_grounding_type(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(&ast.ident, "GroundingType needs named fields")
                    .to_compile_error()
                    .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&ast.ident, "GroundingType derives on structs only")
                .to_compile_error()
                .into();
        }
    };
    let mut entries = Vec::new();
    for field in fields {
        let fname = field.ident.as_ref().unwrap().to_string();
        let fty = match &field.ty {
            syn::Type::Path(p) => p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            _ => "unknown".to_string(),
        };
        let mut doc = String::new();
        let mut is_morph = false;
        for attr in &field.attrs {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(lit) = &nv.value {
                        if let syn::Lit::Str(s) = &lit.lit {
                            if !doc.is_empty() {
                                doc.push(' ');
                            }
                            doc.push_str(s.value().trim());
                        }
                    }
                }
            }
            if attr.path().is_ident("morph") {
                is_morph = true;
            }
        }
        entries.push(quote! {
            ::grounding_coder::grounding::GroundingField {
                name: #fname,
                ty: #fty,
                doc: #doc,
                is_morph: #is_morph,
            }
        });
    }
    let type_name = name.to_string();
    let expanded = quote! {
        impl #name {
            /// Engine-level description: fields, types, docs, morphs.
            pub fn grounding_schema() -> ::grounding_coder::grounding::GroundingSchema {
                ::grounding_coder::grounding::GroundingSchema {
                    type_name: #type_name,
                    fields: vec![#(#entries),*],
                }
            }
        }
    };
    expanded.into()
}
