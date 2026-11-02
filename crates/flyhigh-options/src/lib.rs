//! `#[derive(Options)]` (ARCHITECTURE.md): one declaration per option. The field's doc comment is its description, the
//! instance `describe` is called on gives its default, and `#[option(...)]` adds what the type cannot say:
//!
//! ```text
//! /// Fixed when a session is built.
//! #[derive(Options)]
//! pub struct SessionOptions {
//!     /// Refinement passes of the fine flow.
//!     #[option(values = "1 to 3")]
//!     pub k: u32,
//! }
//! ```
//!
//! `default = "..."` replaces the rendered default, `values` names the range or the choices, `c` the C field (the same
//! name when absent), `skip` leaves a field out. Generated on the struct: `GROUP` (the name without "Options"),
//! `GROUP_DOC` (its doc comment) and `describe(&self) -> Vec<flyhigh_engine::options::OptionInfo>`. A field without a
//! doc comment is a compile error: the reference must say something about every option.
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Expr, ExprLit, Fields, Lit, Meta};

#[proc_macro_derive(Options, attributes(option))]
pub fn derive_options(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input).unwrap_or_else(|e| e.to_compile_error()).into()
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let group = name.to_string().trim_end_matches("Options").to_string();
    let group_doc = doc_of(&input.attrs);
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => return Err(syn::Error::new_spanned(name, "Options needs named fields")),
        },
        _ => return Err(syn::Error::new_spanned(name, "Options derives on structs only")),
    };
    let mut rows = Vec::new();
    for f in fields {
        let attr = FieldAttr::parse(&f.attrs)?;
        if attr.skip {
            continue;
        }
        let ident = f.ident.as_ref().expect("named field");
        let fname = ident.to_string();
        let doc = doc_of(&f.attrs);
        if doc.is_empty() {
            return Err(syn::Error::new_spanned(f, format!("option `{fname}` has no doc comment")));
        }
        let default = match &attr.default {
            Some(d) => quote! { ::std::string::String::from(#d) },
            None => quote! { ::flyhigh_engine::options::default_text(&::std::format!("{:?}", self.#ident)) },
        };
        let values = attr.values.clone().unwrap_or_default();
        let c = attr.c.clone().unwrap_or_else(|| fname.clone());
        rows.push(quote! {
            ::flyhigh_engine::options::OptionInfo { name: #fname, default: #default, values: #values, c: #c, doc: #doc }
        });
    }
    Ok(quote! {
        impl #name {
            pub const GROUP: &'static str = #group;
            pub const GROUP_DOC: &'static str = #group_doc;
            /// Every option with its documentation; `self` supplies the defaults.
            pub fn describe(&self) -> ::std::vec::Vec<::flyhigh_engine::options::OptionInfo> {
                ::std::vec![#(#rows),*]
            }
        }
    })
}

#[derive(Default)]
struct FieldAttr {
    default: Option<String>,
    values: Option<String>,
    c: Option<String>,
    skip: bool,
}

impl FieldAttr {
    fn parse(attrs: &[Attribute]) -> syn::Result<FieldAttr> {
        let mut a = FieldAttr::default();
        for attr in attrs.iter().filter(|a| a.path().is_ident("option")) {
            attr.parse_nested_meta(|m| {
                if m.path.is_ident("skip") {
                    a.skip = true;
                    return Ok(());
                }
                let value: syn::LitStr = m.value()?.parse()?;
                let slot = if m.path.is_ident("default") {
                    &mut a.default
                } else if m.path.is_ident("values") {
                    &mut a.values
                } else if m.path.is_ident("c") {
                    &mut a.c
                } else {
                    return Err(m.error("expected default, values, c or skip"));
                };
                *slot = Some(value.value());
                Ok(())
            })?;
        }
        Ok(a)
    }
}

/// The `///` lines joined: one space between lines, a blank line a paragraph break.
fn doc_of(attrs: &[Attribute]) -> String {
    let mut out = String::new();
    for attr in attrs.iter().filter(|a| a.path().is_ident("doc")) {
        let Meta::NameValue(nv) = &attr.meta else { continue };
        let Expr::Lit(ExprLit { lit: Lit::Str(s), .. }) = &nv.value else { continue };
        let line = s.value();
        let line = line.trim();
        if line.is_empty() {
            out.push_str("\n\n");
        } else {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push(' ');
            }
            out.push_str(line);
        }
    }
    out
}
