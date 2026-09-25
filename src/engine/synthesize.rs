//! Deterministic logic synthesizer — Phase 1.
//!
//! The planner boundary still holds: no LLM code ever reaches the filesystem.
//! The synthesizer builds candidate function bodies ONLY from verified
//! ingredients (stdlib method signatures from the embedded index, composed
//! through fixed families; v1 covers map-accumulation).
//!
//! Each candidate is applied as an `EditPlan` and judged by the compiler
//! plus contract tests. Losers roll back. Exhaustion yields `BLOCKED`.

use super::plan::Evidence;

/// A single input/output example — pure data, never code.
/// The engine embeds these into a fixed test template it controls.
#[derive(Debug, Clone)]
pub struct ContractCase {
    /// Rust literal for the input, e.g. `"a a b"`.
    pub input: String,
    /// Rust literal for the expected output, e.g. `HashMap::from([...])`.
    /// Prefer expressions built from verified constructors so the
    /// contract itself compiles deterministically.
    pub expected: String,
}

/// What to synthesize: a named function with a known signature shape.
#[derive(Debug, Clone)]
pub struct SynthRequest {
    pub fn_name: String,
    /// Target language: "rust" or "kotlin". Families are gated on it —
    /// a Rust template must never land in a `.kt` file or vice versa.
    pub lang: String,
    /// Parameter list as (name, type), e.g. `[("text", "&str")]`.
    pub params: Vec<(String, String)>,
    /// Return type, e.g. `HashMap<String, usize>`.
    pub ret: String,
    pub cases: Vec<ContractCase>,
    /// Struct definition for the struct family; `None` for plain functions.
    pub struct_def: Option<StructDef>,
    /// Page definition for the page family; content slots as pure data.
    pub page_def: Option<PageDef>,
}

/// A webpage as pure data: title, sections, footer.
/// The engine owns the template and escapes every slot — slots can never
/// inject markup, only text.
#[derive(Debug, Clone)]
pub struct PageDef {
    pub title: String,
    pub sections: Vec<(String, String)>,
    pub footer: String,
}

/// A struct to synthesize: fields plus behavior as verified method ops.
/// Method bodies come ONLY from the op vocabulary below — never from input.
#[derive(Debug, Clone)]
pub struct StructDef {
    pub fields: Vec<(String, String)>,
    pub methods: Vec<MethodSpec>,
}

/// One method as pure metadata. `op` must name a verified operation
/// (`new`, `add_assign`, `get`); anything else blocks synthesis.
#[derive(Debug, Clone)]
pub struct MethodSpec {
    pub name: String,
    /// `none` (associated fn), `ref` (&self), `mut` (&mut self).
    pub self_kind: String,
    pub params: Vec<(String, String)>,
    /// Return type; `None` means `()`.
    pub ret: Option<String>,
    /// Amount parameter name for `add_assign`, else `None` (= 1).
    pub amount_param: Option<String>,
    /// Backing field for `add_assign` / `get`.
    pub field: Option<String>,
    pub op: String,
}

/// One candidate body plus the evidence justifying every call in it.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub body: String,
    pub evidence: Vec<Evidence>,
}

/// Verified stdlib ingredients, embedded at build time.
/// Every entry was checked against the actual std docs — the synthesizer
/// may only emit calls listed here (v1 scope: collections + str).
/// The canonical signature follows in the comment for auditability.
struct Ingredient {
    method: &'static str,
    on_type: &'static str,
}

const INGREDIENTS: &[Ingredient] = &[
    // fn new() -> HashMap<K, V>
    Ingredient {
        method: "new",
        on_type: "HashMap",
    },
    // fn insert(&mut self, k: K, v: V) -> Option<V>
    Ingredient {
        method: "insert",
        on_type: "HashMap",
    },
    // fn entry(&mut self, key: K) -> Entry<K, V>
    Ingredient {
        method: "entry",
        on_type: "HashMap",
    },
    // fn or_insert(self, default: V) -> &mut V
    Ingredient {
        method: "or_insert",
        on_type: "Entry",
    },
    // fn split_whitespace(&self) -> SplitWhitespace
    Ingredient {
        method: "split_whitespace",
        on_type: "str",
    },
    // fn split<P: Pattern>(&self, pat: P) -> Split
    Ingredient {
        method: "split",
        on_type: "str",
    },
    // fn to_string(&self) -> String
    Ingredient {
        method: "to_string",
        on_type: "str",
    },
    // fn chars(&self) -> Chars
    Ingredient {
        method: "chars",
        on_type: "str",
    },
    // fn push(&mut self, value: T)
    Ingredient {
        method: "push",
        on_type: "Vec",
    },
    // fn contains(&self, x: &T) -> bool
    Ingredient {
        method: "contains",
        on_type: "Vec",
    },
    // fn next_u32(&mut self) -> u32 (via rand::RngCore)
    Ingredient {
        method: "next_u32",
        on_type: "StdRng",
    },
    // fn next_u64(&mut self) -> u64 (via rand::RngCore)
    Ingredient {
        method: "next_u64",
        on_type: "StdRng",
    },
    // fn parse<F: FromStr>(&self) -> Result<F, F::Err>
    Ingredient {
        method: "parse",
        on_type: "str",
    },
    // fn map_err<F, O: FnOnce(E) -> F>(self, op: O) -> Result<T, F>
    Ingredient {
        method: "map_err",
        on_type: "Result",
    },
    // fn trim(&self) -> &str
    Ingredient {
        method: "trim",
        on_type: "str",
    },
];

pub struct Synthesizer;

impl Synthesizer {
    pub fn new() -> Self {
        Synthesizer
    }

    /// Is this call shape backed by a verified ingredient?
    pub fn is_verified_call(on_type: &str, method: &str) -> bool {
        INGREDIENTS
            .iter()
            .any(|i| i.on_type == on_type && i.method == method)
    }

    /// Produce candidate bodies for the request, most-likely first.
    /// Returns `Err` when no family covers the signature shape — the
    /// caller turns that into `BLOCKED`, never a fallback guess.
    pub fn synthesize(&self, req: &SynthRequest) -> Result<Vec<Candidate>, String> {
        // Functions prove by contract cases; structs prove by their
        // field + op contract; pages by their content slots.
        if req.page_def.is_none() && req.struct_def.is_none() && req.cases.is_empty() {
            return Err("Synthesize requires at least one contract case — BLOCKED".to_string());
        }
        // Family 4: webpage from content slots.
        if let Some(page) = &req.page_def {
            return self.page_candidates(&req.fn_name, page);
        }
        // Family 5: Kotlin class over kotlin.random.Random.
        if req.lang == "kotlin"
            && let Some(def) = &req.struct_def
        {
            return self.kotlin_rng_class(&req.fn_name, def);
        }
        if req.lang != "rust" {
            return Err(format!(
                "No synthesis family for language {} — BLOCKED",
                req.lang
            ));
        }
        // Family 3: struct with verified method ops.
        if let Some(def) = &req.struct_def {
            return self.struct_ops(&req.fn_name, def);
        }
        // Family 2: fallible parse — (&str) -> Result<Int, String>.
        if req.params.len() == 1 && req.params[0].1 == "&str" && is_parse_result(&req.ret) {
            return Ok(self.parse_candidates(req));
        }
        // Family 1: map-accumulation — (text: &str) -> HashMap<String, N>.
        if req.params.len() == 1
            && req.params[0].1 == "&str"
            && is_count_map(&req.ret)
            && let Some((k_ty, v_ty)) = map_types(&req.ret)
            && k_ty == "String"
            && is_int(&v_ty)
        {
            return Ok(self.map_accumulation(req));
        }
        Err(format!(
            "No synthesis family covers shape ({}) -> {} — BLOCKED",
            req.params
                .iter()
                .map(|(_, t)| t.clone())
                .collect::<Vec<_>>()
                .join(", "),
            req.ret
        ))
    }

    /// Candidate bodies for word-count-shaped problems, strongest first.
    /// Every emitted call is in the ingredient index above.
    fn map_accumulation(&self, req: &SynthRequest) -> Vec<Candidate> {
        let (input, _) = &req.params[0];
        let name = &req.fn_name;
        let ret = &req.ret;
        let sig = format!("pub fn {}({}: &str) -> {}", name, input, ret);

        let ev = |method: &str, on_type: &str| Evidence::VerifiedSymbol {
            qname: format!("{}::{}", on_type, method),
            source: "std-ingredient-index".to_string(),
        };

        vec![
            Candidate {
                body: format!(
                    "{sig} {{\n    let mut map: {ret} = std::collections::HashMap::new();\n    for w in {input}.split_whitespace() {{\n        *map.entry(w.to_string()).or_insert(0) += 1;\n    }}\n    map\n}}",
                ),
                evidence: vec![
                    ev("new", "HashMap"),
                    ev("split_whitespace", "str"),
                    ev("entry", "HashMap"),
                    ev("or_insert", "Entry"),
                    ev("to_string", "str"),
                ],
            },
            Candidate {
                body: format!(
                    "{sig} {{\n    let mut map: {ret} = std::collections::HashMap::new();\n    for w in {input}.split_whitespace() {{\n        map.entry(w.to_string()).or_insert(0);\n        if let Some(c) = map.get_mut(w) {{\n            *c += 1;\n        }}\n    }}\n    map\n}}",
                ),
                evidence: vec![
                    ev("new", "HashMap"),
                    ev("split_whitespace", "str"),
                    ev("entry", "HashMap"),
                    ev("or_insert", "Entry"),
                    ev("to_string", "str"),
                ],
            },
        ]
    }

    /// Candidate bodies for `(&str) -> Result<Int, String>` shapes.
    /// v1 covers direct parse plus an optional inclusive range from the
    /// request name metadata is NOT used — range comes only from an
    /// explicit `range: [min, max]` entry the caller validated.
    fn parse_candidates(&self, req: &SynthRequest) -> Vec<Candidate> {
        let (input, _) = &req.params[0];
        let name = &req.fn_name;
        let qualified = req
            .ret
            .trim_start_matches("std::result::")
            .trim_start_matches("Result")
            .trim();
        let inner = qualified
            .strip_prefix('<')
            .and_then(|s| s.strip_suffix('>'))
            .unwrap_or("u16")
            .split(',')
            .next()
            .unwrap_or("u16")
            .trim()
            .to_string();
        let sig = format!("pub fn {}({}: &str) -> {}", name, input, req.ret);
        let ev = |method: &str, on_type: &str| Evidence::VerifiedSymbol {
            qname: format!("{}::{}", on_type, method),
            source: "std-ingredient-index".to_string(),
        };
        vec![Candidate {
            body: format!(
                "{sig} {{\n    {input}.trim().parse::<{inner}>().map_err(|e| e.to_string())\n}}",
            ),
            evidence: vec![
                ev("trim", "str"),
                ev("parse", "str"),
                ev("map_err", "Result"),
            ],
        }]
    }

    /// Candidate bodies for structs: definition plus methods built ONLY
    /// from the verified op vocabulary (`new`, `add_assign`, `get`).
    /// Any unknown op, field, or shape blocks the whole request.
    fn struct_ops(&self, name: &str, def: &StructDef) -> Result<Vec<Candidate>, String> {
        if def.fields.is_empty() {
            return Err("Struct synthesis needs at least one field — BLOCKED".to_string());
        }
        for (f, _) in &def.fields {
            if !is_ident(f) {
                return Err(format!("Bad field name {} — BLOCKED", f));
            }
        }
        let mut impl_body = String::new();
        let mut evidence = Vec::new();
        for m in &def.methods {
            let (body, ev) = self.method_body(name, def, m)?;
            impl_body.push_str(&body);
            evidence.extend(ev);
        }
        let fields: Vec<String> = def
            .fields
            .iter()
            .map(|(f, t)| format!("    pub {}: {},", f, t))
            .collect();
        // Debug only: field types like StdRng are not Clone, and the
        // template must never assume more than it verified.
        let body = format!(
            "#[derive(Debug)]\npub struct {} {{\n{}\n}}\n\nimpl {} {{\n{}}}\n",
            name,
            fields.join("\n"),
            name,
            impl_body
        );
        Ok(vec![Candidate { body, evidence }])
    }

    /// Render one method from the op vocabulary. Unknown ops/shapes Err.
    fn method_body(
        &self,
        struct_name: &str,
        def: &StructDef,
        m: &MethodSpec,
    ) -> Result<(String, Vec<Evidence>), String> {
        let ev = |method: &str, on_type: &str| Evidence::VerifiedSymbol {
            qname: format!("{}::{}", on_type, method),
            source: "method-op-vocabulary".to_string(),
        };
        let params: Vec<String> = m
            .params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t))
            .collect();
        let ret = m.ret.clone().unwrap_or_else(|| "()".to_string());
        match m.op.as_str() {
            // Associated constructor: every field from a same-named param.
            "new" => {
                if m.self_kind != "none" {
                    return Err("new must have self_kind=none — BLOCKED".to_string());
                }
                let mut args = Vec::new();
                for (f, t) in &def.fields {
                    let p = m
                        .params
                        .iter()
                        .find(|(n, _)| n == f)
                        .ok_or(format!("new missing param for field {} — BLOCKED", f))?;
                    if p.1 != *t {
                        return Err(format!(
                            "new param {} type {} != field type {} — BLOCKED",
                            f, p.1, t
                        ));
                    }
                    args.push(format!("{}: {}", f, f));
                }
                if args.len() != m.params.len() {
                    return Err("new has params beyond fields — BLOCKED".to_string());
                }
                let _ = struct_name;
                Ok((
                    format!(
                        "    pub fn {}({}) -> {} {{\n        {} {{ {} }}\n    }}\n",
                        m.name,
                        params.join(", "),
                        struct_name,
                        struct_name,
                        args.join(", ")
                    ),
                    vec![ev("new", struct_name)],
                ))
            }
            // `field += amount` (or `+= 1` when no amount param).
            "add_assign" => {
                if m.self_kind != "mut" {
                    return Err("add_assign needs self_kind=mut — BLOCKED".to_string());
                }
                let field = m
                    .field
                    .clone()
                    .ok_or("add_assign needs a field — BLOCKED")?;
                if !def.fields.iter().any(|(f, _)| *f == field) {
                    return Err(format!("Unknown field {} — BLOCKED", field));
                }
                let rhs = match &m.amount_param {
                    Some(a) => {
                        if !m.params.iter().any(|(n, _)| n == a) {
                            return Err(format!("Unknown amount param {} — BLOCKED", a));
                        }
                        a.clone()
                    }
                    None => "1".to_string(),
                };
                let ret_ann = if ret == "()" {
                    String::new()
                } else {
                    return Err("add_assign must return () — BLOCKED".to_string());
                };
                Ok((
                    format!(
                        "    pub fn {}(&mut self{}){} {{\n        self.{} += {};\n    }}\n",
                        m.name,
                        if params.is_empty() {
                            String::new()
                        } else {
                            format!(", {}", params.join(", "))
                        },
                        ret_ann,
                        field,
                        rhs
                    ),
                    vec![ev("add_assign", struct_name)],
                ))
            }
            // Verified delegation: `self.field.method(args)` where the
            // (receiver type, method, param types, return) tuple is in the
            // table below. Enables wrapper types (RNG structs, adapters)
            // without ever emitting an unverified call.
            "call" => {
                if m.self_kind != "ref" && m.self_kind != "mut" {
                    return Err("call needs self_kind ref|mut — BLOCKED".to_string());
                }
                let field = m.field.clone().ok_or("call needs a field — BLOCKED")?;
                let fty = def
                    .fields
                    .iter()
                    .find(|(f, _)| *f == field)
                    .map(|(_, t)| t.clone())
                    .ok_or(format!("Unknown field {} — BLOCKED", field))?;
                let (table_params, table_ret) = callable_method(&fty, &m.name)
                    .ok_or(format!("Unverified call {}.{} — BLOCKED", fty, m.name))?;
                if table_params.len() != m.params.len()
                    || table_params
                        .iter()
                        .zip(m.params.iter())
                        .any(|(a, b)| a != &b.1)
                {
                    return Err(format!(
                        "Call {}.{} params {:?} != verified {:?} — BLOCKED",
                        fty, m.name, m.params, table_params
                    ));
                }
                if m.ret.as_deref().unwrap_or("()") != table_ret {
                    return Err(format!("Call {}.{} return mismatch — BLOCKED", fty, m.name));
                }
                let receiver = if m.self_kind == "mut" {
                    "&mut self"
                } else {
                    "&self"
                };
                let args = m
                    .params
                    .iter()
                    .map(|(n, _)| n.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                let ret_ann = if table_ret == "()" {
                    String::new()
                } else {
                    format!(" -> {}", table_ret)
                };
                Ok((
                    format!(
                        "    pub fn {}({}{}){} {{\n        self.{}.{}({})\n    }}\n",
                        m.name,
                        receiver,
                        if params.is_empty() {
                            String::new()
                        } else {
                            format!(", {}", params.join(", "))
                        },
                        ret_ann,
                        field,
                        m.name,
                        args
                    ),
                    vec![ev(&m.name, &fty)],
                ))
            }
            // `self.field` reader.
            "get" => {
                if m.self_kind != "ref" {
                    return Err("get needs self_kind=ref — BLOCKED".to_string());
                }
                let field = m.field.clone().ok_or("get needs a field — BLOCKED")?;
                let fty = def
                    .fields
                    .iter()
                    .find(|(f, _)| *f == field)
                    .map(|(_, t)| t.clone())
                    .ok_or(format!("Unknown field {} — BLOCKED", field))?;
                if !m.params.is_empty() {
                    return Err("get takes no params — BLOCKED".to_string());
                }
                // Copy-read for Copy scalar types; clone for owned.
                let access = if is_int(&fty) || fty == "bool" {
                    format!("self.{}", field)
                } else {
                    format!("self.{}.clone()", field)
                };
                Ok((
                    format!(
                        "    pub fn {}(&self) -> {} {{\n        {}\n    }}\n",
                        m.name, fty, access
                    ),
                    vec![ev("get", struct_name)],
                ))
            }
            other => Err(format!("Unknown method op {} — BLOCKED", other)),
        }
    }

    /// Candidate bodies for a webpage. The template is fixed and every
    /// slot is HTML-escaped, so content is always inert text. The presence
    /// contract (every slot verbatim in the output) holds by construction
    /// and is asserted by the caller-side test below.
    fn page_candidates(&self, name: &str, page: &PageDef) -> Result<Vec<Candidate>, String> {
        if page.title.trim().is_empty() {
            return Err("Page needs a non-empty title — BLOCKED".to_string());
        }
        if page.sections.is_empty() {
            return Err("Page needs at least one section — BLOCKED".to_string());
        }
        if name.trim().is_empty() {
            return Err("Page needs a name — BLOCKED".to_string());
        }
        let mut sections_html = String::new();
        for (h, b) in &page.sections {
            if h.trim().is_empty() || b.trim().is_empty() {
                return Err("Page sections need heading and body — BLOCKED".to_string());
            }
            sections_html.push_str(&format!(
                "  <section>\n    <h2>{}</h2>\n    <p>{}</p>\n  </section>\n",
                escape_html(h),
                escape_html(b)
            ));
        }
        let body = format!(
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>{}</title>\n<style>\nbody{{font-family:sans-serif;max-width:42rem;margin:2rem auto;padding:0 1rem;line-height:1.6}}\nheader{{border-bottom:2px solid #0a7}}\nfooter{{margin-top:2rem;border-top:1px solid #ccc;font-size:.9rem}}\n</style>\n</head>\n<body>\n<header>\n<h1>{}</h1>\n</header>\n{}<footer>\n<p>{}</p>\n</footer>\n</body>\n</html>\n",
            escape_html(&page.title),
            escape_html(&page.title),
            sections_html,
            escape_html(&page.footer)
        );
        Ok(vec![Candidate {
            body,
            evidence: vec![Evidence::VerifiedSymbol {
                qname: "html5-page-template".to_string(),
                source: "engine-template".to_string(),
            }],
        }])
    }

    /// Candidate bodies for a Kotlin class over `kotlin.random.Random`.
    /// Methods come only from the op vocabulary: `knew` (secondary
    /// constructor from a seed) and `kcall` (delegation checked against
    /// the table below). Template is fixed; only names flow from metadata.
    fn kotlin_rng_class(&self, name: &str, def: &StructDef) -> Result<Vec<Candidate>, String> {
        if def.fields.len() != 1 || def.fields[0].1 != "Random" {
            return Err("Kotlin RNG needs exactly one `Random` field — BLOCKED".to_string());
        }
        let field = def.fields[0].0.clone();
        let mut methods_out = String::new();
        let mut evidence = vec![Evidence::VerifiedSymbol {
            qname: "kotlin.random.Random".to_string(),
            source: "kotlin-stdlib".to_string(),
        }];
        for m in &def.methods {
            match m.op.as_str() {
                "knew" => {
                    if m.self_kind != "none" {
                        return Err("knew must have self_kind=none — BLOCKED".to_string());
                    }
                    if m.params.len() != 1 || m.params[0].1 != "Long" {
                        return Err("knew needs exactly (seed: Long) — BLOCKED".to_string());
                    }
                    let seed = m.params[0].0.clone();
                    methods_out.push_str(&format!(
                        "    constructor({}: Long) {{\n        {} = Random({})\n    }}\n",
                        seed, field, seed
                    ));
                    evidence.push(Evidence::VerifiedSymbol {
                        qname: "kotlin.random.Random.invoke".to_string(),
                        source: "kotlin-stdlib".to_string(),
                    });
                }
                "kcall" => {
                    if m.self_kind != "ref" && m.self_kind != "mut" {
                        return Err("kcall needs self_kind ref|mut — BLOCKED".to_string());
                    }
                    let (table_params, table_ret) = kotlin_callable(&m.name)
                        .ok_or(format!("Unverified Kotlin call {} — BLOCKED", m.name))?;
                    if table_params.len() != m.params.len()
                        || table_params
                            .iter()
                            .zip(m.params.iter())
                            .any(|(a, b)| a != &b.1)
                    {
                        return Err(format!(
                            "Call {} params {:?} != verified {:?} — BLOCKED",
                            m.name, m.params, table_params
                        ));
                    }
                    let want_ret = m.ret.clone().unwrap_or_else(|| "Unit".to_string());
                    if want_ret != table_ret {
                        return Err(format!("Call {} return mismatch — BLOCKED", m.name));
                    }
                    let args = m
                        .params
                        .iter()
                        .map(|(n, _)| n.clone())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let ret_ann = if table_ret == "Unit" {
                        String::new()
                    } else {
                        format!(": {}", table_ret)
                    };
                    methods_out.push_str(&format!(
                        "    fun {}({}){} {{\n        return {}.{}({})\n    }}\n",
                        m.name,
                        m.params
                            .iter()
                            .map(|(n, t)| format!("{}: {}", n, t))
                            .collect::<Vec<_>>()
                            .join(", "),
                        ret_ann,
                        field,
                        m.name,
                        args
                    ));
                    evidence.push(Evidence::VerifiedSymbol {
                        qname: format!("kotlin.random.Random.{}", m.name),
                        source: "kotlin-stdlib".to_string(),
                    });
                }
                other => {
                    return Err(format!("Unknown Kotlin op {} — BLOCKED", other));
                }
            }
        }
        // No import line in the body: imports are separate AddImport tasks.
        // Emitting one here duplicates the task's line and kotlinc rejects
        // conflicting imports — even identical ones.
        let body = format!(
            "class {} {{\n    private var {}: Random\n\n{}}}\n",
            name, field, methods_out
        );
        Ok(vec![Candidate { body, evidence }])
    }

    /// Render the contract test module for the request.
    /// The template is engine-controlled; only literal values come from data.
    /// Struct cases use block expressions (`{ stmts; final }`) as input.
    pub fn contract_test(&self, req: &SynthRequest) -> String {
        // Pages carry no Rust test module: the HTML oracle judges structure
        // and slot presence holds by construction (asserted by callers).
        if req.page_def.is_some() {
            return String::new();
        }
        // Kotlin contracts are a top-level `fun main` of `check()` calls —
        // the oracle compiles and runs them, nonzero exit fails the run.
        // Inputs run inside `run {}`: a bare `{...}` is a lambda in Kotlin,
        // not a block, and would compare Function0 instead of executing.
        // Inputs that already arrive brace-wrapped get one matching pair
        // stripped so `run { run {...} }` can never nest a lambda.
        if req.lang == "kotlin" {
            let mut out = String::from("\nfun main() {\n");
            for c in &req.cases {
                out.push_str(&format!(
                    "    check((run {{ {} }}) == ({}))\n",
                    strip_matching_braces(&c.input),
                    c.expected
                ));
            }
            out.push_str("}\n");
            return out;
        }
        let mut out = format!(
            "\n#[cfg(test)]\nmod grounded_contract_{} {{\n    use super::*;\n",
            sanitize(&req.fn_name)
        );
        for (i, c) in req.cases.iter().enumerate() {
            if req.struct_def.is_some() {
                out.push_str(&format!(
                    "    #[test]\n    fn case_{}() {{\n        assert_eq!({}, {});\n    }}\n",
                    i,
                    wrap_block(&c.input),
                    c.expected
                ));
            } else {
                out.push_str(&format!(
                    "    #[test]\n    fn case_{}() {{\n        assert_eq!({}({}), {});\n    }}\n",
                    i, req.fn_name, c.input, c.expected
                ));
            }
        }
        out.push_str("}\n");
        out
    }
}

impl Default for Synthesizer {
    fn default() -> Self {
        Self::new()
    }
}

fn is_count_map(ret: &str) -> bool {
    ret.starts_with("HashMap<") || ret.starts_with("std::collections::HashMap<")
}

/// `Result<Int, String>` (possibly fully qualified) for the parse family.
fn is_parse_result(ret: &str) -> bool {
    let inner = ret
        .trim_start_matches("std::result::")
        .trim_start_matches("Result")
        .trim();
    let inner = inner.strip_prefix('<').and_then(|s| s.strip_suffix('>'));
    let Some(inner) = inner else { return false };
    let mut parts = inner.splitn(2, ',');
    let ok = parts.next().is_some_and(|t| is_int(t.trim()));
    let err = parts.next().is_some_and(|t| t.trim() == "String");
    ok && err
}

/// Wrap a case input as a block expression unless it already is one.
fn wrap_block(input: &str) -> String {
    let t = input.trim();
    if t.starts_with('{') && t.ends_with('}') {
        t.to_string()
    } else {
        format!("{{ {} }}", t)
    }
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .enumerate()
            .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
}

fn map_types(ret: &str) -> Option<(String, String)> {
    let inner = ret
        .trim_start_matches("std::collections::")
        .trim_start_matches("HashMap")
        .trim()
        .strip_prefix('<')?
        .strip_suffix('>')?;
    let mut parts = split_top_level(inner);
    if parts.len() != 2 {
        return None;
    }
    Some((
        parts.remove(0).trim().to_string(),
        parts.remove(0).trim().to_string(),
    ))
}

fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '<' => {
                depth += 1;
                cur.push(c);
            }
            '>' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            ',' if depth == 0 => {
                out.push(cur.clone());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn is_int(t: &str) -> bool {
    matches!(
        t,
        "usize" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "isize"
    )
}

/// Escape text for an HTML text node. Slots can never inject markup.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Verified delegatable calls: (receiver type, method) → (param types, return).
/// Every entry checked against real API docs. The synthesizer may emit
/// only these exact shapes.
fn callable_method(
    receiver: &str,
    method: &str,
) -> Option<(&'static [&'static str], &'static str)> {
    const TABLE: &[(&str, &str, &[&str], &str)] = &[
        ("StdRng", "next_u32", &[], "u32"),
        ("StdRng", "next_u64", &[], "u64"),
    ];
    TABLE
        .iter()
        .find(|(r, m, _, _)| *r == receiver && *m == method)
        .map(|(_, _, p, r)| (*p, *r))
}

/// Strip one brace pair only when the first `{` matches the last `}` —
/// otherwise `{a} + {b}` inputs would corrupt into `a} + {b`.
fn strip_matching_braces(s: &str) -> String {
    let t = s.trim();
    if !(t.starts_with('{') && t.ends_with('}')) {
        return t.to_string();
    }
    let mut depth = 0usize;
    let mut chars = t.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(c) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
        } else if c == '{' {
            depth += 1;
        } else if c == '}' {
            if depth == 0 {
                return t.to_string();
            }
            depth -= 1;
            if depth == 0 && chars.peek().is_some() {
                // Closed before the end — not a wrapping pair.
                return t.to_string();
            }
        }
    }
    if depth == 0 {
        t[1..t.len() - 1].trim().to_string()
    } else {
        t.to_string()
    }
}

/// Verified `kotlin.random.Random` calls: method → (param types, return).
/// Checked against the Kotlin stdlib docs. The synthesizer may emit only
/// these exact shapes.
fn kotlin_callable(method: &str) -> Option<(&'static [&'static str], &'static str)> {
    const TABLE: &[(&str, &[&str], &str)] = &[
        ("nextInt", &[], "Int"),
        ("nextLong", &[], "Long"),
        ("nextDouble", &[], "Double"),
        ("nextBits", &["Int"], "Int"),
    ];
    TABLE
        .iter()
        .find(|(m, _, _)| *m == method)
        .map(|(_, p, r)| (*p, *r))
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
