use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for a code symbol in the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u64);

impl SymbolId {
    pub const ZERO: Self = SymbolId(0);
    pub const ROOT: Self = SymbolId(1); // Root node = project root

    pub fn fresh() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(2);
        SymbolId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn from_raw(u: u64) -> Self {
        SymbolId(u)
    }
}

/// What kind of code symbol this node represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    File,      // Source file
    Module,    // Rust module / Kotlin file / package
    Struct,    // struct / class
    Enum,      // enum / sealed class
    Function,  // fn / fun
    Import,    // use / import
    Field,     // struct field
    Const,     // constant
    Trait,     // trait / interface
    Macro,     // macro invocation
    TypeParam, // generic type parameter
    Unknown,   // fallback for unparseable tokens
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl std::fmt::Display for SymbolRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl SymbolKind {
    pub fn label(&self) -> &'static str {
        match self {
            SymbolKind::File => "file",
            SymbolKind::Module => "module",
            SymbolKind::Struct => "struct",
            SymbolKind::Enum => "enum",
            SymbolKind::Function => "fn",
            SymbolKind::Import => "import",
            SymbolKind::Field => "field",
            SymbolKind::Const => "const",
            SymbolKind::Trait => "trait",
            SymbolKind::Macro => "macro",
            SymbolKind::TypeParam => "typeparam",
            SymbolKind::Unknown => "?",
        }
    }
}

/// Typed relationship between two code symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolRelation {
    /// file A contains/imports module B
    Contains,
    /// file A imports/uses B (use statement)
    Imports,
    /// function A calls function B
    Calls,
    /// code A uses type B
    UsesType,
    /// function A returns type B
    Returns,
    /// struct A inherits/extends B
    Inherits,
    /// code A references symbol B
    References,
    /// struct A has field B
    HasField,
    /// enum A has variant B
    HasVariant,
    /// macro A expands to B
    ExpandsTo,
}

/// Data-flow contract between two symbols (from grounded's InvariantContract).
/// Prevents invalid symbol resolution chains.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataContract {
    /// Source is a definition, target is a usage (type compatibility required)
    DefinitionUse {
        output_type: TypeShape,
        input_type: TypeShape,
    },
    /// Source calls target (callee must be callable from caller context)
    Call {
        caller_scope: SymbolId,
        callee_def: SymbolId,
    },
    /// Source imports target (target must be public/exported)
    Import,
    /// No formal contract — relies on symbol resolution only
    Unspecified,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeShape {
    /// A type with this name
    Type(String),
    /// Any type
    Any,
    /// A reference (&T)
    Ref,
    /// A mutable reference (&mut T)
    MutRef,
}

/// A code symbol — grounded's `GroundedNode` repurposed for code.
#[derive(Debug, Clone)]
pub struct CodeSymbol {
    pub id: SymbolId,
    /// Fully-qualified name: e.g., "MainActivity.setContentView" or "android.widget.Button"
    pub qname: String,
    pub kind: SymbolKind,
    /// Source file path relative to project root
    pub file_path: String,
    /// Line number in source file
    pub line: u32,
    /// Column number (for precise location)
    pub col: u32,
    /// Full signature if available: "fn setContentView(view: View)"
    pub signature: String,
    /// The raw source text of this symbol (for code copying/composition)
    pub source: Option<String>,
    /// Edges to other symbols
    pub edges: Vec<SymbolEdge>,
}

impl CodeSymbol {
    pub fn location(&self) -> String {
        format!("{}:{}:{}", self.file_path, self.line, self.col)
    }
}

/// An edge in the code graph — grounded's `Edge` repurposed.
#[derive(Debug, Clone)]
pub struct SymbolEdge {
    pub relation: SymbolRelation,
    pub target: SymbolId,
    /// Weight = confidence in this relationship (1.0 = certain, lower = inferred)
    pub weight: f64,
    /// Optional data-flow contract
    pub contract: DataContract,
}

impl SymbolEdge {
    pub fn new(relation: SymbolRelation, target: SymbolId) -> Self {
        SymbolEdge {
            relation,
            target,
            weight: relation.default_weight(),
            contract: DataContract::Unspecified,
        }
    }

    pub fn with_weight(relation: SymbolRelation, target: SymbolId, weight: f64) -> Self {
        SymbolEdge {
            relation,
            target,
            weight,
            contract: DataContract::Unspecified,
        }
    }

    pub fn with_contract(mut self, contract: DataContract) -> Self {
        self.contract = contract;
        self
    }
}

impl SymbolRelation {
    /// Default weight — grounded's `spread_weight()`. Higher = more
    /// certain/trusted relationship.
    pub fn default_weight(&self) -> f64 {
        match self {
            SymbolRelation::Contains => 1.0,
            SymbolRelation::Imports => 0.95,
            SymbolRelation::Calls => 0.9,
            SymbolRelation::UsesType => 0.85,
            SymbolRelation::Returns => 0.8,
            SymbolRelation::Inherits => 0.9,
            SymbolRelation::References => 0.5,
            SymbolRelation::HasField => 0.8,
            SymbolRelation::HasVariant => 0.7,
            SymbolRelation::ExpandsTo => 0.6,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SymbolRelation::Contains => "contains",
            SymbolRelation::Imports => "imports",
            SymbolRelation::Calls => "calls",
            SymbolRelation::UsesType => "uses_type",
            SymbolRelation::Returns => "returns",
            SymbolRelation::Inherits => "inherits",
            SymbolRelation::References => "references",
            SymbolRelation::HasField => "has_field",
            SymbolRelation::HasVariant => "has_variant",
            SymbolRelation::ExpandsTo => "expands_to",
        }
    }
}

/// Cached path verification result (from grounded's CachedPath).
/// Avoids re-verifying the same symbol dependency chains.
#[derive(Debug, Clone)]
#[allow(dead_code)] // metadata fields are stored for future equivalence checks
struct CachedPath {
    source: SymbolId,
    dest: SymbolId,
    relation_mask: u64,
    weight: f64,
}

/// Arena-backed graph of code symbols — grounded's `GraphArena` repurposed.
///
/// Uses arena allocation (like grounded): nodes stored in a Vec<RwLock<>>,
/// indexed by ID. Path cache for O(1) dependency verification.
pub struct CodeArena {
    nodes: Vec<RwLock<CodeSymbol>>,
    label_index: Vec<(String, SymbolId)>,
    path_cache: parking_lot::Mutex<[Option<CachedPath>; 16]>,
}

impl Default for CodeArena {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeArena {
    pub fn new() -> Self {
        let mut arena = CodeArena {
            nodes: Vec::new(),
            label_index: Vec::new(),
            path_cache: parking_lot::Mutex::new(std::array::from_fn(|_| None)),
        };
        // Index 0: null sentinel
        arena.nodes.push(RwLock::new(CodeSymbol {
            id: SymbolId::ZERO,
            qname: "<null>".into(),
            kind: SymbolKind::Unknown,
            file_path: String::new(),
            line: 0,
            col: 0,
            signature: String::new(),
            source: None,
            edges: Vec::new(),
        }));
        // Index 1: root (project root)
        arena.nodes.push(RwLock::new(CodeSymbol {
            id: SymbolId::ROOT,
            qname: "<root>".into(),
            kind: SymbolKind::Module,
            file_path: String::new(),
            line: 0,
            col: 0,
            signature: "project root".into(),
            source: None,
            edges: Vec::new(),
        }));
        arena.label_index.push(("<root>".into(), SymbolId::ROOT));
        arena
    }

    pub fn with_capacity(cap: usize) -> Self {
        let mut a = Self::new();
        a.nodes.reserve(cap);
        a
    }

    /// Insert a symbol, return its assigned ID.
    pub fn insert(&mut self, mut symbol: CodeSymbol) -> SymbolId {
        let id = SymbolId::from_raw(self.nodes.len() as u64);
        symbol.id = id;
        self.label_index.push((symbol.qname.clone(), id));
        self.nodes.push(RwLock::new(symbol));
        self.invalidate_cache();
        id
    }

    /// Find a symbol by qualified name.
    pub fn lookup(&self, qname: &str) -> Option<SymbolId> {
        self.label_index
            .iter()
            .find(|(l, _)| l == qname)
            .map(|(_, id)| *id)
    }

    /// Find a symbol by simple name (last component of qname).
    pub fn lookup_simple(&self, name: &str) -> Vec<SymbolId> {
        self.label_index
            .iter()
            .filter(|(l, _)| l == name || l.ends_with(name))
            .map(|(_, id)| *id)
            .collect()
    }

    pub fn get(&self, id: SymbolId) -> Option<&RwLock<CodeSymbol>> {
        self.nodes.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn all_symbols(&self) -> Vec<(String, CodeSymbol)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i > 0)
            .map(|(_, n)| {
                let node = n.read();
                (node.qname.clone(), node.clone())
            })
            .collect()
    }

    /// Add a directed edge from source to target.
    pub fn link(&mut self, source: SymbolId, relation: SymbolRelation, target: SymbolId) -> bool {
        if source.0 as usize >= self.nodes.len() || target.0 as usize >= self.nodes.len() {
            return false;
        }
        self.nodes[source.0 as usize]
            .write()
            .edges
            .push(SymbolEdge::new(relation, target));
        self.invalidate_cache();
        true
    }

    /// Verify that a path of symbol IDs forms a valid dependency chain.
    /// This is grounded's `verify_path()` repurposed — instead of checking
    /// edge contracts for energy conservation, we check symbol resolution
    /// chains (does this call chain type-check?).
    pub fn verify_path(&self, path: &[SymbolId]) -> Result<(), String> {
        if path.len() < 2 {
            return Ok(());
        }

        let source = path[0];
        let dest = path[path.len() - 1];

        if self.cache_hit(source, dest) {
            return Ok(());
        }

        let mut seen = HashSet::new();
        for (i, &node_id) in path.iter().enumerate() {
            if node_id == SymbolId::ZERO {
                return Err(format!("Dead node in path at position {}", i));
            }
            if !seen.insert(node_id) && node_id != SymbolId::ROOT {
                return Err(format!("Cycle detected at node {}", node_id.0));
            }

            if i + 1 < path.len() {
                let next_id = path[i + 1];
                let edge = match self.get(node_id) {
                    Some(n) => {
                        let guard = n.read();
                        guard.edges.iter().find(|e| e.target == next_id).cloned()
                    }
                    None => None,
                };

                match edge {
                    Some(e) => {
                        let contract = e.contract;
                        match contract {
                            DataContract::DefinitionUse {
                                output_type,
                                input_type,
                            } => {
                                if !Self::types_compatible(&output_type, &input_type) {
                                    return Err(format!(
                                        "Type mismatch: symbol {} produces {:?} but {} expects {:?}",
                                        node_id.0, output_type, next_id.0, input_type
                                    ));
                                }
                            }
                            DataContract::Call {
                                caller_scope,
                                callee_def,
                            } if (caller_scope != node_id || callee_def != next_id) => {
                                return Err(format!(
                                    "Call contract mismatch: {} -> {}",
                                    node_id.0, next_id.0
                                ));
                            }
                            _ => {}
                        }
                    }
                    None => {
                        return Err(format!(
                            "No edge from {} to {} in path",
                            node_id.0, next_id.0
                        ));
                    }
                }
            }
        }

        self.cache_store(source, dest, 0, 1.0);
        Ok(())
    }

    fn types_compatible(output: &TypeShape, input: &TypeShape) -> bool {
        match (output, input) {
            (_, TypeShape::Any) | (TypeShape::Any, _) => true,
            (TypeShape::Type(a), TypeShape::Type(b)) if a == b => true,
            _ => false,
        }
    }

    fn cache_hit(&self, source: SymbolId, dest: SymbolId) -> bool {
        let cache = self.path_cache.lock();
        cache
            .iter()
            .any(|s| matches!(s, Some(cp) if cp.source == source && cp.dest == dest))
    }

    fn cache_store(&self, source: SymbolId, dest: SymbolId, rel_mask: u64, weight: f64) {
        let mut cache = self.path_cache.lock();
        let entry = CachedPath {
            source,
            dest,
            relation_mask: rel_mask,
            weight,
        };
        for slot in cache.iter_mut() {
            if slot.is_none() {
                *slot = Some(entry);
                return;
            }
        }
        let idx = (source.0.wrapping_mul(2654435761) as usize) % cache.len();
        cache[idx] = Some(entry);
    }

    fn invalidate_cache(&self) {
        self.path_cache.lock().fill(None);
    }

    /// Find the shortest symbol dependency path from start to end.
    pub fn find_path(&self, start: SymbolId, end: SymbolId) -> Option<Vec<SymbolId>> {
        if start == end {
            return Some(vec![start]);
        }
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = std::collections::VecDeque::new();
        let mut parent = vec![SymbolId::ZERO; self.nodes.len()];

        visited[start.0 as usize] = true;
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            if let Some(node) = self.get(current) {
                for edge in &node.read().edges {
                    let next = edge.target;
                    let idx = next.0 as usize;
                    if idx < visited.len() && !visited[idx] {
                        visited[idx] = true;
                        parent[idx] = current;
                        if next == end {
                            let mut path = vec![end, current];
                            let mut p = current;
                            while p != start {
                                p = parent[p.0 as usize];
                                path.push(p);
                            }
                            path.reverse();
                            return Some(path);
                        }
                        queue.push_back(next);
                    }
                }
            }
        }
        None
    }
}
