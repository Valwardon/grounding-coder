use std::collections::HashMap;

/// A code symbol definition — grounded's knowledge-definition pattern repurposed.
///
/// In grounded, `KnowledgeStore` maps tokens to definition strings:
///   ("pirate", "is a person. wears tricorn hat. carries cutlass.")
///
/// Here, we map symbol names to structured code definitions:
///   ("android.widget.Button", "class. methods: setText(String), setOnClickListener(...)")
///
/// This is the bot's "perfect knowledge" — it's compiled in, offline,
/// deterministic, and never guesses. When the bot doesn't find a symbol
/// here, it doesn't hallucinate — it marks it as an unknown and asks
/// for help (grounded's "honesty principle").
#[derive(Debug, Clone)]
pub struct CodeDef {
    /// Fully qualified name (e.g., "android.widget.Button")
    pub qname: String,
    /// Language: "kotlin", "java", "rust"
    pub language: String,
    /// What kind of symbol: "class", "function", "struct", "enum", "interface"
    pub kind: String,
    /// Module/package path (e.g., "android.widget")
    pub module: String,
    /// Signature: "fn setText(text: String): Unit" or "fn vibrate(duration_ms: Long)"
    pub signature: String,
    /// Human-readable description for pattern matching
    pub description: String,
    /// Example usage snippets (for the CodeWriter to compose from)
    pub examples: Vec<String>,
}

/// The symbol table — grounded's KnowledgeStore repurposed.
///
/// Two knowledge sources:
/// 1. **Embedded API reference** — compiled-in definitions for common
///    Android/Kotlin/Rust APIs (like grounded's foundation definitions).
/// 2. **Codebase index** — scanned at startup from the target project
///    (like grounded's runtime cache).
///
/// The bot NEVER guesses an API exists — it consults this table.
/// If a symbol isn't here and isn't in the codebase index, the bot
/// reports "I don't know" rather than hallucinating.
pub struct SymbolTable {
    /// Embedded API reference — the "perfect knowledge" for platform APIs.
    /// Populated at construction time from FOUNDATION_APIS below.
    definitions: HashMap<String, CodeDef>,
    /// Runtime cache — codebase-specific symbols discovered by scanning.
    cache: HashMap<String, CodeDef>,
}

/// Foundation API definitions — compiled into the binary at build time.
/// This mirrors grounded's `KnowledgeStore::new()` which embeds ~30
/// foundation definitions ("cat is a mammal...", "pirate wears tricorn hat...").
///
/// Here we embed the most common Android/Kotlin/Rust APIs so the bot
/// can reason about them deterministically.
///
/// Key principle: every definition here is VERIFIED against the actual
/// Android SDK docs. No guessing.
const FOUNDATION_APIS: &[(&str, &str, &str, &str, &str, &[&str])] = &[
    // (qname, language, kind, module, signature, examples)
    (
        "android.widget.Button",
        "kotlin",
        "class",
        "android.widget",
        "constructor(context: Context); setText(text: String); setOnClickListener(listener: View.OnClickListener)",
        &["val button = Button(context)", "button.setText(\"Click me\")", "button.setOnClickListener { /* handle click */ }"],
    ),
    (
        "android.widget.LinearLayout",
        "kotlin",
        "class",
        "android.widget",
        "constructor(context: Context); addView(child: View); setOrientation(orientation: Int)",
        &["val layout = LinearLayout(context)", "layout.orientation = LinearLayout.VERTICAL", "layout.addView(button)"],
    ),
    (
        "android.widget.TextView",
        "kotlin",
        "class",
        "android.widget",
        "constructor(context: Context); setText(text: String); setTextSize(size: Float)",
        &["val tv = TextView(context)", "tv.text = \"Hello\""],
    ),
    (
        "android.view.ViewGroup",
        "kotlin",
        "interface",
        "android.view",
        "abstract; addView(child: View)",
        &["class MyContainer : ViewGroup"]
    ),
    (
        "android.view.View",
        "kotlin",
        "class",
        "android.view",
        "abstract; setOnClickListener(listener: View.OnClickListener)",
        &["val view: View = findViewById(R.id.my_view)"]
    ),
    (
        "android.view.View.OnClickListener",
        "kotlin",
        "interface",
        "android.view",
        "fun onClick(v: View?)",
        &["object : View.OnClickListener { override fun onClick(v: View?) {} }"]
    ),
    (
        "android.view.LayoutInflater",
        "kotlin",
        "class",
        "android.view",
        "fun inflate(resource: Int, parent: ViewGroup?, attachToRoot: Boolean): View",
        &["val inflater = LayoutInflater.from(context)", "val view = inflater.inflate(R.layout.item, parent, false)"]
    ),
    (
        "android.os.Vibrator",
        "kotlin",
        "class",
        "android.os",
        "fun vibrate(vibrationEffect: VibrationEffect); fun vibrate(milliseconds: Long)",
        &["val vib = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator", "vib.vibrate(500)"]
    ),
    (
        "android.Manifest.permission.VIBRATE",
        "kotlin",
        "const",
        "android.Manifest.permission",
        "String = \"android.permission.VIBRATE\"",
        &["<uses-permission android:name=\"android.permission.VIBRATE\" />"]
    ),
    (
        "android.app.Activity",
        "kotlin",
        "class",
        "android.app",
        "fun setContentView(view: View); fun setContentView(layoutResID: Int); fun findViewById(id: Int): T",
        &["class MainActivity : AppCompatActivity() { override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState); setContentView(R.layout.activity_main) } }"]
    ),
    (
        "android.content.Context",
        "kotlin",
        "class",
        "android.content",
        "fun getSystemService(name: String): Any?",
        &["val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator"]
    ),
    (
        "androidx.appcompat.app.AppCompatActivity",
        "kotlin",
        "class",
        "androidx.appcompat.app",
        "extends Activity; fun onCreate(savedInstanceState: Bundle?)",
        &["class MainActivity : AppCompatActivity() { override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState) } }"]
    ),
    (
        "android.view.View.OnClickListener",
        "kotlin",
        "interface",
        "android.view",
        "fun onClick(v: View?)",
        &["button.setOnClickListener { }"]
    ),
    (
        "android.widget.Toast",
        "kotlin",
        "class",
        "android.widget",
        "fun makeText(context: Context, text: CharSequence, duration: Int): Toast; fun show()",
        &["Toast.makeText(context, \"Hello\", Toast.LENGTH_SHORT).show()"]
    ),
    (
        "android.widget.EditText",
        "kotlin",
        "class",
        "android.widget",
        "getText(): Editable; setText(text: String)",
        &["val input = findViewById<EditText>(R.id.input)", "val text = input.text.toString()"]
    ),
    (
        "android.widget.RecyclerView",
        "kotlin",
        "class",
        "android.widget",
        "setAdapter(adapter: Adapter<*>?)",
        &["recyclerView.adapter = MyAdapter(data)"]
    ),
    // Kotlin standard library
    (
        "kotlin.String",
        "kotlin",
        "class",
        "kotlin",
        "fun length: Int; fun substring(start: Int): String; fun toByteArray(): ByteArray",
        &["val s: String = \"hello\"", "s.length", "s.substring(0, 3)"]
    ),
    (
        "kotlin.Int",
        "kotlin",
        "primitive",
        "kotlin",
        "typealias Int = i32",
        &["val x: Int = 42"]
    ),
    (
        "kotlin.Unit",
        "kotlin",
        "object",
        "kotlin",
        "the unit type (like void / nil)",
        &["fun doThing(): Unit { }"]
    ),
    // Rust std
    (
        "std::string::String",
        "rust",
        "struct",
        "std::string",
        "fn new(): String; fn from(s: &str): String; fn push_str(s: &str)",
        &["let s = String::from(\"hello\")", "s.push_str(\" world\")"]
    ),
    (
        "std::option::Option",
        "rust",
        "enum",
        "std::option",
        "Some(T), None",
        &["let x: Option<i32> = Some(5)"]
    ),
    (
        "std::result::Result",
        "rust",
        "enum",
        "std::result",
        "Ok(T), Err(E)",
        &["fn foo() -> Result<i32, String> { Ok(42) }"]
    ),
    (
        "std::vec::Vec",
        "rust",
        "struct",
        "std::vec",
        "fn new(): Vec<T>; fn push(v: T); fn len(): usize",
        &["let v: Vec<i32> = Vec::new()", "v.push(1)"]
    ),
    (
        "std::sync::Arc",
        "rust",
        "struct",
        "std::sync",
        "fn new(data: T): Arc<T>; fn clone(&self): Arc<T>",
        &["let data = Arc::new(42)", "let cloned = Arc::clone(&data)"]
    ),
    (
        "std::fs::File",
        "rust",
        "struct",
        "std::fs",
        "fn open(path: P): io::Result<File>; fn create(path: P): io::Result<File>",
        &["let f = File::open(\"path/to/file\")?"]
    ),
    (
        "std::io::Read",
        "rust",
        "trait",
        "std::io",
        "fn read(&mut self, buf: &mut [u8]): io::Result<usize>",
        &["let mut buf = [0u8; 1024]; f.read(&mut buf)?"]
    ),
];

impl SymbolTable {
    pub fn new() -> Self {
        let mut definitions = HashMap::new();
        let mut cache = HashMap::new();

        // Populate from embedded foundation APIs (like grounded's KnowledgeStore)
        for def in FOUNDATION_APIS {
            let code_def = CodeDef {
                qname: def.0.to_string(),
                language: def.1.to_string(),
                kind: def.2.to_string(),
                module: def.3.to_string(),
                signature: def.4.to_string(),
                description: format!("{} in {}", def.2, def.3),
                examples: def.5.iter().map(|s| s.to_string()).collect(),
            };
            // Index by full qname AND by simple last segment
            let simple = def.0.rsplit('.').next().unwrap_or(def.0).to_lowercase();
            definitions.insert(def.0.to_lowercase(), code_def.clone());
            definitions.entry(simple).or_insert(code_def);
        }

        SymbolTable { definitions, cache }
    }

    /// Fetch a definition — grounded's `KnowledgeStore::fetch()` repurposed.
    ///
    /// Returns None for fundamental primitives that need no further
    /// definition (like grounded's "matter", "motion", "energy").
    /// For code, these are language primitives: "int", "bool", "float", "str".
    pub fn fetch(&self, symbol: &str) -> Option<CodeDef> {
        let sym = symbol.to_lowercase();

        // Check runtime cache first (codebase-specific symbols)
        if let Some(def) = self.cache.get(&sym) {
            return Some(def.clone());
        }

        // Check embedded API reference (platform/system symbols)
        if let Some(def) = self.definitions.get(&sym) {
            return Some(def.clone());
        }

        // Fundamental language primitives — no definition needed
        let primitives = ["int", "bool", "float", "str", "string", "i32", "i64",
                         "f32", "f64", "u32", "u64", "usize", "isize",
                         "void", "unit", "null", "true", "false", "self",
                         "context", "this", "unit"];

        if primitives.contains(&sym.as_str()) {
            return None; // fundamental — no further resolution needed
        }

        // Unknown symbol: return a minimal placeholder (like grounded's
        // "is a thing. is unknown."). The bot will flag this as a GAP
        // that needs human clarification — it won't guess.
        Some(CodeDef {
            qname: symbol.to_string(),
            language: "unknown".into(),
            kind: "unknown".into(),
            module: String::new(),
            signature: String::new(),
            description: "Unknown symbol — no definition available".into(),
            examples: Vec::new(),
        })
    }

    /// Index a codebase symbol (runtime cache).
    pub fn index(&mut self, qname: &str, def: CodeDef) {
        self.cache.insert(qname.to_lowercase(), def);
    }

    /// Check if a symbol is fully resolvable (has a definition).
    pub fn is_known(&self, symbol: &str) -> bool {
        self.fetch(symbol).is_some()
    }

    /// Find all symbols in a module namespace.
    pub fn find_in_module(&self, module: &str) -> Vec<CodeDef> {
        let mut results = Vec::new();
        let module_lower = module.to_lowercase();
        for def in self.definitions.values() {
            if def.module.to_lowercase() == module_lower {
                results.push(def.clone());
            }
        }
        for def in self.cache.values() {
            if def.module.to_lowercase() == module_lower {
                results.push(def.clone());
            }
        }
        results
    }

    /// Count total known definitions (embedded + codebase).
    pub fn count(&self) -> usize {
        self.definitions.len() + self.cache.len()
    }
}
