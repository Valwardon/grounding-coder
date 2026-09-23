//! Tool provisioner: download what the engine needs, verify it, cache it.
//!
//! Like `pip` fetching a build backend: when an oracle needs a compiler
//! that isn't installed, the engine fetches pinned artifacts instead of
//! giving up. Every file is pinned by SHA-256 in the manifest below —
//! bytes that don't match are deleted, never executed. Network failure
//! degrades to the honest missing-tool error, never a guess.

use sha2::{Digest, Sha256};

/// One pinned artifact.
pub struct ToolFile {
    /// File name in the tool's cache dir.
    pub name: &'static str,
    /// Full download URL.
    pub url: &'static str,
    /// Expected SHA-256 hex of the exact bytes.
    pub sha256: &'static str,
}

/// A provisionable tool: a set of files plus how to reach them.
pub struct ToolSpec {
    pub name: &'static str,
    pub files: &'static [ToolFile],
}

/// Kotlin 2.0.20 compiler + runtime deps. Hashes pinned from Central's
/// `.sha1`-verified Gradle cache copies (same bytes Central serves).
pub const KOTLIN_FILES: &[ToolFile] = &[
    ToolFile {
        name: "kotlin-compiler-embeddable-2.0.20.jar",
        url: "https://repo1.maven.org/maven2/org/jetbrains/kotlin/kotlin-compiler-embeddable/2.0.20/kotlin-compiler-embeddable-2.0.20.jar",
        sha256: "a3604bf350c8bcce27102158bbb383fa4d1814bc486aaed59f278eada51ff6ad",
    },
    ToolFile {
        name: "kotlin-stdlib-2.0.20.jar",
        url: "https://repo1.maven.org/maven2/org/jetbrains/kotlin/kotlin-stdlib/2.0.20/kotlin-stdlib-2.0.20.jar",
        sha256: "fb169596659a518357c4b2c16f43dc75ab1c4980565ed4b4a317a050e5e39006",
    },
    ToolFile {
        name: "kotlinx-coroutines-core-jvm-1.6.4.jar",
        url: "https://repo1.maven.org/maven2/org/jetbrains/kotlinx/kotlinx-coroutines-core-jvm/1.6.4/kotlinx-coroutines-core-jvm-1.6.4.jar",
        sha256: "c24c8bb27bb320c4a93871501a7e5e0c61607638907b197aef675513d4c820be",
    },
    ToolFile {
        name: "annotations-13.0.jar",
        url: "https://repo1.maven.org/maven2/org/jetbrains/annotations/13.0/annotations-13.0.jar",
        sha256: "ace2a10dc8e2d5fd34925ecac03e4988b2c0f851650c94b8cef49ba1bd111478",
    },
    ToolFile {
        name: "trove4j-1.0.20200330.jar",
        url: "https://repo1.maven.org/maven2/org/jetbrains/intellij/deps/trove4j/1.0.20200330/trove4j-1.0.20200330.jar",
        sha256: "c5fd725bffab51846bf3c77db1383c60aaaebfe1b7fe2f00d23fe1b7df0a439d",
    },
];

pub const KOTLIN_TOOL: ToolSpec = ToolSpec {
    name: "kotlin",
    files: KOTLIN_FILES,
};

/// Cache root: `GROUNDING_TOOLS_DIR` wins (tests point it at tmp),
/// then the OS cache dir, then temp. Always writable-or-bust per file.
pub fn tools_dir() -> std::path::PathBuf {
    if let Ok(custom) = std::env::var("GROUNDING_TOOLS_DIR")
        && !custom.trim().is_empty()
    {
        return std::path::PathBuf::from(custom);
    }
    dirs::cache_dir()
        .or_else(dirs::data_dir)
        .map(|p| p.join("grounding-coder").join("tools"))
        .unwrap_or_else(|| std::env::temp_dir().join("grounding-coder-tools"))
}

/// Ensure every file of a tool exists with matching hash, downloading
/// what is missing or corrupt. Returns the tool's cache dir.
pub async fn ensure_tool(spec: &ToolSpec) -> Result<std::path::PathBuf, String> {
    let dir = tools_dir().join(spec.name);
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create tools dir: {}", e))?;
    for f in spec.files {
        let dest = dir.join(f.name);
        if let Ok(bytes) = std::fs::read(&dest)
            && hash_matches(&bytes, f.sha256)
        {
            continue;
        }
        let bytes = crate::http::get_bytes(f.url)
            .await
            .map_err(|e| format!("Download {} failed: {}", f.name, e))?;
        if !hash_matches(&bytes, f.sha256) {
            let _ = std::fs::remove_file(&dest);
            return Err(format!(
                "Checksum mismatch for {} — deleted, refusing to run it",
                f.name
            ));
        }
        std::fs::write(&dest, &bytes).map_err(|e| format!("Cannot store {}: {}", f.name, e))?;
    }
    Ok(dir)
}

fn hash_matches(bytes: &[u8], hex: &str) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize()).eq_ignore_ascii_case(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_hashes_are_well_formed() {
        for f in KOTLIN_FILES {
            assert_eq!(f.sha256.len(), 64, "bad sha for {}", f.name);
            assert!(
                f.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "non-hex in {}",
                f.name
            );
            assert!(
                f.url.starts_with("https://repo1.maven.org/maven2"),
                "off-manifest url {}",
                f.url
            );
            assert!(f.name.ends_with(".jar"));
        }
    }

    #[test]
    fn tools_dir_honors_override() {
        // SAFETY: single-threaded test; no other test touches this var.
        unsafe {
            std::env::set_var("GROUNDING_TOOLS_DIR", "/tmp/gc-tools-test");
        }
        assert_eq!(tools_dir(), std::path::PathBuf::from("/tmp/gc-tools-test"));
        // SAFETY: same as above.
        unsafe {
            std::env::remove_var("GROUNDING_TOOLS_DIR");
        }
    }
}
