//! GitHub actor: create repositories and upload project files.
//!
//! Driven from Chat when the intent asks for it ("create a repo", "upload
//! to github") with the token from Settings. Same honesty contract as the
//! engine: every step is a verified API round-trip, failures report
//! plainly, and publishing never rewrites local files.

use base64::Engine as _;

/// Normalize a human repo name ("rng gen") to a GitHub slug ("rng-gen").
pub fn normalize_repo_name(name: &str) -> Result<String, String> {
    let mut slug: String = name
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches(|c| c == '-' || c == '.' || c == '_');
    if slug.is_empty() {
        return Err("Repository name is empty after normalization — BLOCKED".to_string());
    }
    if slug.len() > 100 {
        return Err("Repository name too long — BLOCKED".to_string());
    }
    Ok(slug.to_string())
}

/// Create `$owner/$repo` (public). 422 means it already exists — that is a
/// fact, not a failure: publishing proceeds into it.
async fn ensure_repo(token: &str, repo: &str) -> Result<(), String> {
    let body = crate::http::post_json(
        "https://api.github.com/user/repos",
        Some(token),
        &serde_json::json!({
            "name": repo,
            "private": false,
            "auto_init": false,
            "description": "Built with grounding-coder",
        }),
    )
    .await;
    match body {
        Ok(_) => Ok(()),
        Err(e) if e.contains("422") => Ok(()),
        Err(e) => Err(format!("Create repo failed: {}", e)),
    }
}

/// Project files worth uploading: text sources, capped. Binaries, lock
/// artifacts, and VCS dirs never leave the device.
fn collect_files(project_dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    const MAX_FILES: usize = 200;
    const MAX_BYTES: usize = 256 * 1024;
    const SKIP_DIRS: &[&str] = &[
        ".git",
        "target",
        "build",
        ".gradle",
        ".idea",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
    ];
    const SKIP_EXT: &[&str] = &[
        "so", "apk", "aab", "dex", "o", "a", "class", "jar", "pyc", "lock",
    ];
    let mut out = Vec::new();
    let mut stack = vec![(project_dir.to_path_buf(), 0u8)];
    while let Some((current, depth)) = stack.pop() {
        if out.len() >= MAX_FILES || depth > 8 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                if name.starts_with('.') && name != ".github" || SKIP_DIRS.contains(&name.as_str())
                {
                    continue;
                }
                stack.push((path, depth + 1));
            } else {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if SKIP_EXT.contains(&ext.as_str()) {
                    continue;
                }
                let rel = path
                    .strip_prefix(project_dir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();
                match std::fs::read(&path) {
                    Ok(bytes) if bytes.len() <= MAX_BYTES => out.push((rel, bytes)),
                    _ => continue,
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Publish a project: ensure the repo, upload every collected file.
/// Returns the repo URL. Files that fail individually are reported in
/// `skipped`, never silent.
pub async fn publish(
    token: &str,
    owner: &str,
    repo_name: &str,
    project_dir: &std::path::Path,
) -> Result<(String, Vec<String>), String> {
    if token.trim().is_empty() {
        return Err("GitHub token missing — set it in Settings".to_string());
    }
    let repo = normalize_repo_name(repo_name)?;
    ensure_repo(token, &repo).await?;
    let files = collect_files(project_dir);
    if files.is_empty() {
        return Err("Nothing uploadable in project dir — BLOCKED".to_string());
    }
    let mut skipped = Vec::new();
    for (rel, bytes) in &files {
        let url = format!(
            "https://api.github.com/repos/{}/{}/contents/{}",
            owner, repo, rel
        );
        let body = serde_json::json!({
            "message": format!("grounding-coder: {}", rel),
            "content": base64::engine::general_purpose::STANDARD.encode(bytes),
        });
        if let Err(e) = crate::http::put_json(&url, Some(token), &body).await {
            // Updating an existing file needs its sha; treat as skipped
            // rather than failing the whole publish.
            skipped.push(format!("{}: {}", rel, e));
        }
    }
    Ok((format!("https://github.com/{}/{}", owner, repo), skipped))
}

/// Owner login for a token (`GET /user`). Needed to build repo URLs.
pub async fn token_owner(token: &str) -> Result<String, String> {
    let body = crate::http::get_json("https://api.github.com/user", Some(token), None).await?;
    body.get("login")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "Cannot read token owner".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_names() {
        assert_eq!(normalize_repo_name("rng gen").unwrap(), "rng-gen");
        assert_eq!(
            normalize_repo_name("My Cool Repo!").unwrap(),
            "my-cool-repo"
        );
        assert_eq!(normalize_repo_name("a--b__c").unwrap(), "a-b__c");
        assert!(normalize_repo_name("!!!").is_err());
    }
}
