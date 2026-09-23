//! Minimal HTTPS client with zero platform trust-store dependency.
//!
//! Why this exists: `reqwest`'s default TLS stack verifies against the OS
//! trust store via `rustls-platform-verifier`, whose Android backend
//! `expect()`s JNI initialization that a plain app never performs —
//! aborting the process with no message. This client verifies against the
//! bundled Mozilla roots (`with_webpki_roots`) instead, so TLS works
//! identically on desktop and on-device with no JNI, no init ritual.
//!
//! IO runs on a small dedicated tokio runtime (spawned, never blocked-on
//! from async contexts), so the UI executor thread is never stalled.

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::TokioExecutor;
use std::sync::LazyLock;
use std::time::Duration;

type HttpsClient = Client<hyper_rustls::HttpsConnector<HttpConnector>, Full<Bytes>>;

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("grounding-http")
        .enable_all()
        .build()
        .expect("grounding http runtime must build")
});

fn client() -> HttpsClient {
    let https = HttpsConnectorBuilder::new()
        .with_webpki_roots()
        .https_or_http()
        .enable_http1()
        .build();
    Client::builder(TokioExecutor::new()).build(https)
}

async fn request(
    req: hyper::Request<Full<Bytes>>,
    timeout: Duration,
) -> Result<(u16, Vec<u8>), String> {
    let work = async move {
        let resp = client()
            .request(req)
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;
        let status = resp.status().as_u16();
        let body = resp
            .into_body()
            .collect()
            .await
            .map_err(|e| format!("Body error: {}", e))?
            .to_bytes()
            .to_vec();
        Ok::<_, String>((status, body))
    };
    // Spawn onto the dedicated runtime: never blocks the caller's executor
    // (an ANR-killed UI thread on Android would be silent death).
    match tokio::time::timeout(timeout, RUNTIME.spawn(work)).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => Err(format!("HTTP task failed: {}", e)),
        Err(_) => {
            Err("LLM timed out queuing (free-tier models are slow) — try Send again.".to_string())
        }
    }
}

/// POST JSON, expect JSON back. Non-2xx is an error with the body attached.
pub async fn post_json(
    url: &str,
    bearer: Option<&str>,
    body: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    send_json("POST", url, bearer, body, None).await
}

/// PUT JSON, expect JSON back. Non-2xx is an error with the body attached.
pub async fn put_json(
    url: &str,
    bearer: Option<&str>,
    body: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    send_json(
        "PUT",
        url,
        bearer,
        body,
        Some("application/vnd.github+json"),
    )
    .await
}

async fn send_json(
    method: &str,
    url: &str,
    bearer: Option<&str>,
    body: &serde_json::Value,
    accept: Option<&str>,
) -> Result<serde_json::Value, String> {
    let bytes = serde_json::to_vec(body).map_err(|e| format!("Serialize error: {}", e))?;
    let mut builder = hyper::Request::builder()
        .method(method)
        .uri(url)
        .header("content-type", "application/json")
        .header("user-agent", "grounding-coder/0.5");
    if let Some(a) = accept {
        builder = builder.header("accept", a);
    }
    if let Some(token) = bearer
        && !token.is_empty()
    {
        builder = builder.header("authorization", format!("Bearer {}", token));
    }
    let req = builder
        .body(Full::new(Bytes::from(bytes)))
        .map_err(|e| format!("Request build error: {}", e))?;
    // Free-tier reasoning models can queue for minutes: bounded, but
    // generous. A timeout still surfaces as text, never a hang.
    let (status, raw) = request(req, Duration::from_secs(300)).await?;
    if !(200..300).contains(&status) {
        return Err(format!(
            "HTTP {}: {}",
            status,
            String::from_utf8_lossy(&raw[..raw.len().min(500)])
        ));
    }
    serde_json::from_slice(&raw).map_err(|e| format!("Parse error: {}", e))
}

/// GET raw bytes. Non-2xx is an error. Used by the tool provisioner —
/// compilers arrive as bytes, never as text.
pub async fn get_bytes(url: &str) -> Result<Vec<u8>, String> {
    let req = hyper::Request::builder()
        .method("GET")
        .uri(url)
        .header("user-agent", "grounding-coder-provision/0.1")
        .body(Full::new(Bytes::new()))
        .map_err(|e| format!("Request build error: {}", e))?;
    // Toolchains are tens of MB; bound generously, still never a hang.
    let (status, raw) = request(req, Duration::from_secs(600)).await?;
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {}", status));
    }
    Ok(raw)
}

/// GET JSON with optional bearer auth and Accept header. Non-2xx errors.
pub async fn get_json(
    url: &str,
    bearer: Option<&str>,
    accept: Option<&str>,
) -> Result<serde_json::Value, String> {
    let mut builder = hyper::Request::builder()
        .method("GET")
        .uri(url)
        .header("user-agent", "grounding-coder-oracle/0.1");
    if let Some(a) = accept {
        builder = builder.header("accept", a);
    }
    if let Some(token) = bearer
        && !token.is_empty()
    {
        builder = builder.header("authorization", format!("Bearer {}", token));
    }
    let req = builder
        .body(Full::new(Bytes::new()))
        .map_err(|e| format!("Request build error: {}", e))?;
    let (status, raw) = request(req, Duration::from_secs(30)).await?;
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {}", status));
    }
    serde_json::from_slice(&raw).map_err(|e| format!("Parse error: {}", e))
}

/// GET returning (status, text body).
pub async fn get_text(url: &str) -> Result<(u16, String), String> {
    let req = hyper::Request::builder()
        .method("GET")
        .uri(url)
        .header("user-agent", "grounding-coder-research/0.1")
        .body(Full::new(Bytes::new()))
        .map_err(|e| format!("Request build error: {}", e))?;
    let (status, raw) = request(req, Duration::from_secs(30)).await?;
    Ok((status, String::from_utf8_lossy(&raw).to_string()))
}
