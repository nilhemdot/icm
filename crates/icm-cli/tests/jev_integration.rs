//! End-to-end tests for Jev reranking in `icm recall` (`[recall] reranker`).
//!
//! Runs the compiled `icm` binary against a fake Jev server on localhost
//! (`ICM_JEV_ENDPOINT`), with an isolated DB, HOME and XDG_CONFIG_HOME.
//! Covers: reorder + truncate by Jev score, 3x over-fetch, auth header,
//! fail-loud errors that leak neither key nor response body, and that the
//! per-prompt hook never calls Jev.
//!
//! Linux-only for the same sqlite-vec child-process reason as
//! `http_api_integration.rs`.
#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

const ICM: &str = env!("CARGO_BIN_EXE_icm");
const KEY: &str = "test-key-do-not-leak";

/// What the fake server saw in one request.
struct Seen {
    auth: String,
    body: Value,
}

/// Serve exactly one request, replying with `status` and a body built
/// from the parsed request. Returns the endpoint URL and a receiver for
/// what the server saw.
fn fake_jev(status: u16, reply: fn(&Value) -> String) -> (String, mpsc::Receiver<Seen>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/v1/systemone", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let Ok((stream, _)) = listener.accept() else {
            return;
        };
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let (mut auth, mut len) = (String::new(), 0usize);
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.trim().is_empty() {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                len = v.trim().parse().unwrap();
            } else if lower.starts_with("authorization:") {
                auth = line["authorization:".len()..].trim().to_string();
            }
        }
        let mut buf = vec![0; len];
        reader.read_exact(&mut buf).unwrap();
        let body: Value = serde_json::from_slice(&buf).unwrap();
        let out = reply(&body);
        let mut stream = stream;
        write!(
            stream,
            "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{out}",
            out.len()
        )
        .unwrap();
        let _ = tx.send(Seen { auth, body });
    });
    (url, rx)
}

/// Score 0.95 for the candidate mentioning WINNER, 0.1 for the rest.
fn score_winner(req: &Value) -> String {
    let answers: serde_json::Map<String, Value> = req["questions"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, q)| {
            let hit = q["instructions"].as_str().unwrap().contains("WINNER");
            let s = if hit { 0.95 } else { 0.1 };
            (id.clone(), json!({ "type": "noul", "noul": s }))
        })
        .collect();
    json!({ "model": "jev-test", "answers": answers }).to_string()
}

fn secret_body(_: &Value) -> String {
    r#"{"error":"SECRET_RESPONSE_BODY"}"#.to_string()
}

/// Isolated env: temp DB/HOME/XDG config with Jev enabled.
struct Env {
    dir: tempfile::TempDir,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("xdg/icm");
        std::fs::create_dir_all(&cfg).unwrap();
        std::fs::write(
            cfg.join("config.toml"),
            "[recall]\nreranker = \"jev:jev-test\"\n",
        )
        .unwrap();
        let env = Self { dir };
        // FTS order is unrelated to Jev's: WINNER is stored in the middle.
        for c in [
            "widget alpha note",
            "widget beta note",
            "widget WINNER private-memory-text",
            "widget gamma note",
            "widget delta note",
        ] {
            let out = env.icm(&["store", "-t", "test", "-c", c], None, None);
            assert!(out.status.success(), "store failed: {out:?}");
        }
        env
    }

    fn icm(&self, args: &[&str], endpoint: Option<&str>, stdin: Option<&str>) -> Output {
        let p = self.dir.path();
        let mut cmd = Command::new(ICM);
        cmd.arg("--no-embeddings")
            .arg("--db")
            .arg(p.join("m.db"))
            .args(args)
            .env("HOME", p)
            .env("XDG_CONFIG_HOME", p.join("xdg"))
            .env("XDG_DATA_HOME", p.join("data"))
            .env("TYPESAFE_API_KEY", KEY)
            .env(
                "ICM_JEV_ENDPOINT",
                endpoint.unwrap_or("http://127.0.0.1:9/unused"),
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().expect("spawn icm");
        if let Some(input) = stdin {
            child
                .stdin
                .take()
                .unwrap()
                .write_all(input.as_bytes())
                .unwrap();
        }
        drop(child.stdin.take());
        child.wait_with_output().unwrap()
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }
}

fn recall_json(env: &Env, endpoint: &str, limit: &str) -> Output {
    env.icm(
        &["recall", "widget", "-l", limit, "--format", "json"],
        Some(endpoint),
        None,
    )
}

#[test]
fn recall_reorders_and_truncates_by_jev_score() {
    let env = Env::new();
    let (url, seen) = fake_jev(200, score_winner);
    let out = recall_json(&env, &url, "2");
    assert!(out.status.success(), "recall failed: {out:?}");

    let results: Value = serde_json::from_slice(&out.stdout).unwrap();
    let results = results.as_array().unwrap();
    assert_eq!(results.len(), 2, "truncated to limit: {results:?}");
    assert!(results[0].to_string().contains("WINNER"), "{results:?}");

    let seen = seen.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(seen.auth, format!("Bearer {KEY}"));
    assert_eq!(seen.body["model"], "jev-test");
    assert_eq!(seen.body["state"]["query_excerpt"], "widget");
    // limit 2 -> pool of 6, so all 5 stored memories are candidates.
    assert_eq!(seen.body["questions"].as_object().unwrap().len(), 5);
}

#[test]
fn http_error_fails_without_leaking_key_or_body() {
    let env = Env::new();
    let (url, _seen) = fake_jev(500, secret_body);
    let out = recall_json(&env, &url, "2");
    assert!(!out.status.success(), "must not fall back silently");
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(all.contains("HTTP 500"), "{all}");
    for leak in [KEY, "SECRET_RESPONSE_BODY", "private-memory-text"] {
        assert!(!all.contains(leak), "leaked {leak}: {all}");
    }
}

#[test]
fn missing_key_fails_explicitly() {
    let env = Env::new();
    let out = Command::new(ICM)
        .args(["--no-embeddings", "--db"])
        .arg(env.path().join("m.db"))
        .args(["recall", "widget"])
        .env("HOME", env.path())
        .env("XDG_CONFIG_HOME", env.path().join("xdg"))
        .env_remove("TYPESAFE_API_KEY")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("TYPESAFE_API_KEY"));
}

#[test]
fn prompt_hook_never_calls_jev() {
    let env = Env::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/v1/systemone", listener.local_addr().unwrap());

    let input = json!({ "prompt": "widget", "cwd": env.path() }).to_string();
    let out = env.icm(&["hook", "prompt"], Some(&url), Some(&input));
    assert!(out.status.success(), "hook failed: {out:?}");
    assert!(
        listener.accept().is_err(),
        "per-prompt hook must not make network calls"
    );
}
