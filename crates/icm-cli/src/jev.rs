//! Optional remote reranking with TypeSafe's Jev decision model.
//!
//! Ported from memsearch's `jev_reranker.py` (zilliztech/memsearch), itself
//! adapted from https://docs.typesafe.ai/cookbooks/rerank_typesafe. Enabled by
//! `[recall] reranker = "jev:jev-latest"`. Calling it sends the query and the
//! candidate memory text to api.typesafe.ai using `TYPESAFE_API_KEY`.
//!
//! Every candidate is scored in one batched request, each as an independent
//! question. Errors surface to the caller: a failed request never silently
//! turns into an unreranked result.

use std::time::Duration;

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};

const ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";
const TIMEOUT: Duration = Duration::from_secs(60);

const INSTRUCTIONS: &str = "The query asks about information recorded in project memory. \
Could the candidate passage be a source for the answer — does it state the specific fact, \
decision, procedure, or event the query asks about?";
const CRITERIA_TRUE: &str = "The candidate passage states or establishes the specific \
information needed to answer the query, or a necessary supporting fact for a query requiring \
multiple passages.";
const CRITERIA_FALSE: &str = "The candidate passage is merely on a similar topic or project; \
it does not supply the specific information the query requires.";

/// Returns the Jev model name if `reranker` selects Jev (`jev:<model>`).
pub fn model_from_config(reranker: &str) -> Option<&str> {
    reranker
        .strip_prefix("jev:")
        .map(str::trim)
        .filter(|m| !m.is_empty())
}

/// Build the batched Noul request, preserving every candidate's full text.
pub fn build_request(model: &str, query: &str, documents: &[&str]) -> Value {
    let questions: Map<String, Value> = documents
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            (
                format!("d{i}"),
                json!({
                    "type": "noul",
                    "instructions": format!("{INSTRUCTIONS}\nCandidate passage:\n{doc}"),
                    "criteria": { "true": CRITERIA_TRUE, "false": CRITERIA_FALSE },
                }),
            )
        })
        .collect();
    json!({
        "model": model,
        "state": { "query_excerpt": query },
        "questions": questions,
    })
}

/// Validate exact answer coverage and finite probabilities in [0, 1].
pub fn scores(response: &Value, count: usize) -> Result<Vec<f32>> {
    let Some(answers) = response.get("answers").and_then(Value::as_object) else {
        bail!("Jev returned missing or unexpected answers");
    };
    if answers.len() != count || (0..count).any(|i| !answers.contains_key(&format!("d{i}"))) {
        bail!("Jev returned missing or unexpected answers");
    }
    (0..count)
        .map(|i| {
            let answer = &answers[&format!("d{i}")];
            if answer.get("type").and_then(Value::as_str) != Some("noul") {
                bail!("Jev returned an unexpected answer type");
            }
            // `as_f64` rejects bools and strings; JSON can't carry NaN/inf.
            let Some(score) = answer.get("noul").and_then(Value::as_f64) else {
                bail!("Jev returned a non-finite or non-numeric score");
            };
            if !(0.0..=1.0).contains(&score) {
                bail!("Jev score must be between zero and one");
            }
            Ok(score as f32)
        })
        .collect()
}

/// Score `documents` against `query`. Returns one score per document, in order.
pub fn score(model: &str, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }
    let api_key = std::env::var("TYPESAFE_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        bail!("Set TYPESAFE_API_KEY to enable Jev reranking");
    }
    // Override exists for integration tests (tests/jev_integration.rs).
    let endpoint = std::env::var("ICM_JEV_ENDPOINT").unwrap_or_else(|_| ENDPOINT.to_string());
    let response = ureq::post(&endpoint)
        .timeout(TIMEOUT)
        .set("authorization", &format!("Bearer {api_key}"))
        .send_json(build_request(model, query, documents));
    // Never echo the response body or key: the body may quote private memory text.
    let body: Value = match response {
        Ok(resp) => resp.into_json()?,
        Err(ureq::Error::Status(code, _)) => bail!("Jev reranking failed (HTTP {code})"),
        Err(_) => bail!("Jev reranking request failed or timed out"),
    };
    scores(&body, documents.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(scores: &[Value]) -> Value {
        let answers: Map<String, Value> = scores
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("d{i}"), json!({ "type": "noul", "noul": s })))
            .collect();
        json!({ "model": "jev-1.13.0", "answers": answers })
    }

    #[test]
    fn request_preserves_full_content() {
        let long = "long memory ".repeat(2000);
        let req = build_request("jev-latest", "question", &[&long, "second"]);
        assert_eq!(req["model"], "jev-latest");
        assert_eq!(req["state"]["query_excerpt"], "question");
        assert_eq!(req["questions"].as_object().unwrap().len(), 2);
        assert!(req["questions"]["d0"]["instructions"]
            .as_str()
            .unwrap()
            .ends_with(&long));
        let criteria = req["questions"]["d1"]["criteria"].as_object().unwrap();
        assert!(criteria.contains_key("true") && criteria.contains_key("false"));
    }

    #[test]
    fn valid_scores_parse_in_order() {
        let s = scores(&response(&[json!(0.2), json!(0.8), json!(1)]), 3).unwrap();
        assert_eq!(s, vec![0.2, 0.8, 1.0]);
    }

    #[test]
    fn invalid_scores_fail() {
        for bad in [
            Value::Null,
            json!(true),
            json!("0.5"),
            json!(-0.1),
            json!(1.1),
        ] {
            assert!(
                scores(&response(std::slice::from_ref(&bad)), 1).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn missing_extra_or_wrong_type_answers_fail() {
        assert!(scores(&response(&[]), 1).is_err());
        assert!(scores(&response(&[json!(0.1), json!(0.2)]), 1).is_err());
        let mut wrong = response(&[json!(0.5)]);
        wrong["answers"]["d0"]["type"] = json!("choice");
        assert!(scores(&wrong, 1).is_err());
    }

    #[test]
    fn empty_candidates_need_no_key() {
        assert!(score("jev-latest", "q", &[]).unwrap().is_empty());
    }

    #[test]
    fn config_selects_jev_only_with_prefix() {
        assert_eq!(model_from_config("jev:jev-latest"), Some("jev-latest"));
        assert_eq!(model_from_config("jev:"), None);
        assert_eq!(model_from_config(""), None);
        assert_eq!(model_from_config("cross-encoder/foo"), None);
    }
}
