from genai.evaluation.stage_4c_orchestrator import run_stage_4c
from genai.llm.llm_prompt_builder import build_llm_prompt
from genai.controller.system_controller import decide_mode
from genai.guardrails.output_guardrails import apply_output_guardrails
from genai.llm.llm_client import GroqLLMClient
from genai.llm.llm_postprocess import (build_clinician_summary, split_references, structure_explanation, extract_references_ids)
import time


def run_llm_stage(stage4):
    # 1) Get grounded payload (Stage 4C)
    payload = dict(stage4)
    
    # 2a) Decide system mode (System Controller)
    decision_mode = stage4.get("decision_mode", "NORMAL")
    confidence = stage4.get("confidence", 0.0)


    payload["risk_score"] = stage4.get("risk_score")
    payload["display_risk_score"] = stage4.get("display_risk_score")
    payload["risk_label"] = stage4.get("risk_label")
    payload["shap_features"] = payload.get("shap", {})
    payload["clinical_shap_summary"] = ""
    payload["shap_mismatch"] = payload.get("shap_mismatch", False)
    payload["shap_missing_features"] = payload.get("shap_missing_features", [])


    # 3) Build LLM prompt
    system_prompt, user_prompt = build_llm_prompt(payload)

    # 4) Call real LLM
    llm = GroqLLMClient()
    if decision_mode!="SAFE":
        llm_start = time.time()
        explanation = llm.generate(system_prompt, user_prompt)
        llm_latency_ms = (time.time() - llm_start) * 1000
    else:
        explanation = ""
        llm_latency_ms = 0    

    if not explanation or not explanation.strip():
        explanation = "Model produced no explanation. SHAP evidence suggests: " + payload.get("clinical_shap_summary", "")

    clinician_summary = ""
    if decision_mode == "SAFE":
        explanation = ("System Reliability is Limited."
                       "Automated Reasoning is Restricted and Clinician Review is Recommended.")
        clinician_summary = explanation
        explanation_body = explanation
        references = ""


    else:
        explanation_body, references = split_references(explanation) 
        clinician_summary = build_clinician_summary("", payload)

    # Ensure clinician summary is concise and not SHAP repetition
    if clinician_summary:
        clinician_summary = clinician_summary.replace("\n", " ").strip()

        if len(clinician_summary) > 200:
            clinician_summary = clinician_summary[:200].rsplit(".", 1)[0] + "."
    
    original_llm_text = explanation

    if not explanation_body:
        explanation_body = original_llm_text

    explanation_body = structure_explanation(explanation_body)

     # 5) output guardrails
    guarded = apply_output_guardrails(
        llm_text=explanation_body,
        decision_mode=decision_mode,
        confidence=confidence,
    )

    #  SAFE MERGE
    if isinstance(guarded, dict):
        extracted = guarded.get("full_text") or guarded.get("text")
        if extracted and extracted.strip():
            explanation_body = extracted
    elif isinstance(guarded, str) and guarded.strip():
        explanation_body = guarded

    if not explanation_body:
        explanation_body = original_llm_text

    if payload.get("retrieved_evidence"):
       evidence_text = payload.get("retrieved_evidence") or []
       # Join list of evidence strings into a single string for reference extraction
       evidence_string = " ".join(evidence_text) if isinstance(evidence_text, list) else evidence_text
       references = extract_references_ids(evidence_string)

    # Explainability health
    explanation_present = bool(explanation_body and explanation_body.strip())

    shap_struct = payload.get("shap") or {}
    vector_available = shap_struct.get("vector_available", False)
    top_pos = shap_struct.get("top_positive", []) 
    top_neg = shap_struct.get("top_negative", []) 
    shap_explanation = payload.get("shap_explanation") or ""

    shap_present = bool(vector_available or top_pos or top_neg)

    driver_list = [d["feature"] for d in top_pos + top_neg]

    guard_blocked = isinstance(guarded, dict) and guarded.get("mode") in ["BLOCKED", "EMPTY"]

    if guard_blocked:
        explainability_status = "Unavailable"

    elif shap_present:
        explainability_status = "Available"
    elif explanation_present:
        explainability_status = "Degraded"
    else:
        explainability_status = "Unavailable"


    if guard_blocked or explainability_status == "Unavailable" or decision_mode == "SAFE":
        reasoning_confidence = "LOW"

    elif explainability_status == "Degraded":
        reasoning_confidence = "MEDIUM"

    else:
        reasoning_confidence = "HIGH"

    retrieved = payload.get("retrieved_evidence")
    retrieved_failed = payload.get("retrieval_failed", False)

    reasoning_metadata = {
    "llm_used": decision_mode != "SAFE",
    "llm_fallback": decision_mode == "SAFE",
    "evidence_used": bool(retrieved),
    "shap_used": shap_present,
    "confidence_score": confidence,
    "decision_mode": decision_mode,
    "retrieval_failed": retrieved_failed,
    "guardrail_mode": guarded.get("mode") if isinstance(guarded, dict) else "UNKNOWN",
    "explainability_status": explainability_status,
    "reasoning_confidence": reasoning_confidence,
    }

    return {
    "risk_score": payload["risk_score"],
    "confidence": payload["confidence"],
    "decision_mode": decision_mode,
    "shap_explanation": payload.get("shap_explanation"),
    "retrieved_evidence": payload.get("retrieved_evidence"),
    "model_version": payload.get("model_version"),
    "clinician_summary": clinician_summary,
    "full_explanation": explanation_body,
    "references": references,
    "reasoning_metadata": reasoning_metadata,
    "llm_latency_ms": llm_latency_ms,
    }



if __name__ == "__main__":
    output = run_llm_stage()
    print("\nFinal Explanation:\n")
    print(output["full_explanation"])
    if output["references"]:
        print("\nReferences:\n")
        print(output["references"])
