# genai/evaluation/stage_4c_orchestrator.py
from genai.retrieval.retriever import search
from genai.retrieval.retrieval_adapter import format_retrieved_chunks
from genai.prompts.shap_adapter import format_shap_features
from genai.explainability.clinical_shap_translator import translate_shap
from genai.explainability.shap_formatter import build_structured_shap
from genai.explainability.clinical_shap_translator import generate_clinical_sentences
from genai.explainability.shap_loader import load_expected_features
from genai.evaluation.reasoning_controller import (compute_reasoning_confidence, detect_explanation_mismatch)
from genai.controller.system_controller import decide_mode
from genai.data.artifact_loader import load_drift_metrics_safe
import logging
import time
from genai.evaluation.explanation_coverage import compute_explanation_coverage
from genai.evaluation.traceability import compute_driver_evidence_traceability
from genai.evaluation.reliability_score import compute_explanation_reliability
from genai.explainability.clinical_shap_translator import extract_driver_features,filter_evidence_by_shap,normalize_drivers
from services.confidence_service import (compute_explainability_reliability, compute_clinician_trust ,compute_explanation_confidence, compute_reasoning_reliability, calibrate_reasoning_confidence)



def run_stage_4c(input_data: dict, risk_score: float, shap_features: dict):

    # SAFETY ASSERTION — risk propagation integrity
    assert isinstance(risk_score, (float, int)), "Risk score must be numeric"
    assert 0.0 <= risk_score <= 1.0, "Risk score out of expected range"

    risk_score = float(risk_score)
    display_risk_score = min(risk_score, 0.99)

    risk_label = "None"

    if risk_score < 0.30:
        risk_label = "LOW"
    elif risk_score < 0.70:
        risk_label = "MODERATE"
    else:
        risk_label = "HIGH"

    # 2. Explanation preparation
    shap_start = 0.0
    shap_start = time.time()
    shap_features = {k: float(v) for k,v in shap_features.items()}
    
    # Build structured shap
    try:
        structured_shap = build_structured_shap(shap_features)
    except Exception:
        structured_shap = {
            "top_positive": [],
            "top_negative": [],
            "vector_available": False
        }

    shap_latency_ms = (time.time() - shap_start) * 1000

    # Flatten for formatting (SAFE)
    flat_shap = {
        f.get("feature"): f.get("impact", 0)
        for f in structured_shap.get("top_positive", []) + structured_shap.get("top_negative", [])
    }

    shap_text = format_shap_features(flat_shap)

    if not shap_text or shap_text.strip() == "":
        shap_text = "SHAP explanation unavailable."

    shap_explanation: str = shap_text
    retrived_evidence: str = ""

    try:
        clinical_sentences = generate_clinical_sentences(structured_shap)
    except Exception:
        clinical_sentences = []

    drivers_list = extract_driver_features(structured_shap)
    drivers_list = normalize_drivers(drivers_list)


    expected_features = load_expected_features()    
    missing_features = set(expected_features) - set(shap_features.keys())
    shap_missing_features = list(missing_features)
    shap_mismatch = False

    # Evidence retrieval
    query = " ".join(drivers_list) 
    retrieval_latency_ms = 0.0
    retrieved, retrieval_latency_ms = search(query, k=4)
    retrieval_count = len(retrieved)

    raw_chunks = retrieved

   

    filtered_chunks = filter_evidence_by_shap(raw_chunks, drivers_list)

    if not filtered_chunks:
        filtered_chunks = raw_chunks[:2]


    retrieved_evidence_llm = "\n".join(
    [c[0] if isinstance(c, tuple) else str(c) for c in filtered_chunks[:2]]
)


    confidence = compute_reasoning_confidence(risk_score, structured_shap, retrieval_count)

    confidence = calibrate_reasoning_confidence(
    confidence,
    risk_score,
    structured_shap.get("vector_available", False),
    retrieval_count
    )

    logging.info(f"STAGE$ SHAP: {shap_features}")
    decision_mode = decide_mode(confidence, risk_score)


      # ===== Explainability reliability ====

    explainability_reliability = compute_explainability_reliability(
        structured_shap,
        shap_missing_features,
        shap_mismatch
    )

    explanation_confidence = compute_explanation_confidence(
        confidence,
        structured_shap.get("vector_available", False),
        retrieval_count
    )

    reasoning_reliability = compute_reasoning_reliability(
        confidence,
        retrieval_count,
        structured_shap.get("vector_available", False)
    )

    clinician_trust = compute_clinician_trust(explainability_reliability, reasoning_reliability)

    # Drift metrics are optional in v1.
    # If unavailable, system defaults to safe assumptions.
    drift_detected = False

    drift_metrics = load_drift_metrics_safe()
    drift_detected = drift_metrics.get("drift_detected", False)
    if drift_detected and shap_text:
        shap_text = " ⚠️ Distribution drift detected. Explanation reliability reduced.\n\n" + shap_text

    # SHAP guardrail detection
    shap_guardrail = {
        "vector_missing": not structured_shap.get("vector_available"),
        "feature_mismatch": shap_mismatch,
        "missing_features": len(shap_missing_features) > 0,
    }
    shap_guardrail["triggered"] = any(shap_guardrail.values())

    # prepend reliability warning if needed
    if shap_guardrail["triggered"] and shap_text:
        shap_text = "⚠️ Explanation reliability reduced.\n\n" + shap_text


    if decision_mode == "SAFE":
        logging.warning(
            f"SAFE Mode triggered - human clinician review recommended | confidence={confidence} | risk_score={risk_score}"
        )
        
        shap_explanation = "\n\n".join([
            "⚠️ Low confidence prediction.\n"
            "Top contributing factors are shown for clinician review only:"
            ,shap_text
        ])
        retrieved_evidence = "Evidence retrieval suppressed due to low confidence."

    elif decision_mode == "VERBOSE":
            safe_shap = shap_text or "SHAP unavailable"

            shap_explanation = "\n\n".join([
                "⚠️ Moderate confidence detected. "
                "The following explanation is probabilistic and should be "
                "interpreted cautiously.",
                shap_text
            ])
            retrieved_evidence = format_retrieved_chunks(retrieved)

    else:
        shap_explanation = shap_text
        retrieved_evidence = format_retrieved_chunks(retrieved)          


    # 6. Final payload (authoritative)
    
    # Extract text from evidence chunks for reliability computation
    evidence_text_chunks = []
    for chunk in filtered_chunks:
        if isinstance(chunk, tuple):
            evidence_text_chunks.append(chunk[0])
        else:
            evidence_text_chunks.append(chunk)
    
    payload = {
        "risk_score": round(risk_score, 4),
        "display_risk_score": display_risk_score,
        "probability_type": "calibrated_estimate",
        "risk_label": risk_label,
        "confidence": confidence,
        "decision_mode": decision_mode,
        "shap": structured_shap,
        "shap_missing_features": shap_missing_features,
        "shap_available": structured_shap.get("vector_available", False),
        "shap_mismatch": shap_mismatch,
        "cinical_factors": clinical_sentences,
        "drivers_list": drivers_list,
        "retrieval_latency_ms": retrieval_latency_ms,
        "model_version": "CKD-Predictor-v1.0",
        "shap_latency_ms": shap_latency_ms,
        "_evidence_chunks": evidence_text_chunks,
        "explainability": {
            "reliability": explainability_reliability,
            "confidence": explanation_confidence,
            "reasoning_reliability": reasoning_reliability,
            "clinician_trust": clinician_trust,
            "guardrail": shap_guardrail,
            "mode": decision_mode
        },
        "explanation_available": structured_shap["vector_available"],
        "guarded_output": {
            "decision_mode": decision_mode,
        },
        "shap_explanation": shap_explanation,
        "retrieved_evidence": retrieved_evidence,
    }

    return payload
