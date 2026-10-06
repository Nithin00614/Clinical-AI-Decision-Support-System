import streamlit as st 
import requests 
import time
import json
import pandas as pd
import os
import requests
from collections import defaultdict
from datetime import datetime
from pathlib import Path

Path("hitl_audit_log.jsonl").touch(exist_ok=True)

API = "http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="CKD Clinical AI", layout="wide")

st.title("CKD Clinical AI Decision Support System")
st.info("⚕️ Clinical Decision Support Tool")
st.caption("This System Assists Clinicians by Highlighting CKD Risk Patterns based on EHR Features.")
st.caption("Predictions are not Medical Diagnoses and Require Physician Validation.")

st.markdown("---")

#========================= HELPERS =========================

def risk_badge(score): 
    if score < 0.3: return "🟢 Low Risk" 
    elif score < 0.7: return "🟡 Moderate Risk" 
    return "🔴 High Risk"

def conf_badge(conf): 
    if conf > 0.75: return "🟢 High confidence" 
    elif conf > 0.45: return "🟡 Moderate confidence" 
    return "🔴 Low confidence"

def mode_badge(mode): 
    if mode == "SAFE": return "🟢 SAFE" 
    elif mode == "VERBOSE": return "🟡 VERBOSE" 
    elif mode == "FALLBACK": return "🔴 FALLBACK" 
    return mode

def abnormal_snapshot(payload): 
    if not payload:
        return []
    abnormal = [] 
    if payload.get("bp", 0) > 140: 
        abnormal.append("High BP") 
    if payload.get("sc", 0) > 1.3: 
        abnormal.append("Elevated Creatinine") 
    if payload.get("hemo", 0) < 11: 
        abnormal.append("Low Hemoglobin") 
    if payload.get("rbc", 1) == 0: 
        abnormal.append("Abnormal RBC")  
    if payload.get("al", 0) > 1: 
        abnormal.append("Albuminuria") 
    return abnormal

def explanation_confidence(mode): 
    if mode == "SAFE": return "LOW" 
    elif mode == "VERBOSE": return "MODERATE" 
    return "HIGH"


if "latest" not in st.session_state: 
    st.session_state.latest = None

if "timeline" not in st.session_state: 
    st.session_state.timeline = []

if "payload" not in st.session_state: 
    st.session_state.payload = None


with st.sidebar:
    st.markdown("### System Status")
    if st.session_state.latest:
        st.success("Prediction Loaded")
    else:
        st.warning("Awaiting Prediction")    

tabs = st.tabs([ "Prediction", "Reasoning", "Clinician Override", "Review Queue", "System" ])


with tabs[0]:

    st.subheader("Clinical Risk Assessment")

    # ---------------- Sample Loaders ----------------

    st.markdown("### Load Sample Case")

    col_s1, col_s2, col_s3 = st.columns(3)

    if col_s1.button("🟢 Low Risk Sample"):
        st.session_state.update({
            "bp": 100, "sg": 1.025, "al": 0, "su": 0,
            "rbc": 1, "bu": 10, "sc": 0.6,
            "sod": 142, "pot": 4.1,
            "hemo": 15.5, "wbcc": 6000,
            "rbcc": 5.2, "htn": 0
        })

    if col_s2.button("🟡 Moderate Risk Sample"):
        st.session_state.update({
            "bp": 120, "sg": 1.020, "al": 0, "su": 0,
            "rbc": 1, "bu": 20, "sc": 1.0,
            "sod": 140, "pot": 4.2,
            "hemo": 14.0, "wbcc": 8000,
            "rbcc": 5.0, "htn": 0
        })

    if col_s3.button("🔴 High Risk Sample"):
        st.session_state.update({
            "bp": 155, "sg": 1.010, "al": 3, "su": 1,
            "rbc": 0, "bu": 60, "sc": 2.8,
            "sod": 132, "pot": 5.5,
            "hemo": 9.5, "wbcc": 11000,
            "rbcc": 3.5, "htn": 1
        })

    st.divider()

    # ---------------- Input Section ----------------

    st.markdown("### Clinical Feature Input")

    col1, col2 = st.columns(2)

    with col1:
        bp = st.number_input("Blood Pressure", key="bp")
        sg = st.number_input("Specific Gravity", format="%.3f", key="sg")
        al = st.number_input("Albumin", key="al")
        su = st.number_input("Sugar", key="su")
        rbc = st.selectbox("RBC (0=abnormal,1=normal)", [0,1], key="rbc")
        bu = st.number_input("Blood Urea", key="bu")
        sc = st.number_input("Serum Creatinine", key="sc")

    with col2:
        sod = st.number_input("Sodium", key="sod")
        pot = st.number_input("Potassium", key="pot")
        hemo = st.number_input("Hemoglobin", key="hemo")
        wbcc = st.number_input("WBC Count", key="wbcc")
        rbcc = st.number_input("RBC Count", key="rbcc")
        htn = st.selectbox("Hypertension (0/1)", [0,1], key="htn")

    st.divider()

    # ---------------- Run Prediction ----------------

    if st.button("Run Prediction"):

        payload = {
            "bp": bp, "sg": sg, "al": al, "su": su,
            "rbc": rbc, "bu": bu, "sc": sc,
            "sod": sod, "pot": pot,
            "hemo": hemo, "wbcc": wbcc,
            "rbcc": rbcc, "htn": htn
        }

        start_time = time.time()
        response = requests.post(f"{API}/reason", json=payload)
        total_latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            st.session_state.latest = data
            st.session_state.prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            st.error("Backend error. Please check API.")

    st.divider()

    # Display prediction results if they exist in session state
    if st.session_state.latest is not None:
        
        data = st.session_state.latest
        pred = data["prediction"]
        full_response = data
        metrics = data.get("system_metrics", {})

        risk = float(pred["risk_score"])
        risk_label = pred["risk_label"]
        reasoning_data = full_response.get("reasoning", {})
        confidence = float(pred["confidence"])
        mode = pred["decision_mode"]

        if risk >= 0.999:
            risk_display = "0.99"
        else:
            risk_display = f"{risk:.3f}"    

        separation = abs(risk - 0.5) 
        if separation >= 0.40:
            stability = "Very Stable"
        elif separation >= 0.25:
            stability = "Stable"
        elif separation >= 0.10:
            stability = "Moderate Stability"
        else:
            stability = "Near Decision Boundary"

        # ================= RISK SUMMARY =================

        st.markdown("## Risk Summary")
        st.caption("Calibrated Risk Probability Reflects Calibrated Mode Output, Not a Confirmed Diagnosis.")
        st.divider()
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("### Estimated Probability (Calibrated)")

            if risk_label == "LOW":
                st.success(f"🟢 Low CKD Risk – {risk_display}")
            elif risk_label == "MODERATE":
                st.warning(f"🟡 Moderate CKD Risk – {risk_display}")
            else:
                st.error(f"🔴 High CKD Risk – {risk_display}")
            
            if "prediction_time" in st.session_state:
                st.caption(f"Prediction Generated: {st.session_state.prediction_time}")
            
            st.caption("Decision Threshold: 0.50(Calibrated)")

            if risk < 0.30:
                category = "LOW"
            elif risk < 0.60:
                category = "MODERATE"
            else:
                category = "HIGH"    

            st.markdown(f"***Decision Stability Distance:** {separation:.3f}")

            if separation >= 0.40:
                st.success("Decision Stability: Very Stable")
            elif separation >= 0.25:
                st.success("Decision Stability: Stable")
            elif separation >= 0.10:
                st.warning("Decision Stability: Moderate Stability")
            else:
                st.error("Decision Stability: Near Decision Boundary")
            st.caption("Distance of prediction from classification boundary (0.5). Higher values indicate more stable classification.")

            st.markdown("### Key Model Drivers (SHAP)")

            explain = data.get("explainability", {})
            shap_block = explain.get("shap", {})

            top_positive = shap_block.get("top_positive", [])
            top_negative = shap_block.get("top_negative", [])

            shown = False

            for item in top_positive[:2]:
                feature = item.get("feature")
                impact = item.get("impact")
                st.markdown(f"• **{feature}** ↑ `{abs(impact):.2f}`")
                shown = True

            for item in top_negative[:2]:
                feature = item.get("feature")
                impact = item.get("impact")
                st.markdown(f"• **{feature}** ↓ `{abs(impact):.2f}`")
                shown = True

            if not shown:
                st.info("No SHAP drivers available for this prediction.")

            st.markdown("### Prediction Trust Score")

            explain_rel = reasoning_data.get("explanation_reliability", {}).get("score", 0)
            trust_score = (confidence + explain_rel) / 2

            st.metric("Trust Score", f"{trust_score:.2f}")
            st.caption("Combined Assessment of Model Prediction Stability and Explanation Grounding.")


            st.markdown("### Prediction Reliability")

            if confidence <= 0.39:
                st.error(f"Prediction Reliability: LOW — {confidence:.2f}")
            elif confidence <= 0.69:
                st.warning(f"Prediction Reliability: MEDIUM — {confidence:.2f}")
            else:
                st.success(f"Prediction Reliability: HIGH — {confidence:.2f}")

            st.caption("Represents Calibrated System Reliability and Prediction Stability - Not Clinical Certainity. ")    

            model_version = pred.get("model_version", "UNKNOWN")
            Training_data_version = "CKD_DATASET_v1.0"
            st.caption(f"MODEL Version: {model_version}")
            st.caption(f"Training Data Version: {Training_data_version}")

        with col_r2:

            st.markdown("### Decision Context")
            st.caption(f"**Decision Mode**: {mode}")
            decision_source = pred.get("decision_source", "MODEL")
            st.caption(f"**Decision Source**: {decision_source}")

            st.markdown("### Model Explainability")
            explain_status = full_response.get("explainability_status")

            if explain_status is None:
                explain_status = "UNAVAILABLE"

            status_lower = explain_status.lower() if isinstance(explain_status, str) else ""
            
            if "available" in status_lower:
                st.success("SHAP Explainability: AVAILABLE")
            elif "degraded" in status_lower:
                st.warning("SHAP Explainability: DEGRADED")
            else:
                st.error("SHAP Explainability: UNAVAILABLE")

            st.caption("Reflects Availability of SHAP-Based Model Drivers.")
            
            st.divider()
            st.markdown("### System Mode")
            if mode == "SAFE":
                st.info("SAFE Mode Triggered Due to Low Prediction Separation (near decision boundary) or Degraded Explainability.")
            elif mode == "VERBOSE":
                st.info("VERBOSE Mode Activated for Extended Reasoning Transparency.")
            else:
                st.success("NORMAL Mode — Stable Prediction State.")

            st.caption("Reflects System Guardrail State Controlling Explanation Behaviour.")
        st.divider()

        # ================= ABNORMAL SNAPSHOT =================

        # Need to reconstruct payload from session state to get abnormal snapshot
        payload_keys = ["bp", "sg", "al", "su", "rbc", "bu", "sc", "sod", "pot", "hemo", "wbcc", "rbcc", "htn"]
        payload = {key: st.session_state.get(key, 0) for key in payload_keys}

        abnormal = abnormal_snapshot(payload)
        if abnormal:
            st.markdown("### Structured Clinical Snapshot")
            renal = []
            hemodynamic = []
            hematology = []
            metabolic = []

            for item in abnormal:
                text = item.lower()
                if "creatinine" in text or "albumin" in text or "gfr" in text:
                    renal.append(item)
                elif "bp" in text:
                    hemodynamic.append(item)
                elif "hemoglobin" in text or "rbc" in text:
                    hematology.append(item)
                else:
                    metabolic.append(item)

            if hemodynamic:
                st.markdown(f"**Hemodynamics:** {', '.join(hemodynamic)}")

            if renal:
                st.markdown(f"**Renal Markers:** {', '.join(renal)}")

            if hematology:
                st.markdown(f"**Hematology:** {','.join(hematology)}")

            if metabolic:
                st.markdown(f"**Metabolic:** {','.join(metabolic)}")            


        # ================= SYSTEM METRICS =================

        retrieval_ms = metrics.get("retrieval_latency_ms", 0)
        shap_ms = metrics.get("shap_latency_ms", 0)
        llm_ms = metrics.get("llm_latency_ms", 0)
        total_ms = metrics.get("total_latency_ms", 0)

        overhead_ms = max(total_ms - (retrieval_ms + shap_ms + llm_ms), 0)

        st.markdown("### Operational Latency (Audit Transparency Metrics)")
        st.caption("System Performance Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric("Retrieval (ms)", int(retrieval_ms))
        m2.metric("SHAP (ms)", round(shap_ms, 2))
        m3.metric("LLM (ms)", int(llm_ms))
        m4.metric("Overhead (ms)", int(overhead_ms))
        m5.metric("Total (ms)", int(total_ms))

        st.caption("Confidence Reflects Model Calibration Reliability and Output Stability - Not Clinical Certainty.")

# ===================== REASONING TAB =====================

with tabs[1]:

    if st.session_state.latest is None:
        st.info("Run prediction to view reasoning.")
    else:
        data = st.session_state.latest

        prediction = data.get("prediction", {})
        explainability = data.get("explainability", {})
        reasoning = data.get("reasoning", {})
        references = reasoning.get("references", [])
        system_metrics = data.get("system_metrics", {})
        mode = prediction.get("decision_mode", "NORMAL")

        metadata = reasoning.get("metadata", {})

        coverage_data = reasoning.get("coverage", {})
        traceability_data = reasoning.get("traceability", {})
        reliability_data = reasoning.get("reliability_score", {})

        st.header("Clinical Reasoning Engine")
        with st.expander("Decision Trace (Pipeline Execution)", expanded=False):
            trace_steps = [
                ("Input validation", metadata.get("input_valid", True)),
                ("Model prediction generated", True),
                ("SHAP attribution computed", metadata.get("shap_used", False)),
                ("Clinical evidence retrieved", metadata.get("evidence_used", False)),
                ("Evidence filtered by SHAP drivers", metadata.get("llm_used", False) and mode!="SAFE"),
                ("LLM interpretation generated", metadata.get("llm_used", False) and mode != "SAFE"),
                ("Guardrails Enforced", True),
                ("Reliability diagnostics computed", True),
            ]

            for step, status in trace_steps:
                icon = "✅" if status else "❌"
                st.write(f"{icon} {step}")
    
                        
        if mode == "SAFE":
            st.error("🔒 Human Review Required — Automated Reasoning Constrained due to limited Reliability Signals.")
            st.caption(
                    "SAFE mode limits automated reasoning visibility when model confidence is low. "
                    "Pipeline diagnostics continue to run for monitoring and audit traceability."
                    )
        # ======================================================
        # SECTION 1 — MODEL ATTRIBUTION (Structured SHAP)
        # ======================================================

        st.subheader("1️⃣ Model Attribution (SHAP Explainability)")

        shap_data = explainability.get("shap", {})
        shap_available = explainability.get("shap_available", False)

        top_positive = shap_data.get("top_positive", [])
        top_negative = shap_data.get("top_negative", [])

        shap_latency = system_metrics.get("shap_latency_ms", None)

        col_a, col_b = st.columns([2,1])

        with col_a:
            if shap_available:
                st.success("Explainability: AVAILABLE")
            else:
                st.error("Explainability: DEGRADED")

        with col_b:
            if shap_latency is not None:
                st.caption(f"SHAP Latency: {round(shap_latency)} ms")

        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("### 🔺 Risk Increasing Drivers")
            if top_positive:
                for idx, item in enumerate(top_positive, start=1):
                    st.write(f"{idx}. {item['feature']} (+) — {round(item['impact'],4)}")
            else:
                st.info("No Significant Protective Contibutors Detected.")

        with col_neg:
            st.markdown("### 🔻 Protective / Negative Drivers")
            if top_negative:
                for idx, item in enumerate(top_negative, start=1):
                    st.write(f"{idx}. {item['feature']} (−) — {round(item['impact'],4)}")
            else:
                st.info("No Significant Negative Contributors Available.")

        st.caption("Drivers Ranked by Absolute SHAP Contribution Magnitude.")
        st.markdown("---")

        # ======================================================
        # SECTION 2 — EVIDENCE RETRIEVAL (RAG Layer)
        # ======================================================

        st.subheader("2️⃣ Evidence Retrieval (Clinical Grounding)")

        retrieval_data = data.get("retrieval", {})
        references = reasoning.get("references", [])

        retrieval_used = retrieval_data.get("used", False)
        raw_evidence = retrieval_data.get("evidence", [])

        # Normalize evidence structure safely
        if isinstance(raw_evidence, str):
            evidence_chunks = [c.strip() for c in raw_evidence.split("[") if c.strip()]
            evidence_chunks = ["[" + c for c in evidence_chunks] # wrap string into list
        elif isinstance(raw_evidence, list):
            evidence_chunks = raw_evidence
        else:
            evidence_chunks = []

        MAX_EVIDENCE_DISPLAY = 3
        evidence_chunks = evidence_chunks[:MAX_EVIDENCE_DISPLAY]    
        retrieval_latency = system_metrics.get("retrieval_latency_ms", None)

        col_r1, col_r2 = st.columns([2,1])

        with col_r1:
            if retrieval_used:
                st.success("Retrieval Used: YES")
            else:
                st.warning("Retrieval Used: NO")

        with col_r2:
            if retrieval_latency is not None:
                st.caption(f"Retrieval Latency: {round(retrieval_latency)} ms")
        
        evidence_count = len(evidence_chunks)
        if mode == "SAFE":
            evidence_count = 0
        st.write("Evidence Count:", evidence_count)

        if mode == "SAFE":
            st.info("Evidence Display Suppressed in SAFE Mode to Limit Reliance on Potentially Unreliable Retrieval Signals.")

        if mode != "SAFE":
            st.caption(f"Showing Top {len(evidence_chunks)} Clinically Relevant Evidence Excerpts")

            if evidence_chunks:
                st.caption("Evidence Supports Key Risk Drivers.")

            if evidence_chunks:
                for idx, chunk in enumerate(evidence_chunks):
                    with st.expander(f"Retrieved Guideline Evidence {idx+1}"):
                        st.write(chunk)
            else:
                st.info("No Evidence Retrieved For This Case.")


        doc_pages = defaultdict(list)

        for ref in references:
            doc = ref.get("document", "Unknown")
            page = ref.get("page")

            if page:
                doc_pages[doc].append(page)

        st.markdown("### Evidence Sources")

        if mode == "SAFE":
            st.info("Evidence Source Details Suppressed in SAFE Mode to Limit Reliance on Potentially Unreliable Retrieval Signals.")

        for doc, pages in doc_pages.items():

            doc_name = doc.replace("_", " ").replace(".pdf", "").upper()

            page_list = ", ".join(str(p) for p in sorted(set(pages)))

            st.success(f"Guideline Source: {doc_name} | Pages: {page_list}")    

        st.markdown("---")

        # ======================================================
        # SECTION 3 — GENERATED CLINICAL REASONING
        # ======================================================

        st.subheader("3️⃣ Generated Clinical Interpretation")

        clinician_summary = reasoning.get("clinician_summary", "")

        full_explanation = reasoning.get("full_explanation", "")
    


        if clinician_summary:
            parts = clinician_summary.split("Key drivers:")
            
            if len(parts) == 2:
                st.markdown(f"• {parts[0].strip()}")
                st.markdown(f"• **Key Drivers**: {parts[1].strip()}")
            else:
                for line in clinician_summary.split("\n"):
                    if line.strip():
                        st.write("•", line.strip())
        

        # Parse structured explanation into sections
        st.markdown("### Explanation Scope")
        
        if mode != "SAFE":
            st.markdown("""
            - Based on Model-Attributed Drivers  
            - Supported by Retrieved Clinical Guidelines 
            - No External Risk Factors Introduced 
            """)

        with st.expander("Full Clinical Explanation", expanded=True):
            if full_explanation:
                # Split by markdown headers to identify sections
                sections = full_explanation.split("###")
                
                for section in sections:
                    section = section.strip()
                    if not section:
                        continue
                    
                    lines = section.split("\n", 1)
                    header = lines[0].strip() if lines else ""
                    content = lines[1].strip() if len(lines) > 1 else ""
                    
                    if header and content:
                        st.markdown(f"### {header}")
                        
                        # Handle bulleted lists for driver sections
                        if "Drivers" in header or "drivers" in header.lower():
                            for line in content.split("\n"):
                                if line.strip().startswith("-"):
                                    st.markdown(f"  {line.strip()}")
                                elif line.strip():
                                    st.markdown(f"  • {line.strip()}")
                        else:
                            st.write(content)
                    elif header:
                        st.markdown(f"### {header}")
            else:
                st.info("No Explanation Available.")
       
           

        # Reasoning confidence and guardrail status
        reasoning_conf = metadata.get("reasoning_confidence", "UNKNOWN")
        guardrail_mode = metadata.get("guardrail_mode", "UNKNOWN")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            if reasoning_conf == "HIGH":
                st.success(f"AI Reasoning Integrity: {reasoning_conf}")

            elif reasoning_conf == "MEDIUM":
                st.warning(f"AI Reasoning Integrity: {reasoning_conf}")

            elif reasoning_conf == "LOW":
                st.error(f"AI Reasoning Integrity: {reasoning_conf}")

            else:
                st.info(f"AI Reasoning Integrity: {reasoning_conf}")

        with col_c2:
            if mode == "SAFE":
                guardrail_mode = "SAFE"

            if guardrail_mode == "SAFE":
                st.error("Guardrail Mode: SAFE")
            else:
                st.info("Guardrail Mode: NORMAL")

        st.caption("System Provides Decision Support Guidance. Final Clinical Judgement Remains With the Clinician.")

        # ======================================================
        # SECTION 4 — EXPLANATION ALIGNMENT VALIDATION
        # ======================================================

        st.subheader("4️⃣ Explanation Alignment Validation")

        decision_mode = metadata.get("decision_mode", prediction.get("decision_mode"))

        if decision_mode == "SAFE":
            st.info("Alignment Validation Skipped Due to Constrained Reasoning Signals in SAFE Mode.")
            st.markdown("---")
        else:
            llm_text = full_explanation.lower()

        shap_features = [f["feature"].lower() for f in top_positive] if top_positive else []

        if decision_mode != "SAFE" and shap_features and llm_text:
            mentioned = sum(1 for f in shap_features if f in llm_text)
            coverage = mentioned / len(shap_features)
            st.caption(f"{mentioned} of {len(shap_features)} top drivers explicitly referenced.")

            st.markdown(f"**Top Feature Coverage:** {coverage:.2f} ({coverage*100:.0f}%)")
            st.caption("Fraction of Top SHAP Drivers Explicitly Referenced in Explanation.")

            if coverage < 0.4:
                st.warning("Weak Alignment Between Model Drivers and Explanation.")
            elif coverage < 0.7:
                st.info("Moderate Alignment Between SHAP Drivers and Explanation.")
            else:
                st.success("Strong Alignment Between Model Drivers and Explanation.")
        else:
            st.info("Alignment Validation Unavailable.")

        st.markdown("---")



        # ==========================================================
        # SECTION 4B — Explanation Reliability Diagnostics
        # ==========================================================

        st.subheader("Explanation Reliability Diagnostics")

        # Extract reasoning block from API response
        response_json = st.session_state.get("latest", {}) 
        reasoning_data = response_json.get("reasoning", {})

        reliability_data = reasoning_data.get("explanation_reliability", {})

        coverage_score = reliability_data.get("coverage", 0)
        traceability_score = reliability_data.get("traceability", 0)
        reliability_score = reliability_data.get("score", 0)
        reliability_label = reliability_data.get("level", "UNKNOWN")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric("Driver Coverage", f"{coverage_score:.2f}")

        with col_r2:
            st.metric("Evidence Traceability", f"{traceability_score:.2f}")

        with col_r3:
            st.metric("Explanation Reliability", f"{reliability_score:.2f}")

        # Reliability indicator
        if reliability_label == "HIGH":
            st.success("Explanation Reliability: HIGH")
        elif reliability_label in ["MEDIUM", "MODERATE"]:
            st.warning("Explanation Reliability: MODERATE")
        else:
            st.error("Explanation Reliability: LOW")
        st.progress(min(max(reliability_score, 0.0), 1.0))
        st.caption(f"Reliability Score: {reliability_score:.2f}")
        st.markdown("---")

        
        # ======================================================
        # SECTION 5 — RELIABILITY & GOVERNANCE DIAGNOSTICS
        # ======================================================

        st.subheader("5️⃣ Reliability & Governance Diagnostics")

        llm_used = metadata.get("llm_used", False)
        llm_fallback = metadata.get("llm_fallback", False)
        shap_used = metadata.get("shap_used", False)
        evidence_used = metadata.get("evidence_used", False)
        decision_mode = metadata.get("decision_mode", prediction.get("decision_mode"))
        decision_source = prediction.get("decision_source")

        # Confidence vs Reasoning Confidence Consistency Check
        # ======================================================

        model_conf = float(prediction.get("confidence", 0.0))
        reasoning_conf = metadata.get("reasoning_confidence", "UNKNOWN")

        # Convert numeric confidence into category
        if model_conf <= 0.39:
            model_conf_label = "LOW"
        elif model_conf <= 0.69:
            model_conf_label = "MEDIUM"
        else:
            model_conf_label = "HIGH"

        st.markdown("### Confidence Consistency Check")

        st.write("Prediction Reliability Level:", model_conf_label)
        st.write("Reasoning Confidence Level:", reasoning_conf)

        # Detect mismatch
        decision_mode = metadata.get("decision_mode", prediction.get("decision_mode"))

        if decision_mode == "SAFE":
            st.info("Alignment Validation Skipped in SAFE Mode Due to Intentionally Constrained Reasoning.")
            st.markdown("---")
        

        elif reasoning_conf != "UNKNOWN" and model_conf_label != reasoning_conf:
            st.warning(" ⚠️ Model Confidence and Reasoning Confidence Level Are Not Aligned.")
        else:
            st.success("Model Confidence and Reasoning Confidence are Aligned.")

        st.subheader("AI System Governance & Safety")

        metadata = reasoning.get("metadata", {})

        llm_fallback = metadata.get("llm_fallback", False)
        evidence_used = metadata.get("evidence_used", False)
        shap_used = metadata.get("shap_used", False)
        retrieval_failed = metadata.get("retrieval_failed", False)
        explainability_status = metadata.get("explainability_status")


        if not explainability_status:
            explainability_status = "UNAVAILABLE"

        critical = False
        degraded = False

        if llm_fallback and decision_mode != "SAFE":
            critical = True    

        if explainability_status in ["DEGRADED","UNAVAILABLE"]:
            degraded = True

        if retrieval_failed:
            degraded = True

        if critical:
            st.error("🔴 Critical: LLM Fallback Activated. Explanation Reliability May Be Limited.")
        elif decision_mode == "SAFE":
            st.warning("🟡 Safe Mode Active: LLM Reasoning Restricted due to Safety Guardrails.")    
        elif degraded:
            st.warning("🟡 Degraded: One or More Reasoning Components Partially Unavailable.")
        else:
            st.success("🟢 System Operating Normally — All Core Reasoning Components Are Active.")

        st.caption("System Health Reflects Reasoning Pipeline Integrity and Safety Guardrails, Not Clinical Certainity.")

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.write("LLM Explanation:", "✅" if llm_used else "❌")
            st.write("LLM Fallback Activated:", "✅" if llm_fallback else "❌")
            st.write("SHAP Used:", "✅" if shap_used else "❌")


        with col_d2:
            if decision_mode == "SAFE":
                st.write("Evidence Retrieval:", "⚠️ Suppressed ")
            elif retrieval_failed:
                st.write("Evidence Retrieval:", "❌ Retrieval Failed")
            elif evidence_used:
                st.write("Evidence Retrieval:", "✅ Retrieved ")
            else:
                st.write("Evidence Retrieval:", " No Relevant Evidence Found")
            st.write("Decision Mode:", decision_mode)
            st.write("Decision Source:", decision_source)


        # ======================================================
        # SECTION 6 — SYSTEM OBSERVABILITY (LLM Layer)
        # ======================================================
        st.divider()
        st.subheader("6️⃣ Operational Health")

        llm_latency = system_metrics.get("llm_latency_ms", None)
        total_latency = system_metrics.get("total_latency_ms", None)

        col_o1, col_o2 = st.columns(2)

        with col_o1:
            if llm_latency is not None:
                st.metric("LLM Latency (ms)", int(llm_latency))

        with col_o2:
            if total_latency is not None:
                st.metric("Total End-to-End Latency (ms)", int(total_latency))

        st.caption("Latency Metrics Enable Scaling Decisions And Performance Diagnostics.")

# ==============================
# Clinician Override Tab
# ==============================

with tabs[2]:

        st.header("Clinician Override Panel")
        st.markdown("### Human-In-The-Loop Clinican Governance")


        # ------------------------------
        # Prediction Guard
        # ------------------------------
        if "latest" not in st.session_state or st.session_state.latest is None:
            st.warning("Run a prediction first to enable clinician override.")
        else:
            # Extract variables from session state
            data = st.session_state.get("latest", {})

            pred = data.get("prediction", {})
            reasoning_data = data.get("reasoning", {})

            risk = float(pred.get("risk_score", 0))
            risk_label = pred.get("risk_label", "UNKNOWN")
            confidence = float(pred.get("confidence", 0))
            
            if risk >= 0.999:
                risk_display = "0.99"
            else:
                risk_display = f"{risk:.3f}"
            
            separation = abs(risk - 0.5)
            if separation >= 0.40:
                stability = "Very Stable"
            elif separation >= 0.25:
                stability = "Stable"
            elif separation >= 0.10:
                stability = "Moderate Stability"
            else:
                stability = "Near Decision Boundary"
            
            explain_rel = reasoning_data.get("explanation_reliability", {}).get("score", 0)
            trust_score = (confidence + explain_rel) / 2

            # UI variables already computed earlier
            ai_risk = risk_label
            ai_prob = risk_display
            ai_conf = confidence
            ai_trust = round(trust_score, 2)
            ai_stability = stability

            st.divider()

            # --------------------------------
            # AI vs Clinician Layout
            # --------------------------------
            col_ai, col_doc = st.columns(2)

            # =========================
            # AI RECOMMENDATION PANEL
            # =========================
            with col_ai:

                st.subheader("AI Recommendation")

                st.metric("Risk Level", ai_risk)
                st.metric("Probability", ai_prob)
                st.metric("Model Confidence", f"{ai_conf:.2f}")
                st.metric("Trust Score", f"{ai_trust:.2f}")
                st.metric("Decision Stability", ai_stability)

                st.caption("AI Output Based On Calibrated Model Prediction and SHAP Explainability.")

                st.markdown("#### Key Model Drivers (SHAP)")

                if top_positive:
                    for item in top_positive[:2]:
                        feature = item.get("feature")
                        impact = item.get("impact")
                        st.write(f"↑ **{feature}** ({abs(impact):.2f})")

                if top_negative:
                    for item in top_negative[:2]:
                        feature = item.get("feature")
                        impact = item.get("impact")
                        st.write(f"↓ **{feature}** ({abs(impact):.2f})")


            # =========================
            # CLINICIAN DECISION PANEL
            # =========================
            with col_doc:

                st.subheader("Clinician Decision")

                case_id = st.text_input("Case ID", value="CKD-CASE-003")

                decision = st.radio(
                    "Clinician Action",
                    ["Accept AI Recommendation", "Override AI Recommendation"]
                )

                override_risk = None
                override_reason = ""
                clean_justification = ""
                clinician_confidence = 0.75

                if decision == "Override AI Recommendation":

                    override_risk = st.selectbox(
                        "Override Risk Level",
                        ["LOW", "MODERATE", "HIGH"]
                    )

                    override_reason = st.text_area(
                        "Clinical Justification",
                        placeholder="Provide clinical justification for overriding the AI Recommendation (lab findings, commorbidities, contextual risk factors)..."
                    )
                    clean_justification = override_reason.replace("\n", " ")
                    clinician_confidence = st.slider(
                        "Clinician Confidence",
                        0.0,
                        1.0,
                        0.80
                    )

                flag_case = st.checkbox("Flag Case for Clinical Review")

            st.divider()

            # ================================
            # Disagreement Detection
            # =================================

            if decision == "Override AI Recommendation":

                if override_risk != ai_risk:

                    st.error(
                        f"⚠️ AI–Clinician Disagreement Detected: Model predicted **{ai_risk}**, Clinician Selected **{override_risk}**."
                    )

                    st.caption(
                        "This Disagreement Will be Logged For Model Monitoring and Clinical Governance."
                    )

            # =================================
            # SUBMIT OVERRIDE
            # =================================
            if st.button("Submit Decision"):

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                final_decision = override_risk if decision == "Override AI Recommendation" else ai_risk

                status = "OVERRIDDEN" if decision == "Override AI Recommendation" else "ACCEPTED"


                audit_entry = {
                    "case_id": case_id,
                    "timestamp": timestamp,
                    "ai_prediction": ai_risk,
                    "ai_probability": ai_prob,
                    "ai_confidence": ai_conf,
                    "trust_score": ai_trust,
                    "decision_stability": ai_stability,
                    "clinician_action": decision,
                    "clinician_decision": final_decision,
                    "override_reason": clean_justification,
                    "override_confidence": clinician_confidence,
                    "flag_for_review": flag_case,
                    "status": status,
                    "model_version": "CKD-Predictor-v1.0"
                }
                # --------------------------------
                # Save to audit log
                # --------------------------------
                with open("hitl_audit_log.jsonl", "a") as f:
                    f.write(json.dumps(audit_entry) + "\n")

                st.success("Decision Logged To Human-In-The-Loop Audit File.")
                # --------------------------------
                # UI status feedback
                # --------------------------------
                if status == "OVERRIDDEN":
                    st.error(f"⚠️ Clinician Override Applied → Final Decision: {final_decision}")
                else:
                    st.success("✅ AI Recommendation Accepted")

                if flag_case:
                    st.info("Case Flagged for Review.")

                # -------------------------
                # FINAL DECISION SUMMARY
                # -------------------------

                st.divider()
                st.subheader("Final Clinical Decision")

                st.write(f"AI Prediction: *{ai_risk}*")

                if decision == "Override AI Recommendation":
                    st.write(f"Clinician Decision: *{override_risk}*")
                    st.error("Override Applied")
                else:
                    st.write(f"Clinician Decision: *{ai_risk}*")
                    st.success("AI Recommendation Accepted")

                st.caption("All Clinician Decisions are Recorded For Audit and Governance Review.")
                st.markdown("---")
                st.markdown("### Clinical Decision Notice")
                st.caption("This System Provides Decision-Support Guidance Only.")
                st.caption("Final Diagnosis and Treatment Decisions Remain Under Physician Responsibility.")
                st.caption("Clinician Overrides are Logged for Audit and Model Governance Review.")

with tabs[3]:

    st.subheader("Clinical AI Review Queue")

    log_file = "hitl_audit_log.jsonl"

    if not os.path.exists(log_file):
        st.warning("No Review Cases Available yet.")
    else:

        records = []

        with open(log_file, "r") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except:
                    pass

        if len(records) == 0:
            st.info("No Cases Logged yet.")
        else:

            df = pd.DataFrame(records)
            df = df.sort_values("timestamp", ascending=False)

            # =============================
            # Governance Metrics
            # =============================

            total_cases = len(df)
            overrides = len(df[df["status"] == "OVERRIDDEN"])
            flagged = len(df[df["flag_for_review"] == True])

            override_rate = (overrides / total_cases) * 100 if total_cases > 0 else 0
            st.success("Clinical AI Governance Dashboard")
            st.caption("Human-in-the-loop Monitoring of AI Clinical Decisions And Overrides.")
            st.divider()
            st.markdown("### Governance Metrics")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Total Cases", total_cases)
            m2.metric("Overrides", overrides)
            m3.metric("Flagged Cases", flagged)
            m4.metric("Override Rate", f"{override_rate:.1f}%")

            st.caption(f"Audit Log Records: {total_cases}")
            st.divider()

            # =============================
            # Risk Distribution
            # =============================

            st.markdown("### Risk Distribution")

            risk_counts = df["clinician_decision"].value_counts()

            r1, r2, r3 = st.columns(3)

            r1.metric("LOW", risk_counts.get("LOW", 0))
            r2.metric("MODERATE", risk_counts.get("MODERATE", 0))
            r3.metric("HIGH", risk_counts.get("HIGH", 0))

            st.caption("Risk Distribution of Reviewed Cases")
            st.divider()

            # =============================
            # Filters
            # =============================

            st.markdown("### Case Filters")

            filter_option = st.selectbox(
                "Filter Cases",
                ["All Cases", "Flagged Cases Only", "Overrides Only"]
            )

            filtered_df = df.copy()

            if filter_option == "Flagged Cases Only":
                filtered_df = df[df["flag_for_review"] == True]

            elif filter_option == "Overrides Only":
                filtered_df = df[df["status"] == "OVERRIDDEN"]

            st.caption(f"Showing {len(filtered_df)} cases")

            st.divider()

            def status_badge(row):
                if row["flag_for_review"]:
                    return "🔴 FLAGGED"
                elif row["status"] == "OVERRIDDEN":
                    return "🟠 OVERRIDDEN"
                else:
                    return "🟢 APPROVED"

            filtered_df["status_indicator"] = filtered_df.apply(status_badge, axis=1)

            # =============================
            # Case Review Table
            # =============================

            st.markdown("### Case Review Table")

            display_cols = [
                "case_id",
                "timestamp",
                "ai_prediction",
                "clinician_decision",
                "status_indicator"
            ]

            st.dataframe(filtered_df[display_cols], width="content")

            st.divider()

            # =============================
            # Disagreement Detection
            # =============================

            disagreements = df[df["ai_prediction"] != df["clinician_decision"]]

            if len(disagreements) > 0:

                st.markdown("### Human–AI Clinician Conflicts")

                for _, row in disagreements.iterrows():

                    st.warning(
                        f"Case {row['case_id']} → AI Predicted {row['ai_prediction']} But Clinician Decided {row['clinician_decision']}"
                    )

                st.divider()

            # =============================
            # Case Detail Viewer
            # =============================

            st.markdown("### Case Detail Viewer")

            case_list = filtered_df["case_id"].tolist()

            if case_list:
                selected_case = st.selectbox("Select Case", case_list)

                case_data = filtered_df[filtered_df["case_id"] == selected_case].iloc[0]
                ai_pred = case_data["ai_prediction"]
                clinician_dec = case_data["clinician_decision"]

                if ai_pred != clinician_dec:
                    disagreement = "⚠ AI–Clinician Disagreement"
                else:
                    disagreement = "✓ AI–Clinician Aligned"

                with st.expander("Case Details", expanded=True):

                    st.write("Case ID:", case_data["case_id"])
                    st.write("Timestamp:", case_data["timestamp"])
                    st.write("AI Prediction:", case_data["ai_prediction"])
                    st.write("Governance Indicator:", disagreement)
                    st.write("AI Probability:", case_data["ai_probability"])
                    st.write("AI Confidence:", case_data["ai_confidence"])
                    st.write("Trust Score:", case_data["trust_score"])
                    st.write("Decision Stability:", case_data["decision_stability"])

                    st.write("Clinician Action:", case_data["clinician_action"])
                    st.write("Clinician Decision:", case_data["clinician_decision"])

                    st.write("Override Confidence:", case_data["override_confidence"])
                    st.write("Flagged For Review:", case_data["flag_for_review"])

                    st.write("**Override Clinical Justification**")
                    st.write("Model Version:", case_data["model_version"])
                    st.info(case_data["override_reason"])
            else:
                st.warning("No cases available for the selected filter.")


# ============================
# System Metadata Tab & Health Tab
# ============================

with tabs[4]:

    st.subheader("AI System Metadata & Health")

    # Initialize default values
    metadata = {
        "Train_data": "N/A",
        "Model_version": "N/A",
        "Model_trained_on": "N/A",
        "Feature_schema": "N/A",
        "Explainability": "N/A",
        "Pipeline_version": "N/A",
        "Llm_provider": "N/A",
        "Llm_model": "N/A"
    }
    
    status = {
        "Model": "Checking...",
        "Retrieval": "Checking...",
        "Llm": "Checking...",
        "Hitl": "Checking...",
        "Version": "N/A"
    }
    
    backend_available = True

    try:
        metadata_response = requests.get("http://127.0.0.1:8000/api/v1/metadata", timeout=2)
        status_response = requests.get("http://127.0.0.1:8000/api/v1/system/status", timeout=2)

        if metadata_response.status_code == 200:
            metadata.update(metadata_response.json())
        
        if status_response.status_code == 200:
            status.update(status_response.json())
            
    except Exception as e:
        backend_available = False
        st.warning("⚠️ Backend service not available. Showing cached/default system information.")

    # ============================
    # System Health Banner
    # ============================

    try:
        if status.get("Model") == "Ready" and status.get("Retrieval") == "Ready" and status.get("Llm") == "Ready":
            st.success("System Operational — All AI Components Are Healthy")
        else:
            st.warning("System Component Status: Some services not ready")
    except:
        st.info("System Status: Unable to determine - backend services may be initializing")

    st.caption("Deployment Environment: Clinical Decision Support Prototype | Backend: FastAPI | UI: Streamlit")
    st.divider()

    # ============================
    # System Health Metrics
    # ============================

    st.markdown("### System Health")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model Service", status.get("Model", "N/A"))
    col2.metric("Retrieval Engine", status.get("Retrieval", "N/A"))
    col3.metric("LLM Service", status.get("Llm", "N/A"))
    col4.metric("HITL Governance", status.get("Hitl", "N/A"))

    st.divider()

    # ============================
    # Model Metadata
    # ============================

    st.markdown("### Model Metadata")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model Version:**", metadata.get("Model_version", "N/A"))
        st.write("**Model Trained On:**", metadata.get("Model_trained_on", "N/A"))
        st.write("**Training_data_version:**", metadata.get("Training_data_version", "N/A"))
        st.write("**Feature Schema:**", metadata.get("Feature_schema", "N/A"))

    with col2:
        st.write("**Pipeline Version:**", metadata.get("Pipeline_version", "N/A"))
        st.write("**Explainability Engine:**", metadata.get("Explainability", "N/A"))

    st.divider()

    # ============================
    # LLM Configuration
    # ============================

    st.markdown("### LLM Configuration")

    col1, col2 = st.columns(2)

    col1.write("**Provider:** " + metadata.get("Llm_provider", "N/A").upper())
    col2.write("**Model:** " + metadata.get("Llm_model", "N/A"))

    st.divider()

    # ============================
    # Backend Information
    # ============================

    st.markdown("### Backend Information")

    st.write("**Backend Version:**", status.get("Version", "N/A"))
    
    if not backend_available:
        st.info("Backend service is currently offline. Using cached configuration.")