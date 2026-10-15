def run_ablation(mode, risk_score, shap_features=None, retrieved_chunks=None):
    if mode == "ml_only":
        return {
            "mode": mode,
            "output": f"Predicted CKD risk probability: {risk_score:.2f}"
        }

    if mode == "ml_shap":
        return {
            "mode": mode,
            "output": {
                "risk": risk_score,
                "shap_features": list(shap_features.keys())
            }
        }

    if mode == "full_system":
        return {
            "mode": mode,
            "output": {
                "risk": risk_score,
                "shap_features": list(shap_features.keys()),
                "evidence_used": len(retrieved_chunks)
            }
        }

    raise ValueError("Unknown Ablation mode")
