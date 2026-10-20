from genai.evaluation.evaluate_faithfulness import evaluate_faithfulness
from genai.evaluation.evaluate_grounding import evaluate_grounding

shap_features = {
    "sc": 0.40,
    "bp": 0.18
}

retrieved_chunks = [
    ("Elevated serum creatinine reflects reduced GFR in CKD patients.", 
     {"source": "kdigo_ckd.pdf:page_12"})
]

print("Faithfulness:", evaluate_faithfulness(shap_features, retrieved_chunks))
print("Grounding:", evaluate_grounding(retrieved_chunks))
