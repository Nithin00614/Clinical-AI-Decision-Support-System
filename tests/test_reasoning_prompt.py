from genai.evaluation.reasoning_pipeline import build_reasoning_prompt

# Mock inputs (simulate ML + SHAP output)
risk_score = 0.82
shap_dict = {
    "sc": 0.42,
    "hemo": -0.21,
    "bp": 0.28
}

# Mock retrieved chunks (from FAISS sanity test)
retrieved_chunks = [
    (
        "Elevated serum creatinine reflects reduced glomerular filtration and is associated with CKD progression.",
        {"source": "kdigo_ckd.pdf::page_12"}
    )
]

system_prompt, user_prompt = build_reasoning_prompt(
    risk_score, shap_dict, retrieved_chunks
)

print("SYSTEM PROMPT:\n", system_prompt)
print("\nUSER PROMPT:\n", user_prompt)
