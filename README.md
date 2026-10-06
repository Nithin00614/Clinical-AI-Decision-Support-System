# AI-Powered Clinical Decision Support System for Chronic Kidney Disease (CKD)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikitlearn&logoColor=white)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-purple)
![RAG](https://img.shields.io/badge/RAG-Clinical%20Evidence-blueviolet)
![LLM](https://img.shields.io/badge/LLM-Reasoning-black)

An **Explainable AI Clinical Decision Support System** for Chronic Kidney Disease (CKD) that combines machine learning prediction, SHAP explainability, clinical evidence retrieval, and LLM-based reasoning to provide transparent and reliable clinical insights.

The system integrates **ML model predictions, explainability, retrieval-augmented reasoning, safety guardrails, and monitoring** to support clinicians with interpretable decision support.

## Project Status

This project is a **production-grade architecture demonstration** demonstrating the architecture of an explainable AI clinical decision support system.

It focuses on **system design, explainability, and AI reasoning pipelines** rather than production deployment.

The implementation is intended for **educational and research purposes**.

---

# Overview

Chronic Kidney Disease (CKD) requires early risk detection and transparent clinical decision support. Traditional machine learning models often provide predictions without clear reasoning, limiting their usefulness in real-world clinical settings.

This project implements a **layered AI clinical decision support system** that:

- Predicts CKD risk using machine learning models
- Explains predictions using SHAP feature attribution
- Retrieves supporting clinical guideline evidence
- Generates evidence-grounded reasoning using a large language model
- Applies safety guardrails before presenting outputs
- Supports human-in-the-loop clinician review
- Monitors system reliability and data drift

The system design emphasizes **interpretability, reliability, and responsible AI deployment**.

---
## Quick Navigation

| Section | Description |
|--------|-------------|
| Overview | Project goals and background |
| Project Highlights | High-level system components|
| Key Features | Core capabilities of the system |
| Architecture | High-level system architecture |
| System Pipeline | End-to-end decision pipeline |
| Project Structure | Repository organization |
| Tech Stack | Technologies used |
| Documentation | Detailed technical documents |
| Running the System | Setup and execution instructions |
| Responsible AI | Safety and governance principles |
| Future Improvements | Potential system extensions |


---
## Project Highlights

| Component | Description |
|-----------|-------------|
| CKD Risk Prediction | Machine learning model predicts CKD risk from patient clinical features |
| SHAP Explainability | Identifies which features contribute most to the prediction |
| Clinical Evidence Retrieval | Retrieves relevant clinical guidelines using vector similarity search |
| LLM Clinical Reasoning | Generates evidence-grounded explanations using SHAP insights and retrieved clinical knowledge |
| AI Safety Guardrails | Validates generated explanations using safety filters and consistency checks |
| Human-in-the-Loop Review | Enables clinicians to review model explanations before final decisions |
| Monitoring & Governance | Tracks prediction confidence, system reliability, and data drift |
---


# Key Features

### CKD Risk Prediction
Machine learning model predicts CKD risk from patient clinical features.

### Explainable AI
SHAP feature attribution explains which variables contribute to model predictions.

### Evidence Retrieval (RAG)
Clinical guideline evidence is retrieved using vector similarity search.

### LLM Clinical Reasoning
An LLM synthesizes explanations using both SHAP insights and retrieved clinical evidence.

### AI Safety Guardrails
Safety filters, clinical disclaimers, and reasoning validation ensure responsible outputs.

### Human-in-the-Loop Review
Clinicians can review explanations and override system decisions when necessary.

### Monitoring & Governance
Prediction confidence tracking, drift detection, and system observability ensure reliability.

---

# Architecture

<p align="center">
  <img src="docs/architecture/System_architecture.png" width="900">
</p>

### Architecture Layers Summary

| Layer | Responsibility |
|------|----------------|
| Presentation Layer | Streamlit dashboard for clinician interaction |
| API Layer | FastAPI gateway handling request routing |
| Orchestration Layer | AI pipeline controller managing prediction workflow |
| AI Intelligence Layer | CKD prediction model, SHAP explainability, RAG retrieval, and LLM reasoning |
| Data & Knowledge Layer | Model registry, SHAP store, vector index, and clinical knowledge base |
| Monitoring & Governance | Model monitoring, data drift detection, and system observability |
>
> The architecture prioritizes **interpretability, reliability, and clinical safety**.

### Detailed Architecture Documentation

- **System Architecture Overview**  
  [docs/architecture/architecture_overview.md](docs/architecture/architecture_overview.md)

---


# System Pipeline

The orchestration controller executes the following pipeline:

Patient Input → Risk Prediction → SHAP Explainability → Evidence Retrieval → LLM Clinical Reasoning → Guardrail Validation → Clinician Review

Each stage ensures predictions remain **interpretable, evidence-grounded, and safety validated** before reaching clinicians.

---

# System Capabilities

This system integrates multiple AI capabilities to provide reliable and interpretable clinical decision support.

### Explainable Machine Learning
The CKD prediction model provides interpretable outputs using **SHAP feature attribution**, allowing clinicians to understand which clinical variables influence predictions.

### Retrieval-Augmented Clinical Evidence
Relevant clinical guideline evidence is retrieved using **vector similarity search**, enabling evidence-grounded reasoning.

### LLM Clinical Reasoning
A large language model synthesizes explanations using both **SHAP insights and retrieved clinical evidence**.

### AI Safety Guardrails
Generated outputs are validated through safety filters, reasoning consistency checks, and clinical disclaimers.

### Human-in-the-Loop Decision Support
Clinicians can review system explanations and override system decisions when necessary.

### Monitoring and Governance
The system tracks prediction confidence, monitors calibration stability, and detects data drift to maintain reliability.

---


# Project Structure

```text
clinical-ai-system/
│
├── api/                # FastAPI endpoints and API gateway
├── configs/            # Configuration files
├── data/               # Dataset and preprocessing artifacts
│
├── genai/              # Generative AI pipeline
│   ├── controller/     # Pipeline orchestration
│   ├── explainability/ # SHAP explanation logic
│   ├── retrieval/      # Clinical evidence retrieval (RAG)
│   ├── llm/            # LLM reasoning components
│   ├── guardrails/     # Safety validation
│   ├── evaluation/     # Model evaluation
│   └── prompts/        # Prompt templates
│
├── models/             # Machine learning models
├── services/           # Core prediction services
├── tests/              # System tests
│
├── docs/
│   ├── architecture/
│   │   ├── architecture_overview.md
│   │   └── System_architecture.png
│   │
|   ├── failure_modes.md
|   ├── deployment_staretgy.md
│   ├── model_card.md
│   └── data_shift.md
│
├── system_design.md    # Detailed system design
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Tech Stack

### Machine Learning
- Python
- Scikit-learn
- SHAP

### Generative AI
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

### Backend
- FastAPI

### Frontend
- Streamlit

### Infrastructure
- Vector similarity search for clinical evidence retrieval
- Monitoring and observability components

---

# Key System Design Decisions

This project follows several design principles commonly used in production AI systems.

### Explainability First
Clinical AI systems require interpretability.  
The model integrates **SHAP feature attribution** so clinicians can understand the factors influencing predictions.

### Evidence-Grounded Reasoning
LLM reasoning is combined with **retrieved clinical guideline evidence**, reducing hallucination risk and improving reliability.

### Modular AI Services
The architecture separates prediction, explainability, retrieval, reasoning, and guardrails into **independent services**, enabling easier maintenance and scalability.

### Human-in-the-Loop Validation
Clinical decisions require expert oversight.  
The system allows clinicians to review explanations and override system outputs.

### Safety and Guardrails
AI-generated explanations pass through **validation layers** to ensure safety, consistency, and appropriate disclaimers.

### Monitoring and Governance
Model predictions and system performance are monitored through:

- Prediction confidence tracking
- Data drift detection
- System observability

These components help ensure **long-term reliability of the AI system**.

---

# Documentation

Detailed documentation for the system is available in the `docs` directory.

| Document | Description |
|--------|-------------|
| Architecture Overview | High-level system architecture |
| Model Card | Model behavior, evaluation, and limitations |
| Failure Modes | Identified system failure scenarios and mitigation strategies |
| Deployment Strategy | Design-level deployment architecture, services, and scaling considerations |
| Data Shift | Data drift monitoring and mitigation |
| System Design | Detailed design decisions and implementation |

Links:

- **System Architecture Overview**  
  → [Open](docs/architecture/architecture_overview.md)

- **Model Card**  
  → [Open](docs/model_card.md)

- **Data Drift Monitoring**  
  → [Open](docs/data_shift.md)

- **Detailed System Design**  
  → [Open](system_design.md)

- **Failure Modes and Safety**  
  → [Open](docs/failure_modes.md)

- **Deployment Strategy**  
  → [Open](docs/deployment_strategy.md)
  
---

# Running the System

Follow these steps to run the Clinical AI Decision Support System locally.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
### 2. Setup Environment Variables

Copy the example environment file and update it with your API key:

```bash
cp .env.example .env

```env
GROQ_API_KEY=your_api_key_here
```

### 3. Start the Backend API (FastAPI)

```bash
uvicorn api.main:app --reload
```

This starts the FastAPI server responsible for routing requests to the AI services.

### 4. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The Streamlit dashboard provides the interface for interacting with the CKD risk prediction system.


---

<details>
<summary><b>Responsible AI Considerations</b></summary>

This project incorporates several responsible AI principles to ensure safe and interpretable clinical decision support.

### Explainability
All predictions include **SHAP explanations** so clinicians can understand which clinical features influence the model output.

### Evidence-Grounded Reasoning
LLM explanations are supported by **retrieved clinical guideline evidence**, helping reduce hallucinations and improve reliability.

### Human Oversight
Clinicians remain responsible for reviewing predictions and explanations through a **human-in-the-loop decision process**.

### Safety Guardrails
Generated reasoning passes through **validation layers** that enforce safety checks and clinical disclaimers.

### Monitoring
The system monitors:

- Prediction confidence
- Model calibration
- Data drift
- Pipeline reliability

These mechanisms help maintain long-term stability of the AI system.

</details>
---

<details>
<summary><b>Future Improvements</b></summary>

Potential future extensions include:

- Integration with real Electronic Health Record (EHR) systems
- Continuous model retraining pipelines
- Federated learning across hospitals
- Real-time clinical alert systems

</details>

---

# Disclaimer

This project is a **research and educational implementation** of an AI-powered clinical decision support system.

It is **not intended for real medical diagnosis or treatment decisions**.