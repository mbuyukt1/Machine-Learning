# Production Machine Learning Systems — Coursework Portfolio

**Lead Researcher:** Miray Büyüktepe  
**Student ID:** 73681  
**Curriculum:** Machine Learning Systems (2025–2026)  
**Reference:** Reddi, *Machine Learning Systems*, Harvard University  

---

## Technical Scope & Architecture

This repository delivers an end-to-end, reproducible laboratory portfolio adhering to industrial MLOps and systems engineering standards:

### Project Directory
1. **`lab01/` — Runtime Profiling & Deployment Tiers:** Benchmarks Logistic Regression vs Random Forest across training duration, single-sample latency, peak RAM, and disk footprint. Evaluates deployment feasibility on TinyML microcontrollers, mobile edge, and cloud.
2. **`lab02/` — First-Principles Neural Network:** Modular NumPy architecture with He initialization, stable softmax cross-entropy, and finite-difference gradient checks ($\epsilon = \pm 10^{-5}$) guaranteeing relative error < 1e-6.
3. **`lab03/` — Architectural Efficiency & Systems Cost:** Multi-model evaluation (MLP, CNN, Vision Transformer) on Fashion-MNIST exploring parameter budgets, hand-derived FLOPs vs `thop` MACs, and inference speed.
4. **`lab04/` — Production Data Quality Pipeline:** Leakage-free tabular ETL pipeline on Adult Census income data featuring zero silent row drops, training-isolated scalers, feature synthesis, and duplicate label noise audits.

## Environment Reproduction
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
