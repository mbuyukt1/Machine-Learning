# Machine Learning Systems Engineering — Coursework Portfolio

**Student:** Miray Büyüktepe  
**Student ID:** 73681  
**Curriculum Textbook:** *Vijay Janapa Reddi, Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems.*  
**Environment Specification:** Python 3.11.8 (Pinned Stack)  

---

## Portfolio Organization

This repository implements the four core laboratory assignments following the standardized structural specification:

```
ml-systems-labs/
├── README.md                      # Primary portfolio index
├── requirements.txt               # Pinned dependencies
├── .gitignore                     # Repository cleanliness rules
├── lab01/                         # Lab 01: Environment & System Measurements (Ch. 1, 2, 12)
│   ├── report.md
│   ├── src/
│   └── results/
├── lab02/                         # Lab 02: Neural Network from First Principles (Ch. 3)
│   ├── report.md
│   ├── src/
│   └── results/
├── lab03/                         # Lab 03: Architectures and Their Costs (Ch. 4)
│   ├── report.md
│   ├── src/
│   └── results/
└── lab04/                         # Lab 04: Data Pipeline and Quality Assurance (Ch. 5-6)
    ├── report.md
    ├── src/
    └── results/
```

---

## Laboratory Modules

| Module | Title | Reddi Textbook | Delivered Evidence | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Lab 01** | **Environment and First System Measurements** | Ch. 1, 2, 12 | Pinned versions, latency distributions, RSS memory, edge/TinyML budget matrix | Verified |
| **Lab 02** | **A Neural Network from Scratch** | Ch. 3 | Matrix calculus chain rule, 10/10 gradient verification, GEMM profiling | Verified |
| **Lab 03** | **Architectures and Their Costs** | Ch. 4 | PyTorch MLP, CNN, Attention; hand FLOPs vs thop, Pareto trade-off curves | Verified |
| **Lab 04** | **Data Pipeline and Quality Engineering** | Ch. 5, 6 | OpenML Adult dataset, zero-leakage preprocessing, feature synthesis, duplicate audit | Verified |

---

## Execution Guide

```bash
# Clone the repository
git clone https://github.com/mbuyukt1/ml-systems-labs.git
cd ml-systems-labs

# Activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run modules
python lab01/src/baseline_experiments.py
python lab02/src/experiment_runner.py
python lab03/src/evaluate_efficiency.py
python lab04/src/pipeline_runner.py
```
