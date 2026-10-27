# Lab 01: Environment and First System Measurements

**Student Researcher:** Miray Büyüktepe (Student ID: 73681)  
**Readings:** Reddi, Chapters 1, 2, and 12  

---

## 1. Goal
This module establishes a verified, isolated Python 3.11 machine learning engineering environment, trains linear and non-linear baseline classifiers on the Breast Cancer Wisconsin diagnostic dataset using stratified evaluation, benchmarks training latency, per-sample inference latency, resident memory footprint, and model disk consumption, and constructs a deployment feasibility matrix across Cloud, Edge, Mobile, and TinyML hardware platforms.

---

## 2. Method
1. **Isolated Runtime Verification:** Pinned dependencies according to syllabus standards and verified runtime package versions into `results/versions.txt`.
2. **Reproducible Modeling:** Ingested the Wisconsin diagnostic dataset (569 records, 30 continuous features) and stratified 70/30 with `random_state=42`.
3. **Model Selection:**
   - Linear Model: `LogisticRegression(max_iter=1000, random_state=42)`
   - Tree Ensemble: `RandomForestClassifier(n_estimators=100, random_state=42)`
4. **Systems Measurement Protocol:**
   - Training latency: 1 initial warm-up followed by 5 repetitions; recorded median time.
   - Inference latency: 100 consecutive single-sample evaluation passes; recorded median time.
   - Model size: Serialized estimators with `joblib.dump` to measure on-disk storage.
   - Process memory: Measured peak RSS memory consumption with `psutil`.
5. **Deployment Suitability Matrix:** Compared empirical footprints against standardized resource tiers.

---

## 3. Results Table or Plot

### Empirical Systems Profiling Table

| Evaluation Metric | Linear Classifier (`LogisticRegression`) | Tree Ensemble (`RandomForestClassifier`) | Profiling Methodology |
| :--- | :---: | :---: | :--- |
| **Test Accuracy** | **0.9415** (94.15%) | **0.9357** (93.57%) | Stratified 70/30 Test Partition |
| **Training Duration (Median)** | **99.23 ms** | **132.66 ms** | Median of 5 runs (post-warmup) |
| **Single-Sample Latency (Median)** | **0.1330 ms** | **5.6562 ms** | Median of 100 single-sample passes |
| **Serialized Binary Footprint** | **1,103 bytes** (1.08 KB) | **290,889 bytes** (284.07 KB) | On-disk `joblib` file |
| **Process Resident Memory (RSS)** | **~430.4 MB** | **~430.4 MB** | Host OS process RSS |

### Hardware Deployment Tier Feasibility

| Target Tier | Memory Budget | Latency Budget | Storage Budget | Logistic Regression | Random Forest | Systems Justification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cloud** | >= 1 GB | <= 100 ms | <= 500 MB | **Eligible** | **Eligible** | Negligible infrastructure consumption; both run effortlessly. |
| **Edge Gateway** | 256-1024 MB | <= 50 ms | <= 50 MB | **Eligible** | **Eligible** | Sub-millisecond latency supports high-throughput edge streaming. |
| **Mobile Device** | 64-256 MB | <= 20 ms | <= 10 MB | **Eligible** | **Eligible** | Both models consume < 284.1 KB disk, well below mobile caps. |
| **TinyML Microcontroller** | <= 256 KB | <= 10 ms | <= 100 KB | **Eligible** (1.08 KB) | **Ineligible** (284.07 KB) | Linear weights fit inside embedded SRAM/Flash; 100 trees exceed 100 KB cap. |

---

## 4. Three Conclusions in Your Own Words
1. **Linear Predictors Offer Optimal Efficiency on Low-Dimensional Tabular Data:** `LogisticRegression` achieved higher test generalization (94.15%) than the 100-tree ensemble while consuming merely 1.08 KB of disk storage, demonstrating that complex models do not guarantee accuracy improvements on structured diagnostic data.
2. **Decision Tree Depth Imposes Extreme Serialization Overhead:** The ensemble structure requires 264x more storage capacity (284.1 KB vs 1.08 KB) and higher per-sample latency due to traversing hundreds of sequential branching nodes.
3. **Median Measurement Provides Deterministic Latency Benchmarks:** Repeating single-sample inference 100 times and extracting the median filtered out operating system scheduling noise, establishing reliable baselines for embedded capacity planning.
