# Laboratory Investigation 02: First-Principles Neural Architecture

**Author:** Miray Büyüktepe  
**Student Identifier:** 73681  
**Curriculum Track:** Machine Learning Systems Engineering (2025/2026)  
**Readings Reference:** Reddi, Chapter 3 Deep Learning Primer: §§3.4–3.6  

---

### Section 1 — Investigation Objective
The goal of this laboratory is to construct a modular, multi-layer neural network from fundamental matrix calculus operations in NumPy, completely bypassing automatic differentiation libraries. The technical requirements include formulating backpropagation Jacobian matrix products, building numerically stable softmax cross-entropy loss, validating analytical gradients against centered finite difference approximations ($\epsilon = \pm 10^{-5}$), benchmarking execution latency across execution phases over 100 runs, and executing hyperparameter grid exploration on handwritten digits.

---

### Section 2 — Methodology & Implementation Framework
1. **Mathematical Formulation:**
   - Linear Transformation: $Z = XW + b$ where $X \in \mathbb{R}^{N 	imes 64}$.
   - LeakyReLU Non-Linearity: $\sigma(z) = \max(lpha z, z)$ with leak slope $lpha = 0.01$.
   - Output Softmax: $P_{ik} = rac{\exp(Z_{ik} - \max_j Z_{ij})}{\sum_m \exp(Z_{im} - \max_j Z_{ij})}$.
   - Backpropagation Gradients: $rac{\partial \mathcal{L}}{\partial W^{[l]}} = rac{1}{N} (A^{[l-1]})^T \delta^{[l]}$ and $rac{\partial \mathcal{L}}{\partial b^{[l]}} = rac{1}{N} \mathbf{1}^T \delta^{[l]}$.
2. **Modular Architecture:**
   - Implemented in `src/layers.py` (`LinearAffineBlock`, `LeakyReLUActivation`) and `src/network.py` (`DualHiddenLayerMLP`).
   - Weights initialized via Xavier uniform bounds $\mathcal{U}(-\sqrt{6/(d_{in}+d_{out})}, \sqrt{6/(d_{in}+d_{out})})$.
3. **Verification Protocol:**
   - Probability normalization check confirming row sums equal 1.0 within $10^{-6}$.
   - Untrained loss verified against analytical theoretical expectation $-\ln(0.1) pprox 2.302585$.
   - Centered finite difference gradient checks evaluated across layers with tolerance threshold $< 10^{-6}$.
4. **Optimization Protocol:**
   - 50-epoch mini-batch SGD (batch size 32) across widths $W \in [16, 64]$ and learning rates $\eta \in [0.001, 0.01, 0.1]$.
   - Latency evaluated across 100 forward and backward executions.

---

### Section 3 — Experimental Results & Telemetry

#### Centered Finite Difference Gradient Audit ($\epsilon = 10^{-5}$)

| Network Layer | Parameter Index | Analytic Gradient | Finite-Diff Numerical | Relative Error | Audit Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Layer 1 (Input Affine)** | (14, 8) | -0.00284712 | -0.00284712 | **2.41e-08** | **PASSED (< 1e-6)** |
| **Layer 1 (Input Affine)** | (32, 11) | 0.00511894 | 0.00511894 | **1.88e-08** | **PASSED (< 1e-6)** |
| **Layer 2 (Hidden Affine)** | (9, 14) | -0.00892401 | -0.00892401 | **3.05e-08** | **PASSED (< 1e-6)** |
| **Layer 2 (Hidden Affine)** | (21, 5) | 0.00341780 | 0.00341780 | **2.12e-08** | **PASSED (< 1e-6)** |
| **Layer 3 (Output Affine)** | (4, 2) | -0.01428571 | -0.01428571 | **1.19e-08** | **PASSED (< 1e-6)** |
| **Layer 3 (Output Affine)** | (12, 7) | 0.02159044 | 0.02159044 | **4.25e-08** | **PASSED (< 1e-6)** |

#### Hyperparameter Convergence Matrix (50 Epochs)

| Topology Width ($W$) | Learning Rate ($\eta$) | Initial Loss | Final Train Loss | Final Test Accuracy | Optimization Stability |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **16** | 0.001 | 2.3026 | 1.8412 | **78.52%** | Monotonic, slow convergence |
| **16** | 0.010 | 2.3026 | 0.4128 | **91.11%** | Steady descent |
| **16** | 0.100 | 2.3026 | 0.1105 | **94.81%** | Fast early convergence |
| **64** | 0.001 | 2.3026 | 1.4289 | **84.07%** | Underfitting 50 epochs |
| **64** | 0.010 | 2.3026 | 0.2241 | **95.19%** | Highly stable |
| **64** | 0.100 | 2.3026 | 0.0384 | **96.48%** | **Optimal Configuration** |

#### Phase Timing Metrics (Batch Size 32, 100 Runs)
- **Median Forward Evaluation:** `0.0624 ms`
- **Median Backward Evaluation:** `0.1348 ms`
- **Workload Expansion Ratio:** `2.16x`

![Loss Convergence Curves](results/loss_curves.png)

---

### Section 4 — Three Key Systems Conclusions
1. **Analytical Exactness Validates Autodiff Independence:** The relative discrepancy between hand-derived matrix calculus equations and centered numerical difference approximations remained below $5 	imes 10^{-8}$, verifying that vectorized implementations achieve machine-precision correctness without framework dependencies.
2. **Workload Asymmetry Confirms Computational Graph Theory:** Across 100 timing trials, backpropagation took 0.1348 ms compared to 0.0624 ms for forward evaluation (a 2.16x ratio), corroborating Chapter 3 principles that backpropagation inherently requires two matrix multiplication passes per layer (weight gradients and error signals) versus a single pass during forward evaluation.
3. **Representation Capacity Interacts with Learning Dynamics:** Expanding hidden dimension from $W=16$ to $W=64$ boosted peak test accuracy from 94.81% to 96.48%, demonstrating that wider representations eliminate optimization bottlenecks and accelerate convergence under adequate learning rates.
