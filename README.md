# Optimizing Incident Management Systems
## An Operations Research & Machine Learning Approach

**Authors:**
- **Divagar S** (CB.SC.U4AIE23223)
- **Abhishek Pandey** (CB.SC.U4AIE23205)
- **Prem** (CB.SC.U4AIE23241)

---

### 1. Executive Summary & Problem Context
In IT incident support systems, inefficient ticket routing and arbitrary staffing allocations create compounding operational bottlenecks:
- **Severe Resolver Workload Imbalance:** Unequal ticket assignments among 9 active resolvers cause acute resolver burnout (e.g., *Sam* handled over 49,000 actions, while *David* handled 26,000, and *Ava* handled only 455).
- **High SLA Failure Rate:** Over **51.3%** of all customer incidents violated Service Level Agreements (SLAs) in the baseline log.
- **Depressed Customer Satisfaction:** Baseline CSAT averaged **3.25 / 5.0**, severely damaged by prolonged queue delays.
- **Structural Tier Misallocation:** Tier 1 and Tier 2 resolvers operated under unstable traffic conditions ($\rho > 1.0$), while Tier 3 resolvers remained over 80% idle.

This project delivers a mathematical Operations Research and Machine Learning solution combining:
1. **Model 1: Integer Linear Programming (ILP)** for real-time prescriptive ticket assignment.
2. **Model 2: Open Jackson Queueing Network ($M/M/c$)** with **SimPy discrete-event simulation** for multi-tier staffing optimization.
3. **An Interactive Web Application (Streamlit)** allowing users to select and benchmark both models side-by-side.

---

### 2. Dataset Overview
- **Source:** Process Mining Event Log (`Incident_Management_CSV.csv`)
- **Total Logged Events:** 242,901
- **Unique Incident Cases:** 31,588
- **Active Resolvers:** 9 resolvers distributed across 3 support tiers:
  - **Level 1 (L1):** David, Sam
  - **Level 2 (L2):** Emma, Michael, Sarah
  - **Level 3 (L3):** Ava, James, Olivia, William
- **Process Variants:** 13 distinct workflow variants
- **Priorities:** High, Medium, Low
- **Channels:** Website, Email, App, Phone

---

### 3. Machine Learning Predictive Pipeline
Trained on **57,418** reconstructed task traces:
- **Service Time Predictor ($S_{i,k}$):**
  - Model: Random Forest Regressor ($R^2 = 0.741$, $\text{MAE} = 0.899$ hours)
  - Predicts expected active handling time given ticket priority, issue type, channel, resolver identity, and tier.
- **Customer Satisfaction Predictor ($CSAT_{i,k}$):**
  - Model: Regularized Ridge Regressor
  - Estimates expected user rating (1–5) based on service duration, tier, and ticket attributes.

---

### 4. Model 1: Integer Linear Programming (ILP) Assignment
Formulates the tactical dispatching problem solved at decision epochs:

$$\min_{x_{i,k} \in \{0, 1\}} \quad Z = \alpha \sum_{k \in K} \sum_{i \in I} P_k \cdot S_{i,k} \cdot x_{i,k} - \beta \sum_{k \in K} \sum_{i \in I} CSAT_{i,k} \cdot x_{i,k} + \gamma \sum_{i \in I} d_i$$

**Subject to:**
1. **Single Assignment:** $\sum_{i \in I_k} x_{i,k} = 1 \quad \forall k \in K$
2. **Resolver Capacity:** $W_i^{\text{current}} + \sum_{k \in K} x_{i,k} \le C_i \quad \forall i \in I$
3. **Workload Fairness & Deviation ($d_i$):** Linearized penalization of absolute deviation from tier mean load $\bar{W}$.
4. **Tier Routing Constraint:** $x_{i,k} = 0$ for resolvers outside the ticket's tier.

**Results:**
- Reduces Workload Gini inequality index by **20.7% to 47.0%**.
- Fast, global optimality via SciPy HiGHS solver in $< 15$ ms.
- Reduces SLA violations to $36.8\%$.

---

### 5. Model 2: Jackson Queueing Network & Staffing Optimization
Models the 3 tiers as an open network of $M/M/c$ queues:
- **Traffic Equations:**
  - $\Lambda_1 = \lambda = 3.593$ tickets/h
  - $\Lambda_2 = \gamma_2 + P_{1 \to 2} \Lambda_1 = 2.491$ tickets/h ($P_{1 \to 2} = 58.47\%$)
  - $\Lambda_3 = P_{2 \to 3} \Lambda_2 = 0.368$ tickets/h ($P_{2 \to 3} = 14.76\%$)
- **Erlang-C Metrics:**
  - Offered Load: $a_j = \Lambda_j / \mu_j$
  - Probability of waiting: $C(c_j, a_j) = P(W_q > 0)$
  - Expected queue delay: $W_{q, j} = \frac{C(c_j, a_j)}{c_j \mu_j - \Lambda_j}$
- **Staffing Optimization:**
  $$\min_{c_1, c_2, c_3} \sum_{j=1}^3 \text{Cost}_j \cdot c_j \quad \text{s.t.} \quad \rho_j \le 0.80, \quad W_{q, j} \le \text{SLA}_j$$
- **Optimal Staffing Recommendation:**
  - Tier 1: Increase from 2 to **7 resolvers** ($\rho_1 = 0.73$, $W_{q, 1} = 0.27$ h)
  - Tier 2: Increase from 3 to **6 resolvers** ($\rho_2 = 0.69$, $W_{q, 2} = 0.29$ h)
  - Tier 3: Reallocate from 4 to **3 resolvers** ($\rho_3 = 0.25$, $W_{q, 3} = 0.05$ h)
- **Validation:** SimPy stochastic simulation over 1,500 hours confirms theoretical Erlang-C wait times within $\pm 0.03$ hours.

---

### 6. Comparative Benchmark

| Approach | SLA Compliance (%) | Avg Resolution Time (h) | Workload Gini Index | CSAT Rating (1–5) | Operational Scope |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Historical Baseline (Event Log)** | 48.7% | 15.03 h | 0.545 | 3.26 | Status quo: High bottlenecks & burnout |
| **Model 1: ILP Prescriptive Assignment** | 63.2% | 12.33 h | 0.289 | 3.22 | Real-time optimal dispatching & workload fairness |
| **Model 2: Jackson Queueing & Staffing** | 92.1% | 3.27 h (wait: 0.48h) | 0.180 | 3.85 | Strategic capacity planning & bottleneck elimination |
| **Integrated Synergy (Model 1 + Model 2)** | **94.8%** | **2.85 h** | **0.143** | **4.15** | Strategic staffing capacity + Real-time ILP dispatching |

---

### 7. Interactive Web Application
The Streamlit interactive application provides 4 comprehensive views:
1. **📊 Executive Overview & Event Log:** Process mining diagnostic dashboard, variant breakdown, resolver burnout charts.
2. **🎯 Model 1: ILP Ticket Assignment:** Live tuning of $\alpha, \beta, \gamma$, batch sizing, resolver capacity constraints, and live HiGHS optimization solver execution.
3. **⏱️ Model 2: Jackson Queueing & Staffing:** Parameter tuning of arrival rate $\lambda$, tier SLA thresholds, Erlang-C analytical metrics, trade-off curves, and live SimPy simulation runner.
4. **⚖️ Dual-Model Comparative View:** Select any 2 models to display head-to-head metrics, grouped bar charts, and comparative tables.

#### Launching the Application:
```bash
# Run with the virtual environment
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.
