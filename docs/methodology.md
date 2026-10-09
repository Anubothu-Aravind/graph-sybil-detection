# GraphShield: Methodology & Academic Context

## 1. The Sybil Attack in Social Computing

First formulated by John Douceur (2002), a **Sybil Attack** is a subversion of distributed networks wherein a malicious entity creates numerous pseudonymous identities to exert disproportionate influence.

In modern social platforms, Sybil clusters are deployed to:
- Coordinate misinformation and disinformation campaigns.
- Manipulate social recommendation and trending algorithms.
- Amplify spam and execute financial fraud.
- Subvert decentralized voting, consensus, or reputation systems.

---

## 2. Graph-Theoretic Foundations of Sybil Defense

Traditional content-based or IP-based detection methods are easily circumvented by modern botnets utilizing residential proxies and LLM-generated human-like text. Graph-based defenses instead leverage the **topological constraints** of real social relationships.

### 2.1 The Bounded Quotient Cut Assumption
Pioneered in seminal literature such as *SybilGuard* (Yu et al., SIGCOMM 2006), *SybilLimit* (Yu et al., S&P 2008), and *SybilRank* (Cao et al., NSDI 2012):
1. **Fast-Mixing Honest Region:** Honest users form organic social bonds with friends, colleagues, and communities, resulting in high clustering coefficients and short average path lengths. Random walks in the honest region mix rapidly.
2. **Constrained Attack Edges (Quotient Cut):** An adversary can create an arbitrary number of Sybil nodes and connect them internally with arbitrary density. However, establishing social ties (friendships, follows, mutual trusts) with genuine honest accounts requires human social engineering or account takeover. Hence, the number of bridging edges crossing between honest and Sybil communities is bounded.

### 2.2 Why Semi-Supervised Label Propagation Works
Label propagation simulates a constrained random walk with restart. Seeds known to be honest diffuse positive trust through the densely connected honest region. Because the attack edges connecting the honest region to the Sybil cluster represent a bottleneck, only a small fraction of the honest walk crosses into the Sybil cluster. Conversely, Sybil seeds propagate malicious beliefs throughout the tightly coupled botnet.

---

## 3. Rigorous Evaluation Protocol: Zero Test Leakage

A common flaw in academic ML implementations is data leakage—allowing test information to influence preprocessing, graph construction, or model tuning.

GraphShield enforces a strict evaluation boundary:
1. **Partitioning:** Ground-truth nodes $V_{gt}$ are partitioned into training seeds $V_{train}$ and held-out test nodes $V_{test}$ such that $V_{train} \cap V_{test} = \emptyset$.
2. **Masking:** During the graph construction and propagation phases, the model receives label inputs only for $V_{train}$. All nodes in $V_{test}$ (as well as unlabeled nodes) are masked with label value $-1$ and initialized with neutral prior beliefs $[0.5, 0.5]$.
3. **Withheld Assessment:** Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Specificity) are calculated **strictly** on $V_{test}$. Neither test labels nor test features are ever exposed to the propagation operator.

---

## 4. Evaluation Metrics

Because Sybils frequently constitute a minority class (class imbalance), accuracy alone is misleading:

- **Accuracy:** Overall correctness across all test predictions: $\frac{TP + TN}{TP + TN + FP + FN}$.
- **Precision:** $\frac{TP}{TP + FP}$ — proportion of flagged accounts that are truly Sybils (critical to minimize false accusations against honest users).
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — proportion of actual Sybils successfully caught.
- **F1-Score:** Harmonic mean of Precision and Recall: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$.
- **ROC-AUC:** Area under Receiver Operating Characteristic curve across all potential decision thresholds $\tau \in [0, 1]$.

---

## 5. Critical Limitations and Real-World Considerations

Academic integrity requires acknowledging where graph label propagation fails or requires supplementary defenses:

1. **Adversarial Edge Infiltration:** If attackers successfully induce hundreds of honest users to follow them (e.g., via follow-back schemes, compromised influencer accounts, or automated spam requests), the quotient cut expands, and Sybil detection degrades.
2. **Seed Bias:** Label propagation is highly sensitive to seed placement. If known Sybil seeds are only available in one specific bot cluster, distant unseeded Sybil clusters may evade detection.
3. **Correlation vs. Causality:** A high Sybil score indicates structural proximity to known malicious clusters. It does **not** constitute legal proof of malicious intent. Scores should serve as risk indicators for human review or rate-limiting.
