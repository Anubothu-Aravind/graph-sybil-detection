# GraphShield: Label Propagation Algorithm Documentation

## 1. Overview & Problem Formulation

In a social network represented as an undirected graph $G = (V, E)$, Sybil accounts represent malicious or artificial identities created by an adversary.

Let:
- $V = \{v_1, v_2, \dots, v_n\}$ be the set of $n = |V|$ nodes (user accounts).
- $E \subseteq V \times V$ be the set of $m = |E|$ edges (social relationships).
- $A \in \{0, 1\}^{n \times n}$ be the graph adjacency matrix, where $A_{ij} = 1$ if $(v_i, v_j) \in E$, and $A_{ij} = 0$ otherwise.
- $D = \text{diag}(d_1, \dots, d_n)$ be the diagonal degree matrix, where $d_i = \sum_{j=1}^n A_{ij}$.

We observe ground-truth class labels for a small subset of nodes $L \subset V$, partitioned into:
- Class $0$: Normal (Honest) account.
- Class $1$: Sybil (Malicious) account.

The vast majority of nodes $U = V \setminus L$ are **unlabeled**. The goal of GraphShield is to compute posterior class beliefs $\hat{y}_i \in [0, 1]$ indicating the probability that node $v_i \in U$ is a Sybil account.

---

## 2. Mathematical Formulation

### 2.1 Initial Belief Matrix $Y^{(0)}$

We construct an initial label matrix $Y^{(0)} \in \mathbb{R}^{n \times 2}$, where the columns correspond to classes $[\text{Normal}, \text{Sybil}]$:

$$
Y_{i}^{(0)} = 
\begin{cases}
[1.0, 0.0] & \text{if } v_i \in L \text{ and } y_i = 0 \text{ (Known Normal)}, \\
[0.0, 1.0] & \text{if } v_i \in L \text{ and } y_i = 1 \text{ (Known Sybil)}, \\
[0.5, 0.5] & \text{if } v_i \in U \text{ (Unlabeled Account)}.
\end{cases}
$$

### 2.2 Propagation Matrix Operators

GraphShield supports two graph diffusion operators:

1. **Random-Walk Transition Matrix ($P_{rw}$):**
   $$P_{rw} = D^{-1} A$$
   where each entry $P_{ij} = \frac{A_{ij}}{d_i}$ denotes the probability of transitioning from node $v_i$ to neighbor $v_j$ in a standard discrete random walk. For isolated nodes ($d_i = 0$), a self-loop is added ($P_{ii} = 1$) to preserve row-stochasticity ($\sum_j P_{ij} = 1$).

2. **Symmetric Normalized Laplacian Operator ($S$):**
   $$S = D^{-1/2} A D^{-1/2}$$
   where $S_{ij} = \frac{A_{ij}}{\sqrt{d_i d_j}}$. This normalizes by degrees of both endpoints, commonly used in spectral clustering and Zhou et al.'s *Learning with Local and Global Consistency*.

### 2.3 Iterative Propagation Update

At each discrete iteration $t \in \{0, 1, \dots, T - 1\}$, label information propagates according to:

$$
Y^{(t+1)} = \alpha P Y^{(t)} + (1 - \alpha) Y^{(0)}
$$

where:
- $\alpha \in (0, 1)$ is the **propagation coefficient** (teleportation/damping parameter, typically $0.80 \le \alpha \le 0.95$).
- $\alpha P Y^{(t)}$ represents information absorbed from the immediate graph neighborhood.
- $(1 - \alpha) Y^{(0)}$ is the **restarting regularizer**, preventing over-smoothing and preserving alignment with initial evidence.

---

## 3. Clamping Modes

GraphShield provides two clamping behaviors:

### 3.1 Hard Clamping (`clamping=True`)
In hard clamping (Zhu & Ghahramani Harmonic Function formulation), known training seed labels are immutable boundary conditions:

$$
Y_{L}^{(t+1)} = Y_{L}^{(0)} \quad \forall t \ge 0
$$

The labeled nodes act as fixed source/drain terminals for the random walk.

### 3.2 Soft Clamping (`clamping=False`)
In soft clamping (Zhou et al. Label Spreading), the labeled nodes are regularized rather than pinned, allowing noise in the training labels to be moderated by consensus among neighbors:

$$
Y^{(t+1)} = \alpha P Y^{(t)} + (1 - \alpha) Y^{(0)}
$$

---

## 4. Convergence Analysis

Since $P$ is row-stochastic, its spectral radius $\rho(P) \le 1$. Because $0 < \alpha < 1$, the linear operator $\alpha P$ is a strict contraction mapping with spectral radius $\rho(\alpha P) = \alpha < 1$.

By the Banach Fixed-Point Theorem:
$$
\lim_{t \to \infty} Y^{(t)} = (1 - \alpha) (I - \alpha P)^{-1} Y^{(0)}
$$

The iteration converges geometrically at rate $O(\alpha^t)$.

**Stopping Criterion:**
Propagation halts when either:
1. Convergence tolerance is reached:
   $$\|Y^{(t+1)} - Y^{(t)}\|_F < \text{tol} \quad (\text{default } 10^{-4})$$
2. Iteration limit is exceeded: $t \ge \text{max\_iter} \quad (\text{default } 100)$.

---

## 5. Output Probabilities and Decision Rule

After convergence at step $T$, posterior class scores are normalized across each row:

$$
p_i(\text{Sybil}) = \frac{Y_{i, 1}^{(T)}}{Y_{i, 0}^{(T)} + Y_{i, 1}^{(T)}} \in [0.0, 1.0]
$$

Given a configurable decision threshold $\tau \in [0.0, 1.0]$ (default $\tau = 0.50$):

$$
\hat{y}_i = 
\begin{cases}
1 & \text{if } p_i(\text{Sybil}) \ge \tau \quad (\text{Flagged as Sybil}), \\
0 & \text{if } p_i(\text{Sybil}) < \tau \quad (\text{Classified as Normal}).
\end{cases}
$$

---

## 6. Algorithmic Complexity

- **Memory / Space Complexity:** $O(|V| + |E|)$ utilizing CSR sparse matrix storage.
- **Time Complexity per Iteration:** $O(|E| \cdot C)$ where $C = 2$ classes. Sparse matrix-vector product $P Y$ scales linearly with the number of graph edges.
- **Hardware Acceleration:** Supports direct PyTorch Apple Metal Performance Shaders (MPS) execution on Apple Silicon (M1/M2/M3/M4) unified memory, with automatic fallback to CPU.
