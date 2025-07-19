# 🏢 Customer-based Instances

An important variant of scheduling problems arises when the objective includes weighted criteria, such as completion time or tardiness.
These weights, typically denoted $w_j$ for job $j$, reflect the cost per unit of the metric being minimized.
For instance, the classic objective $\gamma = \sum_j w_j T_j$ corresponds to paying $w_j$ units for each unit of tardiness for job $j$.
While weights are often treated as known constants, they may instead reflect latent job characteristics.
For example, in our setting, a customer-specific importance weight affects scheduling decisions, making decisions involving high-priority customers towards minimizing delay at the expense of lower-priority ones, is not explicitly encoded in the observed instance.
This introduces a new layer of complexity: although the structure of the scheduling problem remains the same, the weights must now be inferred from proxy information.
Crucially, this couples otherwise independent instances via a shared latent structure, turning the problem into a more general setting with contextual latent variables.

## Assigning Weights to customers

In this work, we explore a scenario where job weights are not explicitly known and instead depend on the customer associated with each job.
That is, every job $j$ belongs to a customer $c \in \mathcal{C}$, and the associated weight $w_j$ is determined solely by $c$.
Our goal is to learn or recover the customer-wise weights $\{w_c\}_{c \in \mathcal{C}}$ from observed scheduling behavior, without assuming direct knowledge of $\omega_j$.
To simulate this setting, we propose a method to reconstruct benchmark scheduling instances such that latent customer weights drive scheduling decisions, while preserving the same optimal solution.
Rather than generating new instances from scratch, we re-purpose existing instances by mapping job weights to customers using a stochastic assignment that reflects real-world patterns, such as uneven customer distributions.

Each customer $c \in \mathcal{C}$ is assigned a latent presence score $s_c > 0$, such that $\sum_{c \in \mathcal{C}} s_c = 1$, reflecting the proportion of jobs they own across all instances.
We assume the original instance includes jobs with known weights from a discrete set $w_j \in W \subset \mathbb{N}_+$ and we define a mapping $\nu:\mathcal{J} \to \mathcal{C}$ from jobs to customers, such that:
    
- All jobs assigned to the same customer share the same weight,
    $$\nu(i) = \nu(j) \implies w_i = w_j.$$

- The proportion of jobs owned by each customer approximates its presence score,
    $$|\{j \in \mathcal{J} : \nu(j) = c\}| \approx s_c \cdot|\mathcal{J}|.$$

The assignment is made using a stochastic procedure: we first sample a Gumbel-noised permutation of customers ordered by presence score, then greedily assign customers to weight classes until the empirical job distribution over weights is satisfied.
This is done by assigning customers appearing first in the permutation to lower weights, in order, until every customer is assigned to a weight value.
Finally, we normalize the weights ${\omega_c}$ onto the simplex $\Delta^{|\mathcal{C}|}$, allowing continuous-valued policies and facilitating comparisons against rational or soft weight estimates.
This formulation supports both parameter interpretability and flexible modeling, enabling the use of continuous loss functions or soft-inference techniques without restricting integral weights.
The original optimal values and solutions remain unchanged, as normalization and job assignment only adjust the job's weight by a scaling factor.
Therefore, we can build a test dataset of instances with known optimal solutions, alleviating the requirement of solving hard instances to optimality, limiting our evaluations.
