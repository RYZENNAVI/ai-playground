# Module 05: Fine-Tuning — Low-Rank Adaptation From the Assumption Up

> One question runs through this module: **a weight update has millions of entries, so why is it
> safe to represent it with a few thousand?**
>
> Everything here is built around answering that with measurements rather than with a citation:
>
> 1. **The assumption** — a trained update really is nearly low rank, and the same measurement
>    shows that the frozen weight beside it is not
> 2. **The mechanism** — what an adapter is, where it attaches, and what each choice costs
> 3. **The data** — labels generated from rules, so accuracy is a measurement and not an impression
> 4. **The chains** — supervised fine-tuning, reward-driven training, inference-time control, and
>    an adapter on a vision-language model
>
> Seven scripts, all of them run. **Every number below comes from an actual run** on a single
> 12 GB consumer laptop GPU (RTX 5070 Ti Laptop, Blackwell, sm_120). Two of the seven need no GPU
> at all.

---

## 1. When Changing the Model Is the Right Move

### 1.1 Three Causes, Three Routes

A wrong answer has a cause, and the cause decides the route:

| Cause | Route | Why |
| :--- | :--- | :--- |
| **The request was unclear** | Prompt engineering | The capability is there; the instruction was not |
| **The background knowledge is missing** | Retrieval | The model never saw your private material; fetch it and hand it over |
| **The capability is missing** | **Fine-tuning** | It is not in the parameters, so the parameters have to change |

**The ordering is by cost**: a prompt edit changes one line, retrieval changes a document,
fine-tuning needs a GPU, data and time. **Exhaust the cheaper routes first.**

### 1.2 The Useful Test: Will the Knowledge Be Reused

Treat the model as a memory. Two questions decide whether a fact is worth burning into weights:
**will it be used repeatedly, and is it connected to other knowledge?**

| Situation | Route | Reason |
| :--- | :--- | :--- |
| Common knowledge the model lacks, **used repeatedly** | Fine-tune it in | Train once, permanent |
| **One-off** knowledge, unconnected to anything else | Do not fine-tune | It can disturb what the surrounding parameters already encode |
| **Volatile** facts (dates, prices, promotions) | Retrieval | Training cannot keep up with the change rate |
| **Stable** facts (domain fundamentals) | Fine-tune it in | Only stable things earn a permanent slot |

At the other end of the range, anything that changes by the minute needs incremental indexing even
for retrieval — **fine-tuning is not in the conversation for that class of content**.

### 1.3 Three Conditions Before Declaring "Capability Gap"

All three must hold at once:

- Prompt engineering has been tried and **did not help**
- Retrieval augmentation has been tried and **did not help**
- The failure traces to the model's **base capability**

Then: collect domain data → design the training run → evaluate the result.

### 1.4 What This Module Targets

"Fine-tuning" means two different jobs, and conflating them produces absurd cost estimates:

| | **Building a base model** | **Adapting one** |
| :--- | :--- | :--- |
| Goal | General language ability | A narrow domain or a private format |
| Scale | 100B+ parameters | **Adapters on a 0.25B–70B base** |
| Data | General corpora, **trillions of tokens** | **1k–1M** instruction-response pairs from the work itself |
| Cost | Thousand-GPU months | Minutes to days |
| Artefact | Public weights or an API | **An adapter of a few MB, hot-swappable** |

**This module is entirely about the second column.** The artefact being a few megabytes is the
premise behind every trade-off that follows.

---

## 2. Low Rank: Why 1% of the Parameters Is Enough

### 2.1 What Low Rank Means

Picture a 1000 × 1000 table of numbers:

| | |
| :--- | :--- |
| **High rank** | Every row and column carries unique information. There is no shortcut — **all one million numbers have to be stored** |
| **Low rank** | Rows are **strongly related**. If every row is a combination of a handful of basis rows, storing those few rows reproduces the whole table |

> **Rank is the smallest number of independent pieces of information the matrix needs.**
> Lower rank means more redundancy and more compressibility.

A physical analogy: a global temperature report does not need every city minute by minute.
**Latitude and season** are enough to reconstruct most of it.

### 2.2 What "Intrinsically" Low Rank Means

When a pretrained model learns a new task, its weight matrix `W` is adjusted. **That adjustment is
ΔW.** ΔW has the same shape as `W` — a million entries for a 1000 × 1000 layer — but the claim is
that the *effective* change is concentrated in very few directions:

> Decompose a real ΔW and its singular values decay steeply. Most are near zero; a few are large.
> The update that matters lives in the directions of those few.

**The direction of that argument matters.** It is not "we want ΔW to be low rank, so we force it".
It is **"ΔW turns out to be nearly low rank, so representing it that way loses little"**.
Which means the claim is falsifiable — and worth actually testing.

### 2.3 Measuring the Assumption Instead of Citing It

`03_lora_low_rank_hypothesis.py` performs the test. It takes eight attention projections of a
1.5B model and **updates them with no rank constraint at all** — every entry free to move, plain
gradient descent, 40 steps:

```
Updating 8 matrices, 11,010,048 parameters (0.620% of the model)
  step   1  loss 3.4988
  step  40  loss 0.0466
Loss moved 3.4988 -> 0.0466
```

Then it decomposes the resulting ΔW and asks how many directions each share of the energy needs.
**The two control rows are what make this a measurement rather than a demonstration:**

| Matrix (1536 × 1536) | 50% of energy | 90% | 99% | σ₁ / σ₆₄ |
| :--- | ---: | ---: | ---: | ---: |
| **Trained update ΔW** | **3** | **36** | 493 | **22.3×** |
| Frozen weight `W`, same shape | 156 | 612 | 1063 | 1.9× |
| Random noise, same shape and scale | 279 | 783 | 1184 | 1.1× |

**Fast decay is not a property of matrices in general.** The frozen weight sitting right next to
the update needs 156 directions for the same half of its energy; noise needs 279. Only the update
is compressible, and the update is exactly what an adapter has to represent.

The same decomposition prices the rank budget:

| Rank r | Share of the update kept | Dropped | Adapter parameters |
| ---: | ---: | ---: | ---: |
| 1 | 30.82% | 69.18% | 3,072 |
| 4 | 66.61% | 33.39% | 12,288 |
| **8** | **80.09%** | 19.91% | 24,576 |
| **16** | **85.13%** | 14.87% | 49,152 |
| 64 | 92.66% | 7.34% | 196,608 |

⇒ **r = 16 keeps 85% of the update for 2.08% of the matrix.** The conventional choice of 8 or 16
has an empirical basis, and this is it — for a task of this width.

### 2.4 The Same Reading as the Task Widens

Those shares come from one narrow task: eight pairs of a single template, trained until the loss
sits at 0.05. A task that narrow may concentrate the update into fewer directions than a broader
one would, so the script measures that too. Ten short input-output tasks are defined and the tiers
are nested prefixes of them, so moving up a tier only ever adds kinds of work. **Pool size, batch
size, step count and learning rate are held fixed**, every tier restarts from the original
checkpoint, and only the number of task kinds moves:

| Task kinds | Final loss | 50% of energy | 90% | 99% |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0399 | 2.1 | **20.2** | 256.0 |
| 3 | 0.0468 | 2.6 | **25.8** | 321.9 |
| 10 | 0.2520 | 4.1 | **37.6** | 359.1 |

Seven of the eight matrices rise at every tier. **The concentration survives** — even at ten kinds
of task, 90% of the energy still fits in about 38 of 1536 directions, roughly 2.5% — but the count
is not a constant, and a rank read off the truncation table above is a rank for that task.

**One caveat belongs with the last row.** The tiers are matched on steps, not on convergence: tiers
1 and 3 finish at almost the same loss, so the rise between them is clean, while tier 10 ends five
times higher and part of its rise may be that it is less converged rather than that it is broader.
Training to equal loss instead would fix that and break the equal-step control instead; the script
keeps equal steps and prints the loss column so the trade is visible.

**One implementation note that is not cosmetic**: the script loads the model in **float32**, while
every other script here uses bfloat16. bfloat16 carries about three decimal digits, and ΔW is far
smaller than the weights it is added to — stored in bfloat16, the quantity being measured is partly
rounded away. The cost is double the memory (**9.75 GB peak**), which a 1.5B model can still afford.

### 2.5 The Decomposition Itself

```
A = U Σ Vᵀ
```

| Symbol | Meaning | Shape |
| :--- | :--- | :--- |
| **A** | The matrix being decomposed | m × n |
| **U** | Left singular vectors | m × m |
| **Σ** | Singular values on the diagonal | m × n |
| **V** | Right singular vectors | n × n |

**Written as a weighted sum, this is the load-bearing identity of the whole module:**

```
A = σ₁ u₁ v₁ᵀ + σ₂ u₂ v₂ᵀ + … + σₖ uₖ vₖᵀ
```

Any matrix is **a weighted sum of rank-1 blocks, and the weights are the singular values**.
⇒ **If the singular values decay quickly, the later blocks can be dropped.** The ΔW assumption,
image compression and rating completion are all this one identity applied to different matrices.

### 2.6 A Small Case Worth Following By Hand

`01_svd_image_compression.py` starts with a 3 × 2 matrix, small enough to check every claim:

```
A =
[[1. 2.]
 [1. 1.]
 [0. 0.]]

U (left singular vectors, one column per term):
[[-0.85065081 -0.52573111]
 [-0.52573111  0.85065081]
 [ 0.          0.        ]]

singular values: [2.61803399 0.38196601]

V_T (right singular vectors, one row per term):
[[-0.52573111 -0.85065081]
 [ 0.85065081 -0.52573111]]

eigenvalues of A_T A : [6.85410197 0.14589803]
singular values squared: [6.85410197 0.14589803]
sqrt of eigenvalues    : [2.61803399 0.38196601]
```

**The last two lines are identical**, so "singular values are the square roots of the eigenvalues of
AᵀA" stops being something to memorise. The script then prints each rank-1 block on its own and
adds them back:

```
sum of both terms (should equal A):
[[1. 2.]
 [1. 1.]
 [0. 0.]]
max absolute deviation from A: 8.88e-16
```

### 2.7 Singular Vector Signs Flip in Pairs

This is easy to get wrong and hard to notice afterwards. **Negating column j of `U` together with
row j of `Vᵀ` leaves the product unchanged; negating only one of them silently produces a different
matrix.** The script demonstrates both:

```
U with column 1 negated, V_T untouched:
[[-1.34164079 -1.78885438]
 [-0.4472136  -1.34164079]
 [ 0.          0.        ]]
max absolute deviation from A: 3.788854

both column 1 of U and row 1 of V_T negated:
[[1. 2.]
 [1. 1.]
 [0. 0.]]
max absolute deviation from A: 8.88e-16
```

⇒ **Tidying the signs of a written-down decomposition to look neater breaks it**, even when every
individual number in it is correct.

### 2.8 Truncation Made Visible

The same script renders a 512 × 512 grayscale test image locally — no photograph is shipped —
and rebuilds it from the top k terms. The drawing deliberately mixes three kinds of structure,
because each behaves differently under truncation:

- **Smooth gradients** are captured by the first few terms
- **Hard edges** need mid-range terms
- **Fine text and noise** live in the tail that truncation discards first

```
    k  rel. error     stored      dense     ratio
    1     35.47%       1025     262144    0.39%
    5     12.24%       5125     262144    1.96%
   20      6.08%      20500     262144    7.82%
   50      3.78%      51250     262144   19.55%
  100      2.51%     102500     262144   39.10%
```

Visually: at k=1 only blurred bands survive; at k=5 the circle and rectangle emerge; **at k=20 the
word rendered on the image becomes readable**; by k=100 it is hard to tell from the original.

**Counting the cost**: a rank-k factorisation stores k columns of `U`, k singular values and k rows
of `Vᵀ`, so it costs `k × (rows + columns + 1)` numbers against `rows × columns`. The script prints
both sides of the division rather than only the percentage, because **this ratio is easy to misplace
by a factor of ten**:

```
Same accounting on a 1000x1000 matrix, where the ratio is easy to misstate:
  k=3       6003 / 1000000  = 0.60%
  k=10     20010 / 1000000  = 2.00%
  k=50    100050 / 1000000  = 10.01%
```

### 2.9 Energy Share Is a Flattering Metric

**The most transferable point in this chapter.** For that same image the first term alone holds
**87.42%** of the energy, which sounds like one term is nearly enough. The k=1 rebuild has a
**35.47%** relative error and shows nothing.

The reason is that energy is the *square* of the error: `relative error = sqrt(1 - cumulative energy)`.

```
Energy share against relative error (error = sqrt(1 - energy)):
  k=1   energy 87.42%  ->  relative error 35.47%
  k=2   energy 92.64%  ->  relative error 27.12%
  k=3   energy 95.59%  ->  relative error 21.01%
  k=8   energy 99.06%  ->  relative error  9.70%
```

⇒ **A rank chosen at a 90% energy threshold still rebuilds the matrix with about a third of its
magnitude wrong.** Choose k by reconstruction error against the task, never by the energy
percentage alone.
---

## 3. The Same Idea Where Most of the Matrix Is Missing

Approximating a large matrix with two thin ones is not new, and a recommendation matrix is the
version of it you can watch happen. `02_als_low_rank_factorization.py` builds one with the
structure known in advance.

### 3.1 A 12 × 9 Matrix With Three Groups Baked In

Twelve users, nine items, three groups, two interactions per user — and every user touches only
part of their own group, so a correct factorisation has to recommend the group item they have not
touched yet:

```
Matrix shape: 12 users x 9 items
Observed cells: 24 of 108 (22.2% dense)

        i1  i2  i3  i4  i5  i6  i7  i8  i9
  u1     1   1   .   .   .   .   .   .   .
  u2     1   .   1   .   .   .   .   .   .
  u3     .   1   1   .   .   .   .   .   .
  u4     1   1   .   .   .   .   .   .   .
  u5     .   .   .   1   1   .   .   .   .
  u6     .   .   .   1   .   1   .   .   .
  u7     .   .   .   .   1   1   .   .   .
  u8     .   .   .   1   1   .   .   .   .
  u9     .   .   .   1   1   .   .   .   .
  u10    .   .   .   .   .   .   1   1   .
  u11    .   .   .   .   .   .   .   1   1
  u12    .   .   .   .   .   .   1   .   1
```

**Three blocks are visible by eye** — that is what low rank looks like in data: twelve rows,
three underlying patterns. The singular values confirm it before any fitting happens:

```
Singular values: [2.715 2.358 2.    1.276 1.199 1.    1.    1.    1.   ]
First three terms hold 70.56% of the energy.
Gap between value 3 and value 4: 2.000 vs 1.276
```

**Two arrays are needed, not one**: the ratings and a **mask** of which cells were actually
observed. Without the mask a missing cell and a genuine zero are indistinguishable, and the fit
would spend its capacity explaining zeros nobody recorded.

### 3.2 What Alternating Least Squares Does

The objective is a sum over **observed cells only**, plus a penalty on the factor sizes:

```
min  Σ_{observed}  ( r_ui − x_uᵀ y_i )²  +  λ ( Σ_u ‖x_u‖² + Σ_i ‖y_i‖² )
```

> **The `observed` subscript is the whole reason this works on sparse data**: empty cells never
> enter the loss, so the matrix does not have to be filled in before training.

Holding one side fixed turns a hard joint problem into two easy ones. For each row it becomes an
ordinary ridge regression with a closed form:

```
x_u = ( Y_u Y_uᵀ + λI )⁻¹ Y_u R_u
```

In code that is one `solve` per row, over the observed columns only:

```python
for row in range(ratings.shape[0]):
    observed = mask[row]
    factors = fixed[observed]
    targets = ratings[row, observed]
    gram = factors.T @ factors + regularisation * eye
    result[row] = np.linalg.solve(gram, factors.T @ targets)
```

### 3.3 The Quantity Being Minimised Is Not the One Usually Printed

Running twenty iterations produces a result that looks contradictory until the objective is
printed alongside the error:

```
  iteration  1  objective  4.663190  RMSE 0.008536
  iteration 10  objective  1.296704  RMSE 0.011209
  iteration 20  objective  1.065799  RMSE 0.016945

Objective: 4.663190 -> 1.065799, decreasing every iteration: True
RMSE:      0.008536 -> 0.016945
```

**The objective falls monotonically for twenty iterations while the plain RMSE rises.** Each
half-step solves a *penalised* regression, so what descends is squared error **plus** the penalty.
RMSE omits the penalty term — the solver is trading a little training error for much smaller
factors, which is precisely what the penalty is for.

⇒ **Before reporting that the loss went down, confirm that the number reported is the one being
minimised.**

### 3.4 Scoring With Something That Cannot Be Argued With

The loss cannot settle this, so the script scores **group agreement** instead: every user belongs
to exactly one group, and the item they have not touched is known in advance.

```
Recommendations after 20 iterations:
  u1   group A  top i3 [in group]   i3:0.973, i7:0.751, i9:0.751
  u5   group B  top i6 [in group]   i6:0.971, i8:0.379, i9:0.379
  u10  group C  top i9 [in group]   i9:0.980, i2:0.841, i1:0.841
  top-1 inside the correct group: 12/12 = 100.0%

Recommendations after 2 iterations:
  u1   group A  top i7 [WRONG GROUP]   i7:0.802, i9:0.772, i3:0.727
  u10  group C  top i1 [WRONG GROUP]   i1:1.296, i2:1.113, i9:1.100
  top-1 inside the correct group: 7/12 = 58.3%
```

| | Objective | RMSE | **Group agreement** |
| :--- | ---: | ---: | ---: |
| After 2 iterations | 3.105801 | **0.007293** (lower) | **58.3%** |
| After 20 iterations | 1.065799 | 0.016945 | **100.0%** |

**The early-stopped fit has the better RMSE and the worse recommendations.** Stopping the
alternation early still prints a number; the group check is what decides the outcome.

### 3.5 What Actually Decides Whether Recovery Is Possible

On a 300 × 120 matrix generated from a known rank-3 signal, holding the penalty and the iteration
count fixed and moving only the sparsity:

```
 density  min obs/row  mean obs/row  train RMSE  held-out    share
    0.03            0           3.6      0.2053    2.0076     167%
    0.05            1           6.1      0.3987    2.0704     172%
    0.08            3           9.6      0.2951    1.0219      85%
    0.15            7          18.0      0.0244    0.0481       4%
    0.30           22          36.1      0.0099    0.0148       1%
```

**A held-out error above 100% of typical magnitude is worse than answering zero everywhere.**
The dividing line falls exactly where **a row holds no more observations than the rank**: three
observed cells against three free parameters fits those cells perfectly and predicts nothing.

A stronger penalty softens the failure without replacing the missing data:

```
  at density 0.05, where the thinnest row holds 1 observed cell
    penalty  train RMSE  held-out    share
       0.05      0.4202    2.6824     223%
       0.30      0.3987    2.0704     172%
       1.00      0.4074    1.4330     119%
       3.00      0.5360    1.0472      87%
```

⇒ **Rank is a claim about how much data each row needs.** The same arithmetic applies when the
rank is an adapter's: capacity that the data cannot support is capacity that memorises.

---

## 4. The Adapter: Shape, Placement and Cost

### 4.1 The Forward Pass

```
h = Wx + BAx = (W + BA)x
```

| Symbol | Role |
| :--- | :--- |
| **W** | The pretrained weight, **frozen** |
| **A** | The **down** projection, randomly initialised |
| **B** | The **up** projection, initialised to **zero** |
| **BA** | Together they form ΔW, and **that ΔW is low rank by construction** |
| **r** | The rank, the width of the bottleneck |

`03_lora_low_rank_hypothesis.py` writes the layer out by hand rather than importing it:

```python
class LowRankAdapter(torch.nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.down = torch.nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.up = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, hidden):
        return self.base(hidden) + (hidden @ self.down.T @ self.up.T) * self.scale
```

### 4.2 Why the Up Projection Starts at Zero

If both matrices started random, `BAx` would already be a non-zero perturbation before training
began — **the pretrained model would be damaged on step zero**. With `B = 0` the layer answers
`h = Wx` exactly, so training starts from the original behaviour and learns the increment from
there.

```
down shape (8, 256), up shape (256, 8)
up starts at zero: True
max difference from the frozen layer: 0.00e+00
rank of the equivalent update matrix: 0 of 256

after up is given values, max difference: 0.0744
rank of the equivalent update matrix: 8 of 256
```

**Those two rank readings say what `r` means**: the equivalent update can never exceed rank `r`,
by construction and not by hope.

**The `alpha / r` scale** keeps the size of the added term roughly constant as `r` changes.
Raising the rank should add capacity — **it should not also multiply the effective learning rate.**

### 4.3 No Extra Inference Cost, and Hot Swapping

`(W + BA)x` means the adapter can be folded into the frozen matrix before serving, leaving an
ordinary model and an unchanged number of matrix multiplies. Because the structure is **additive**,
swapping tasks means subtracting one adapter and adding another — one base in memory, many
downstream tasks.

### 4.4 Where Adapters Attach, and What Each Choice Costs

**An adapter is not paired one-to-one with every weight in the network.** It attaches to a chosen
list of module names, and that list is a decision with a price. The script surveys them:

```
Model parameters: 1.78B

  projection   count              shape    dense params
      q_proj      28          1536x1536      66,060,288
      k_proj      28           256x1536      11,010,048
      v_proj      28           256x1536      11,010,048
      o_proj      28          1536x1536      66,060,288
   gate_proj      28          8960x1536     385,351,680
     up_proj      28          8960x1536     385,351,680
   down_proj      28          1536x8960     385,351,680
     lm_head       1        151936x1536     233,373,696
```

> **`k_proj` and `v_proj` output 256 rather than 1536** — the signature of grouped-query attention,
> where several query heads share one key-value head, so those projections are much narrower.

```
                       selection   modules         r=1         r=2         r=4         r=8        r=16
                 q_proj + v_proj        56      0.008%      0.015%      0.031%      0.061%      0.123%
  all four attention projections       112      0.015%      0.031%      0.061%      0.123%      0.245%
      attention and feed forward       197      0.074%      0.147%      0.294%      0.589%      1.177%

Updating 8 matrices, 11,010,048 parameters (0.620% of the model)
```

The conventional choice is the first row; the most aggressive is the last. **The familiar "about
1% of the parameters" corresponds to the bottom-right corner of that table**, not to adapters in
general.

Measured trainable shares across this module:

| Configuration | Trainable / total | Share |
| :--- | :--- | ---: |
| 1.5B, r=8, four attention projections | 2,179,072 / 1,779,267,072 | **0.1225%** |
| 1.5B, r=16, four attention projections | 4,358,144 / 1,781,446,144 | **0.2446%** |
| 256M vision-language, r=16, language tower only | 1,843,200 / 258,328,128 | **0.7135%** |

**Two regularities**: doubling `r` doubles the trainable count; and **at the same `r`, a smaller
base gives a larger share**, because the denominator shrank.

### 4.5 Selecting Modules by Suffix Silently Over-Attaches

A vision-language model has attention projections in both towers, **named identically**. Asking for
the suffix `q_proj` attaches adapters in the image encoder as well as the language model.

Measured in `07_vision_lora_gauge_reading.py`: the intended 120 language-tower modules became
**156**, and the trainable share went from 0.71% to 1.05%. Passing full module paths fixes it, and
the script verifies the fix rather than assuming it:

```
Modules named for adaptation: 120
trainable params: 1,843,200 || all params: 258,328,128 || trainable%: 0.7135
Adapters actually attached: 120, of which in the image encoder: 0
```

⇒ **"I only adapted the language tower" is a claim, and the last line is its evidence.**

### 4.6 Rank Selection in Practice

| Situation | Rank |
| :--- | :--- |
| Ordinary tasks | **1, 2, 4, 8** |
| Larger tasks | Higher, at increasing cost |
| Common range | **4 / 8 / 16 / 32** |

Too small and the update cannot be represented (see the truncation table in 2.3 — r=1 keeps only
30.82%); too large and the adapter approaches the cost it was introduced to avoid.

**A small trainable count is also insurance.** On a tiny training set, a full-parameter update has
enough freedom to overfit it completely; an adapter constrained to rank 8 does not.

### 4.7 The Neighbouring Methods

| Method | Trainable share | Idea | Where it fits |
| :--- | :--- | :--- | :--- |
| **Prompt tuning** | <0.01% | **Soft prompts**: freeze everything, train a vector sequence at the **input layer** | Only effective when the base is very large; extreme resource limits |
| **P-tuning v1/v2** | <0.1%–1% | Soft prompts inserted at **every layer** | Understanding tasks; works on mid-size bases too |
| **Prefix tuning** | ~0.1%–3% | Trainable vectors as virtual **prefix tokens**, produced by a small network | Generation tasks |
| **LoRA** | ~1%–10% | **Low-rank update** beside the frozen weight | General purpose; what this module uses |
| **QLoRA** | ~1%–10% | LoRA on top of a **4-bit quantised** base | Fitting a very large base on one consumer card |

> **How to read it**: the first three act **on the input side** and never touch a weight matrix;
> the last two **attach beside the weights**. That boundary sets their ceiling — input-side methods
> are cheaper but depend on the base already being strong enough.

The `peft` library implements all of them; this module uses its `LoraConfig` directly.
---

## 5. Data: What Actually Decides the Outcome

### 5.1 Fewer Trainable Parameters Means Stricter Data, Not Looser

The intuition runs backwards here. **Because so few parameters are trainable, the model leans
harder on exactly what it is shown.** There is no spare capacity to absorb noise — every pattern in
the data, including the unintended ones, lands more directly in the weights.

Three properties matter, in this order:

**Consistency** — one template for everything. A set that mixes "write a poem → [poem]" with
"summarise this → [summary]" leaves the model no stable target to learn.

**Accuracy** — the answers must be right. **The model learns every pattern present, including the
mistakes.**

**Diversity** — within that fixed form, cover the range of inputs the task will see.

> **Consistency and diversity pull against each other.** Consistency governs the **form**,
> diversity governs the **content**. The order is not negotiable: **fix the form first, then vary
> within it.**

Two additions that come from practice rather than theory: the data must cover **every value a field
can take** (a gauge reader needs samples of every gear position, not just the common one), and it
should come from **real work**, not from a generator imitating it.

### 5.2 Labels From a Rule Make Evaluation a Measurement

`04_lora_sft_instruction_tuning.py` generates its own English dataset. Age and prior-claim count go
in; a tier and an action come out, by a rule stated in the code:

```python
def classify(age, claims):
    if age < 25 or claims >= 3:
        tier = "C"
    elif claims == 0 and age >= 30:
        tier = "A"
    else:
        tier = "B"
    return tier, ACTIONS[tier]
```

**Why generate rather than collect**: the labels come from a rule, not from a person, so every
held-out input has exactly one correct answer and a wrong answer is wrong without argument.
A set scraped from free text cannot be scored that way, and **"the answers look better" would be
the only verdict available**.

**The split is over inputs, not over rendered strings.** If a held-out input also appeared in
training, a high score would only prove memorisation and the comparison would say nothing about
whether the rule was learned.

```
Training examples: 480, evaluation examples: 60
Tier distribution in training: {'A': 77, 'B': 168, 'C': 235}
Two training examples as the model sees them:
  input : age=22; claims=3; vehicle_value=15000
  output: TIER: C | ACTION: decline
  input : age=42; claims=1; vehicle_value=15000
  output: TIER: B | ACTION: refer
```

### 5.3 The Template, the Stop Token and the Mask

The instruction template is the conventional three-part form:

```
Below is an instruction that describes a task, paired with an input that provides
further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```

**The `input` field exists for instructions that need context** ("translate the following" plus a
passage). When a task has only a question and an answer, the right move is **a two-slot template**,
not an empty string forced into a three-slot one.

Two details decide whether the training run works at all:

```python
def encode(tokenizer, record, max_length):
    prompt = TEMPLATE.format(instruction=INSTRUCTION, input=record["input"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(record["output"], add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return input_ids, labels
```

- **The end-of-sequence token has to be appended to the answer.** Without it nothing ever tells the
  model to stop, and generation runs until it hits the token limit.
- **The prompt tokens have to carry the ignore label.** Otherwise the model spends capacity
  learning to write the questions back out instead of answering them.

The accounting is worth seeing once:

```
Tokens in the rendered example: 104
Tokens carrying a label: 9 (8.7% of the sequence)
Supervised text: 'TIER: C | ACTION: decline<｜end▁of▁sentence｜>'
```

⇒ **Nine tokens out of 104 carry the training signal.** That ratio explains why instruction tuning
needs more examples than intuition suggests.

### 5.4 How Much Data

| Base size | Task | Pairs |
| :--- | :--- | :--- |
| ~7B | Style, simple question answering | **1,000 – 5,000** |
| ~7B | Reasoning, specialist domains | **5,000 – 50,000+** |
| 13B–70B | General or complex | **10,000 – 100,000+** |
| >70B | Continued pretraining | **GB-scale corpora** |

**Spend the effort on the first thousand.** A widely repeated split of the work puts **80% of the
time into cleaning, format unification and answer verification** — a thousand clean pairs beat a
hundred thousand messy ones. Start small, measure, and add data only after underfitting is
demonstrated rather than assumed.

**Quality gates quantity, not the other way round**: more data helps *when the quality holds*;
without that, more data can actively hurt.

---

## 6. Tooling: What This Module Uses, and What It Declined

Purpose-built fine-tuning frameworks (Unsloth and similar) wrap the same operations with custom
kernels, quantised loading and export helpers, and report **2–5× faster training with 50–80% less
memory**.

**This module uses `peft` + `transformers` with hand-written training loops instead**, for one
concrete reason: the GPU here is Blackwell (sm_120) and needs CUDA 12.8 or newer, and the
accelerated stacks — the kernel compilers, the 4-bit quantisation library, the high-throughput
inference engine — are either unverified or unsupported in that combination on Windows.

| Concern | Choice here |
| :--- | :--- |
| Loading | `transformers` — `AutoModelForCausalLM`, `AutoModelForImageTextToText` |
| Adapters | `peft` — `LoraConfig` + `get_peft_model` |
| Supervised training | Hand-written loop: `torch.optim.AdamW` and a local collate function |
| Reward-driven training | Hand-written loop: sampling, advantages and the KL term written out |
| Generation | `transformers` `generate`, plus token-by-token decoding where intervention is needed |
| Precision | bfloat16, no quantisation library (float32 in one script, see 2.3) |

**What is given up is speed and one-line GGUF export.** What is being shown does not change: the
low-rank identity, the zero-initialised start, module selection, trainable share, prompt masking,
adapter save/reload/merge, group-relative advantages and the KL constraint all live in `peft` and
`torch`. If anything, **the wrappers hide the parts worth seeing** — a supervised-tuning helper
class conceals exactly the prompt-masking step that 5.3 depends on.

---

## 7. Chain One: Supervised Fine-Tuning

### 7.1 Configuration

```python
RANK = 8
ALPHA = 16
DROPOUT = 0.05
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
STEPS = 120
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 128
```

```
Rank 8, alpha 16, dropout 0.05
Attached to: q_proj, k_proj, v_proj, o_proj
trainable params: 2,179,072 || all params: 1,779,267,072 || trainable%: 0.1225
```

**That last line is the number that explains why this fits on one consumer card.** The frozen
weights still occupy memory for the forward pass, but they carry **no gradients and no optimiser
state** — and those two are usually what exhausts the card.

**Steps, batches and epochs**: one step is one forward-and-backward pass over `BATCH_SIZE` examples.
With gradient accumulation the effective batch is `batch × accumulation`, which is also the answer
to "how do I train when the data will not fit in memory" — accumulate gradients across several
batches and update once.

### 7.2 Measuring the Base Model First

The base is scored **before** the adapter is attached, on the same 60 held-out inputs, and it is
scored **twice**. The first pass gives it the task description only:

```
Base model, instruction only:
  answers in the required shape: 7/60 (11.7%)
  answers exactly correct:       0/60 (0.0%)
  correct by expected tier:      A: 0/7  B: 0/24  C: 0/29
    input    age=54; claims=1; vehicle_value=30000
    expected TIER: B | ACTION: refer
    produced 'TIER: C | ACTION: refer  Wait, but the input has age=54, which is a specific'
```

**That number alone would overstate what training buys.** The rule is three lines, so a prompt can
carry it, and prompting is the cheaper thing a person tries first. The second pass writes the rule
into the instruction and changes nothing else:

```
Base model, rule written into the prompt:
  answers in the required shape: 60/60 (100.0%)
  answers exactly correct:       29/60 (48.3%)
  correct by expected tier:      A: 0/7  B: 0/24  C: 29/29
```

Format compliance goes straight to 100%: most of the rambling was the model working out an answer
it had no way to know. Correctness reaches 48.3% — **but the per-tier column shows what that number
really is.** Every tier C case is right and every A and B case is wrong, so the model is answering
C to everything and collecting the half of the set that happens to be tier C. It has not applied the
rule at all.

The wording of that second prompt matters more than it looks. Three phrasings were compared on the
same cases: the operator form kept here, a prose version, and the operator form with "output only
that line" appended. The prose version reasons slightly better but never stops cleanly, so it scores
zero on exact matching; the no-explanation version is worse on both counts. **A baseline is only
worth reporting if it is the best the cheaper method can do**, so the strongest of the three is the
one in the script.

**Without that baseline the after number means nothing** — there would be no way to tell whether an
accuracy came from the training or was there all along.

### 7.3 Two Scores, Not One

Schema compliance and correctness are tracked separately on purpose:

> A model can produce the right tier inside a paragraph of prose, which is useless to a caller
> parsing the line; and it can produce a perfectly shaped line with the wrong tier in it.
> **Collapsing both into one accuracy hides whichever of the two failed.**

```
  step   1  loss 1.6288  mean of last 1 1.6288
  step  40  loss 0.1200  mean of last 20 0.1209
  step 120  loss 0.0069  mean of last 20 0.0479
Loss: 1.4920 -> 0.0351 (mean of the last ten steps)
Wall clock: 18.1 s for 120 steps (151 ms per step)
Peak VRAM reserved: 6.13 GB

Adapted model, after training:
  answers in the required shape: 60/60 (100.0%)
  answers exactly correct:       47/60 (78.3%)
  correct by expected tier:      A: 7/7  B: 13/24  C: 27/29
    input    age=54; claims=1; vehicle_value=30000
    expected TIER: B | ACTION: refer
    produced 'TIER: B | ACTION: refer'

                                 setting   schema    exact
                  base, instruction only   11.7%    0.0%
      base, rule written into the prompt  100.0%   48.3%
     adapter, rule learned from examples  100.0%   78.3%
```

**Read the middle row before the last one.** Stating the rule costs nothing and already buys 48.3%,
so the figure for what training bought is **48.3% to 78.3%, thirty points**, not the eighty the
first and last rows would suggest. And since the middle row earns its 48.3% by answering C to
everything, even thirty points understates the gap in how the two actually behave.

**The per-tier breakdown is what makes the residual error actionable.** A single 78.3% could mean
small errors everywhere or one branch missed; here tier A is perfect, tier C is nearly so, and
**tier B is 13 of 24**. Tier B is the branch defined by exclusion — neither C nor A — and it is the
one the adapter learned least well. That is a different problem from scattered noise and calls for
a different fix.

**This is also what a larger evaluation set is for.** An earlier version of this script held out 24
cases instead of 60. On that set the same training scored 91.7% with tiers A and B both perfect,
because tier A had two cases in it and tier B had ten. The weakness in tier B was not absent; the
sample was too small to show it.

### 7.4 A Number That Changes Between Runs Cannot Be Quoted

The first two runs of this script reported **83.3%** and **87.5%** with identical code and a fixed
data seed. **Adapter dropout draws from the global torch generator**, which was never seeded:

```python
# Adapter dropout draws from the global torch generator, so without this the
# same script prints a different accuracy on every run and no number in the
# output can be quoted.
torch.manual_seed(SEED)
```

With the seed set, the run reproduces its number exactly. **Reproducibility is a precondition for
reporting, not a nicety.**

### 7.5 Save, Reload, Merge

Three separate facts get checked, because they fail in different ways:

```
Adapter directory: .../outputs/sft_adapter
  files: README.md, adapter_config.json, adapter_model.safetensors
  size: 8.75 MB

Reloaded adapter reproduces the trained answers: True
Merged model reproduces the same answers: True
Adapter modules left after merging: 0
```

- **The saved directory holds only the adapter** — 8.75 MB beside a multi-gigabyte checkpoint
- **Reloading onto a freshly loaded base must reproduce the same answers**, otherwise the artefact
  on disk is not the thing that was trained
- **Merging folds the adapter into the frozen matrices**, after which it is an ordinary model with
  nothing left to swap — convenient to deploy, no longer swappable

The configuration file records which base the adapter belongs to, which is why a directory of a few
megabytes can be loaded on its own. It also means **copying the adapter to another machine is not
enough — the base has to be resolvable there too.**
---

## 8. Chain Two: Controlling How Long a Model Thinks

**This chain changes no weights at all.** It sits at inference time, and it is the cheapest thing
in the module — which is why it comes before the training chain that targets the same goal.

### 8.1 The Mechanism

A reasoning model emits its deliberation between delimiters before answering. Three interventions
are possible:

1. **Cap the thinking** — set a maximum in tokens; if the model finishes inside it, nothing happens
2. **Force the end** — once the cap is spent, **write the closing delimiter into the stream**,
   which pushes the model into its answer
3. **Extend the thinking** — **ban the closing delimiter** and append a nudge, giving the model a
   chance to re-examine what it just concluded

### 8.2 Delimiters Have to Be Located, Not Assumed

`06_thinking_budget_control.py` resolves them first, because whether the closing delimiter is a
single token decides how it can be controlled:

```
  '<think>' -> 1 token(s), ids [151648]
  '</think>' -> 1 token(s), ids [151649]
```

**A single id can be banned outright in the logits**, which is how the extension in step 3 works.
If a checkpoint spelled it across several tokens, banning the first would be the equivalent move —
so the script prints the count rather than assuming it.

### 8.3 Why the Decoding Loop Is Written Out by Hand

A single `generate` call finishes the sequence before anything can be applied to it. Intervening
means feeding one token at a time and keeping the cache:

```python
while thinking_tokens < budget:
    if extensions - nudges_used > 0:
        logits = logits.clone()
        logits[close_ids[0]] = float("-inf")
    token = pick_token(logits, temperature)
    ...
# Close the thinking phase by hand and let the model produce its answer.
forced = close_ids + tokenizer(CLOSING_HINT, add_special_tokens=False)["input_ids"]
```

**Both interventions share the one loop**: while extensions remain the closing delimiter is banned
and a nudge is appended each time the model reaches for it; when the budget runs out the delimiter
is written in regardless of what the model preferred.

### 8.4 What the Budget Buys

Eight questions with decidable numeric answers (letter counting and arithmetic), greedy decoding,
four settings:

```
                   setting  accuracy  mean thinking tokens  stopped on its own   seconds
                    cap 24 1/8                        24.0             0/8          31.1
                    cap 64 2/8                        64.0             0/8          44.6
                   cap 160 4/8                       110.9             7/8          53.2
        cap 160 + 2 nudges 4/8                       156.9             2/8          58.6

From the tightest cap to the widest: accuracy 12.5% -> 50.0%,
mean thinking tokens 24.0 -> 110.9
Peak VRAM reserved: 3.78 GB
```

**Three readings, and the third column carries most of them:**

1. **Truncation is real**: accuracy goes 1/8 → 2/8 → 4/8 with the budget as the only variable.
2. **"Stopped on its own" is the evidence that the cap bound**: at 24 and 64 tokens it is **0/8** —
   not one question ended because the model wanted to stop. At 160 it is **7/8**.
3. **At a cap of 160 the mean spend is 110.9** — the allowance is not exhausted. These questions
   need a little over a hundred tokens, and raising the ceiling further would not be spent.

### 8.5 The Extension Direction Did Not Reproduce

**Banning the delimiter and appending "Wait" cost 46 more thinking tokens and returned the same
4/8.** The canonical demonstration of this technique is a letter-counting question; here it stayed
wrong under every setting:

```
  cap 160
    thinking tokens spent: 130, nudges: 0, stopped on its own: True
    answer: '2 Step-by-step explanation: 1. The word "raspberry" is analyzed...'
    expected 3, read 2, correct False

  cap 160 + 2 nudges
    thinking tokens spent: 160, nudges: 1, stopped on its own: False
    answer: '2 </think> To determine how many times the letter **r** appears...'
    expected 3, read 2, correct False
```

⇒ **On a 1.5B distilled model, "think again" did not recover the answer.** The truncation
direction holds; the extension direction does not. A plausible explanation is that the published
results come from models an order of magnitude larger, and that re-reading a question is itself a
capability this size of model lacks — **but that is a conjecture, not something measured here.**

**This is reported as it ran.** A technique that works at one scale is not thereby a technique.

---

## 9. Chain Three: Training on Scored Samples

Same target as chain two — a model that shows its reasoning — but reached by changing weights.
The distinction between the two routes is **where the reasoning comes from**: bought as annotation
and trained with ordinary supervision, or discovered by the model and selected by a reward.

### 9.1 Group-Relative Advantages

The method samples a **group** of answers to the same question and lets the group grade itself.
No value network is trained: **the group mean is the baseline**, and every sample is judged by how
far it beats its own siblings on that same question.

```python
def advantages_from_rewards(rewards):
    tensor = torch.tensor(rewards, dtype=torch.float32)
    spread = tensor.std()
    if spread < 1e-6:
        return torch.zeros_like(tensor), float(tensor.mean()), 0.0
    return (tensor - tensor.mean()) / (spread + 1e-6), float(tensor.mean()), float(spread)
```

**The guard on the standard deviation is not defensive coding — it is the failure mode.** When every
sample in a group scores the same, the advantages are zero, and the step is correctly a no-op.

### 9.2 Five Rewards, Weighted Deliberately

| Reward | Test | Maximum |
| :--- | :--- | ---: |
| **Correctness** | The extracted answer equals the expected one | **2.0** |
| **Integer** | A bare integer sits inside the answer tags | 0.5 |
| **Strict format** | Exact line layout including newlines | 0.5 |
| **Soft format** | Tags in the right order, whitespace ignored | 0.5 |
| **Tag count** | 0.125 per tag appearing exactly once | 0.5 |

```
--- 2. The five reward terms, scored on two hand-written samples ---
    well formed, correct: total 4.000  (tags 0.500, soft 0.500, strict 0.500, integer 0.500, correct 2.000)
          prose, no tags: total 0.000  (tags 0.000, soft 0.000, strict 0.000, integer 0.000, correct 0.000)
```

**Correctness is worth 2.0; all the format terms together are worth 2.0.** Form and substance carry
equal weight, so a model that only learns to look right can earn at most half.

**The tag-count term is the only one that pays partial credit**, and it exists for a specific
reason:

> If every reward were all-or-nothing, a model that never once produced the full shape would score
> identically on every sample, the advantages inside the group would all be zero, and there would
> be no gradient to learn from.

### 9.3 Staying Near the Base Without Loading It Twice

The KL term needs the frozen policy's log probabilities. Rather than keeping a second copy of the
model in memory, **switching the adapter off turns the same weights back into the original**:

```python
with torch.no_grad():
    with model.disable_adapter():
        reference, _ = sequence_log_probabilities(
            model, piece, prompt_length, pad_token_id)
```

Without that term the policy is free to collapse onto whatever the reward functions happen to
reward, **and fluency is not among the things they check**.

### 9.4 A Flat Zero Reward Curve, and the Fix

The first run produced this, for ten steps:

```
 step  reward  spread       kl    tags     soft   strict  integer  correct
    1   0.000   0.000   0.0000   0.000    0.000    0.000    0.000    0.000
    5   0.000   0.000   0.0012   0.000    0.000    0.000    0.000    0.000
   10   0.000   0.000   0.0002   0.000    0.000    0.000    0.000    0.000
```

**A cold start.** Asked for tagged output, the base model writes a paragraph of prose —
`'Okay, so I have this problem about beads in a jar. Let me try to figure it out step by step...'` —
so no sample in any group earned anything, every advantage was zero, and no gradient existed.

The fix is to hand over the first token of the structure by writing it into the prompt:

```python
PREFILL = "<reasoning>\n"
```

**Not a formatting convenience.** Making some samples better than their siblings is the *only*
condition under which a group-relative method can start, and prefilling one tag is the cheapest way
to create that difference.

### 9.5 What 24 Steps Bought

```
 step  reward  spread       kl    tags     soft   strict  integer  correct
    1   0.760   1.281   0.0000   0.260    0.042    0.042    0.083    0.333
    4   1.427   1.459   0.0034   0.344    0.250    0.250    0.250    0.333
   11   0.219   0.058  -0.0001   0.219    0.000    0.000    0.000    0.000
   19   1.625   1.676   0.0025   0.375    0.250    0.250    0.250    0.500
   23   1.438   1.708   0.0010   0.312    0.208    0.208    0.208    0.500
   24   0.781   0.954   0.0008   0.281    0.125    0.125    0.083    0.167

Mean reward, first five steps 0.900, last five steps 1.160
Wall clock: 751.0 s for 24 steps (31.3 s per step)
Peak VRAM reserved: 7.07 GB

Before training:
  answers holding the tag structure: 0/12 (0.0%)
  answers with the right integer:    0/12 (0.0%)

After training:
  answers holding the tag structure: 8/12 (66.7%)
  answers with the right integer:    7/12 (58.3%)

Tag structure 0.0% -> 66.7%
Correct integer 0.0% -> 58.3%
```

**Two things have to be said plainly about that table:**

1. **The curve is noisy, not a smooth ascent.** Steps 11 and 17 collapse back to about 0.2.
   Twenty-four steps is nowhere near enough to speak of convergence; the defensible statement is
   the one the script prints — **first five steps 0.900, last five 1.160**.
2. **31 seconds per step is almost entirely sampling**: two questions × six answers × 160 tokens,
   all decoded token by token. **Reward-driven training is slow because it must generate as it
   trains**, and no configuration change removes that.

### 9.6 A Memory Trap Specific to Large Vocabularies

The first implementation took a full log-softmax over the logits to gather the chosen tokens'
probabilities. On a 151,936-entry vocabulary that allocates **a second array the size of the
logits** — memory went to **11.8 GB of 11.9 GB** and GPU utilisation fell to 18% as the allocator
thrashed.

The fix is arithmetically identical and allocates nothing extra:

```python
chosen = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1).float()
normaliser = torch.logsumexp(logits.float(), dim=-1)
gathered = chosen - normaliser
```

Combined with backpropagating in chunks of two sequences, **the peak fell to 4.7 GB**:

```python
for start in range(0, group_size, chunk_size):
    ...
    (loss_scale * (policy_loss + kl_coefficient * kl)).backward()
```

Gradients accumulate in the parameters exactly as they would from one large batch; **what changes
is the peak, and the peak is what decides whether the script runs at all.**

---

## 10. Chain Four: An Adapter on a Vision-Language Model

### 10.1 Rendering the Data Instead of Labelling It

`07_vision_lora_gauge_reading.py` draws its instrument panels **from the labels**, rather than
labelling photographs. That decision buys three things at once:

- **The label cannot disagree with the image** — the label is the drawing's input
- **Any number of samples can be produced** — which removes the overfitting that a two-sample set
  guarantees
- **No copyright or privacy question** attaches to the material

The needle angle and the zone label come from the same number, so those two can never contradict
each other either:

```python
fraction = rng.uniform(0.03, 0.97)
zone = ZONES[0] if fraction < 0.34 else ZONES[1] if fraction < 0.67 else ZONES[2]
```

**The four fields are deliberately graded in difficulty**: gear is one of four, the lamp is on or
off, the needle zone is a threshold on a continuous quantity, and the odometer requires reading six
digits off a small picture.

```
Rendered 112 panels at 384x384
Example label: GEAR: R | LAMP: off | NEEDLE: low | ODO: 226355
Panels: 96 for training, 16 held out
```

### 10.2 Two Towers, and Which One to Adapt

A vision-language model is two networks joined by a projection: the image encoder turns pixels into
embeddings, the language model consumes them alongside the text, and a connector maps between the
widths.

```
       tower  linear layers    dense params    share
      vision             72      84,934,656   33.1%
    language            211     134,553,600   52.5%
   connector              1       7,077,888    2.8%

Adapter cost at rank 16:
     language tower only:  120 modules,  1,843,200 parameters (0.719% of the model)
     language and vision:  156 modules,  2,727,936 parameters (1.064% of the model)
```

**The choice is real, not cosmetic**: if the answer depends on something the encoder already
represents, adapting the language side is enough and touching the encoder mostly costs memory.
This script adapts the language tower only — and 4.5 shows how that claim is verified rather than
asserted.

### 10.3 Masking When an Image Is in the Input

The prompt length cannot be guessed, because the processor **replaces the image placeholder with a
block of image tokens whose count depends on the picture**. The script measures it by encoding the
prompt alone with the same image, then checks its own assumption:

```python
prompt_batch = processor(text=prompt, images=[record["image"]], return_tensors="pt")
full_batch = processor(text=prompt + record["target"] + processor.tokenizer.eos_token,
                       images=[record["image"]], return_tensors="pt")

prompt_length = prompt_batch["input_ids"].shape[1]
full_ids = full_batch["input_ids"]
if not torch.equal(full_ids[0, :prompt_length], prompt_batch["input_ids"][0]):
    raise RuntimeError("the full sequence does not start with the prompt sequence")
```

**Guessing that number would silently shift the mask and supervise the wrong positions** — a bug
that produces a plausible loss curve and a useless model.

Examples are accumulated one at a time rather than padded into a batch: image token counts differ
between pictures, so a naive stack needs padding rules for both the text and the pixel tensors,
while accumulating gradients over single examples reaches the same update with none of it.

### 10.4 Scoring Each Field Separately

```
Before training:
  answers in the required shape: 0/16 (0.0%)
     gear correct:   0/16 (0.0%)
     lamp correct:   0/16 (0.0%)
   needle correct:   0/16 (0.0%)
      odo correct:   0/16 (0.0%)
    expected GEAR: R | LAMP: off | NEEDLE: low | ODO: 226355
    produced '226355'
    expected GEAR: D | LAMP: off | NEEDLE: high | ODO: 897595
    produced 'Answer: <P|R|N|D>.'
```

> 🔍 **Read the first "produced" line again.** The untouched model **already reads the odometer
> correctly** — it emits `226355` and nothing else. Another answer copies the format specification
> back verbatim.
>
> ⇒ **All four fields score zero not because the model cannot see, but because the answer cannot be
> parsed.** What the adapter supplies is compliance with an agreed output contract, not sight.

```
  step   1  loss 1.9049  mean of last 1 1.9049
  step  75  loss 0.0260  mean of last 25 0.0578
  step 150  loss 0.0129  mean of last 25 0.0292
Loss: 1.9049 -> 0.0292 (mean of the last ten steps)
Wall clock: 303.5 s for 150 steps (2023 ms per step)
Peak VRAM reserved: 4.43 GB

After training:
  answers in the required shape: 16/16 (100.0%)
     gear correct:  16/16 (100.0%)
     lamp correct:  16/16 (100.0%)
   needle correct:  14/16 (87.5%)
      odo correct:  16/16 (100.0%)

   field   before    after
   shape    0.0%  100.0%
    gear    0.0%  100.0%
    lamp    0.0%  100.0%
  needle    0.0%   87.5%
     odo    0.0%  100.0%

Saved the adapter to .../outputs/vision_adapter (7.42 MB)
```

**Four fields differ in what they demand, which is why one number would have been the wrong
report.** Three are choices among a handful of options; the fourth asks for six digits. The only
field short of perfect is **the needle zone (14/16)** — the one field defined by a threshold on a
continuous quantity, where the two misses sit near a boundary.
---

## 11. Memory: The Part That Can Be Calculated

### 11.1 Four Contributors

```
total ≈ model weights + optimiser state + gradients + forward activations
```

| Contributor | Size |
| :--- | :--- |
| **Weights** (the dominant term) | parameters × 2 bytes at 16-bit precision |
| **Optimiser state** | trainable × 4 bytes × 2 states, for AdamW |
| **Gradients** | trainable × 2 bytes — only for trainable parameters |
| **Activations** | strongly dependent on batch and sequence length; **estimate at 20–50% of the weights** |

> **This table is why an adapter saves memory.** The first term is unchanged — the frozen base is
> still loaded in full. The second and third fall by around 99%, because the trainable count falls
> by that much. **What is saved is the cost of training, not the cost of loading.**

A worked example at 7B with 1% trainable: weights 14 GB, optimiser state 0.56 GB, gradients
0.14 GB, activations ≈ 4.2 GB ⇒ **about 19 GB**. Compressed into one line:

```
total ≈ (parameters × 2 bytes) × (1 + activation factor) + (trainable × 10 bytes)
```

where the 10 bytes is 8 for optimiser state plus 2 for gradients.

### 11.2 Measured Here

| Script | Hardware | Peak VRAM | Training time |
| :--- | :--- | ---: | ---: |
| `01_svd_image_compression` | **CPU only** | — | ~10 s (whole script) |
| `02_als_low_rank_factorization` | **CPU only** | — | ~30 s (whole script) |
| `03_lora_low_rank_hypothesis` | GPU | **9.75 GB** | **78 s / 160 steps** (4 runs of 40) |
| `04_lora_sft_instruction_tuning` | GPU | **6.13 GB** | **18 s / 120 steps** (~2 min 50 s whole script, 180 generations) |
| `05_grpo_reward_shaping` | GPU | **7.07 GB** | **751 s / 24 steps** |
| `06_thinking_budget_control` | GPU | **3.78 GB** | no training, ~4 min |
| `07_vision_lora_gauge_reading` | GPU | **4.43 GB** | **303.5 s / 150 steps** |

**Script 03 is the outlier by design** — float32 rather than bfloat16, for the reason given in 2.3.

**A measurement trap worth knowing**: "how much did training itself cost" is often computed as the
difference between peak memory before and after. **If inference already ran before training, that
peak is already set and the difference reads 0.0 GB** — a meaningless number that looks like a
result. Reset the peak statistics first, or do not report the figure.

### 11.3 Weights Are Reused, Not Re-downloaded

Training and inference both require the weights on the GPU, so they must be local first. The four
language-model scripts here (03–06) **share one already-downloaded 1.5B checkpoint** through an
environment variable, adding nothing to disk:

```python
SIBLING_CACHE = Path(__file__).parent.parent / "01-llm-foundation" / "weights"
CACHE_DIR = os.getenv("HF_CACHE_DIR", str(SIBLING_CACHE))
```

Only the vision chain pulls its own model, a 518 MB checkpoint. **Weight directories are never
committed.**

---

## 12. Evaluation: Proving It Worked

### 12.1 The Three Splits

| Split | Purpose |
| :--- | :--- |
| **Training** | Updates the weights |
| **Validation** | Monitors during training, tunes hyperparameters, **selects a checkpoint** — never used for the final report |
| **Test** | The **single, final** measurement on data the model has never seen, kept sealed throughout |

### 12.2 Five Dimensions

| Dimension | What it checks | Why it exists |
| :--- | :--- | :--- |
| **Task metric** | Accuracy, F1, exact match, or whatever the task defines | The hard standard for whether the training worked |
| **Retained general ability** | Common-sense reasoning, basic code, ordinary conversation | **Catastrophic forgetting** — confirming the model did not get worse at everything else |
| **Generalisation** | Same task, different phrasing, harder or noisier inputs | Separates memorisation from learning |
| **Human judgement** | Fluency, relevance, usefulness | Catches what automatic metrics cannot |
| **Case analysis** | Concrete wins and concrete failures, side by side | Points at the next iteration |

**The order of operations after training**: score **both the base model and the adapted one** on the
test set, then check general ability and generalisation, then inspect samples by hand.

> **Scoring the base model is the step most often skipped.** Without that baseline the adapted
> score has no meaning — there is no way to tell what the training contributed.

Scripts 04, 05 and 07 all follow it: the 11.7%, 0.0% and 0/16 baselines in this document are
measured, not assumed.

### 12.3 Four Rules Earned the Hard Way

Each of these corresponds to a specific failure inside this module:

**1. The criterion must be decidable, not impressionistic.**
Script 04's labels come from a rule; script 07's images are rendered from their labels. Neither
"78.3% correct" nor "16/16 odometers" requires anyone's opinion.

**2. Count the two kinds of failure separately.**
Script 04 tracks schema compliance apart from correctness, because a right answer buried in prose
and a well-formed wrong answer need different fixes. Script 07 scores four fields separately, which
is the only reason the needle zone shows up as the weak one.

**3. Put the criterion on an intermediate quantity, not on the loss.**
Script 02's early-stopped fit had the *better* RMSE and 58.3% group agreement against 100%. In
reward-driven training the quantities that matter are **generation length** and **each reward term
separately** — a total loss near zero is equally consistent with "learned everything" and "produced
no gradient at all".

**4. A number that changes between runs cannot be reported.**
Two runs of script 04 gave 83.3% and 87.5% before the global generator was seeded (7.4).

### 12.4 The Step This Module Adds

**Generalisation has to be checked on inputs that were never trained on.** Scripts 04 and 07 split
by input, so held-out cases are genuinely unseen. Script 07's result is worth stating carefully:
the odometer digits are read correctly on **held-out panels**, which is a different and stronger
claim than reading them correctly on a training image.

Where a specialised tool exists, **use it to cross-check details**: OCR on a rendered gauge reads
the digits independently of whatever the language model concluded. **The general model reads the
whole; the specialised tool reads the part.**

---

## 13. Boundaries With Neighbouring Techniques

### 13.1 Distillation and Low-Rank Adaptation

**One sentence separates them: distillation transfers through training samples, an adapter
approximates the parameter change.**

| | **Distillation** | **Low-rank adaptation** |
| :--- | :--- | :--- |
| Level | **Data** | **Parameters** |
| Method | Use a large model's input-output pairs as a training set for a small one | Attach a low-rank branch beside frozen weights |
| Result | **A different, smaller model** | **A patch for the same model** |
| Combinable | Yes — an adapter can implement the training half of a distillation | |

### 13.2 Retrieval and Fine-Tuning

**Fine-tuning holds stable knowledge; retrieval holds volatile knowledge.** The test in 1.2 is the
practical form of that. They compose: a fine-tuned model that answers in the right format, with
retrieval supplying today's facts, is a common and sensible arrangement.

### 13.3 When a Language Model Is the Wrong Tool

Some tasks want a neural network but not a language model — a game-playing agent, for instance,
needs **an environment, a reward, and an action space**, and reinforcement learning over a
purpose-built network.

> Those three map exactly onto chain three: **the environment is the question set, the reward is
> the five scoring functions, and the action space is every token sequence the model can emit.**

---

## 14. Questions This Module Answers Directly

**Are adapters paired one-to-one with every weight matrix?**
No. Placement is an explicit list of module names. Attaching to two attention projections touches
56 modules and 0.123% of the parameters at r=16; attaching to all seven touches 197 modules and
1.177% (4.4).

**What has to be loaded at inference time?**
**Base plus adapter.** The adapter directory is megabytes and records which base it belongs to;
the base still has to be present and resolvable (7.5).

**What is one training step?**
One forward-and-backward pass over `BATCH_SIZE` examples. With gradient accumulation the effective
batch is `batch × accumulation`, which is also how training proceeds when the data will not fit in
memory at once (7.1).

**How do I know the adapter added the capability?**
Baseline the untouched model, monitor intermediate quantities during training, re-score the same
held-out inputs afterwards, cross-check details with a specialised tool, and confirm on inputs that
were never trained on (12.2–12.4).

**Will an English-only training set damage the model's other languages?**
Fine-tuning at this scale changes the mapping to behaviour, not the base's multilingual
representations. Tokenisation and embedding are untouched.

**Why is reward-driven training so much slower than supervised training?**
Because it generates while it trains: 24 steps took 751 seconds against 120 supervised steps in
26.6 seconds. **Sampling six answers per question dominates everything else** (9.5).

**Does more data always help?**
Only when quality holds. Below that bar more data can hurt, and the cheapest experiment is a
thousand clean examples plus a measurement (5.4).

---

## 15. Scripts in This Module

| Script | What it demonstrates | Hardware |
| :--- | :--- | :--- |
| `01_svd_image_compression.py` | Decomposition by hand, singular values against eigenvalues, **paired sign flips**, rank-k reconstruction with storage accounting, **why energy share flatters** | CPU |
| `02_als_low_rank_factorization.py` | Sparse factorisation with a mask, the penalised objective **against** the printed RMSE, early stopping scored by group agreement, **observations per row versus rank** | CPU |
| `03_lora_low_rank_hypothesis.py` | A hand-written adapter that starts as a no-op, parameter accounting per rank, **the ΔW spectrum against two controls**, how much of an update each rank keeps, where adapters can attach, **how the direction count moves as the task widens** | GPU |
| `04_lora_sft_instruction_tuning.py` | Rule-generated labels, instruction template with stop token and prompt masking, adapter attachment and trainable share, **the adapter against a prompted-rule baseline on unseen inputs**, save → reload → merge | GPU |
| `05_grpo_reward_shaping.py` | Five reward functions, group sampling and advantage normalisation, **KL against the base via adapter disabling**, the zero-variance cold start, the large-vocabulary memory trap | GPU |
| `06_thinking_budget_control.py` | Locating thinking delimiters, token-by-token decoding for mid-stream intervention, **capping and extending deliberation**, budget against accuracy | GPU |
| `07_vision_lora_gauge_reading.py` | Images rendered from labels, two-tower survey and adapter pricing, image-aware prompt masking, **per-field scoring**, full before/after | GPU |

**All seven have been run**, and every number in this document comes from those runs. Scripts 01
and 02 need no GPU; 03–06 share one local 1.5B checkpoint; 07 uses a 518 MB vision-language model.

**Reproducing them**: `pip install peft` on top of the module's base requirements is enough for
03–06; 07 additionally needs `torchvision`, which the image processor depends on. Every script
seeds its generators, so the figures above should reproduce exactly on the same hardware.
