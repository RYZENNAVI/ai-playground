# Machine Learning and Deep Learning Foundations

Two lines of work run side by side in this module.

One is **statistical learning on tables**: the classic models, the three gradient boosting
libraries, feature engineering, and model ensembling. The other is **neural networks**,
from a forward pass written with nothing but arrays up to the same network expressed in
three frameworks and served as a graph.

They meet on one question: **how do you know the number you just printed means what you
think it means?**

That question is why almost every result below is stated against something it can be
checked against — a noise floor, a generating coefficient, an untouched holdout, a finite
difference, a hand calculation. Eight scripts produce every measured figure in this
document; the full index is in the last section.

---

## 1. Where these models sit

### 1.1 Deterministic decisions

A generative model answers in text. A great many business questions cannot accept text as
an answer. *Should this loan be approved? Will this customer buy? Will this employee
leave?* The output has to be a decision that a downstream process can act on, and the path
to that decision has to be auditable.

| | Analytical models | Generative models |
| :--- | :--- | :--- |
| **Goal** | Explain data; produce a prediction or a decision | Produce new content resembling the training distribution |
| **Output** | A number, a class, a ranked list | Text, images, audio, video |
| **Data** | High quality, **clearly labelled** | Large volumes of **unlabelled** data |
| **Typical use** | Risk scoring, demand forecasting, churn, pricing | Drafting, summarising, creation |
| **Main risk** | Overfitting; bias in the labels becomes bias in the decision | Fabrication; unverifiable claims |

Three properties decide it, and none of them is about accuracy:

| Analytical models offer | Generative models cannot offer it here |
| :--- | :--- |
| A decision, not a paragraph | Text is the wrong shape for a control flow |
| **A determinate output** | The same prompt can produce different answers |
| **An auditable path from input to output** | The reasoning is not inspectable |

The two are not rivals. A large model is very good at **writing the code** that builds the
small model, and at **explaining** what the small model found. It is the wrong tool for
making the call.

### 1.2 The map of classic algorithms

| Task | Family |
| :--- | :--- |
| **Predict a continuous value** | Regression |
| **Group without labels** | Clustering |
| **Find co-occurring combinations** | Association analysis |
| **Find influential nodes** | Link analysis |

A useful distinction that gets blurred often: **classification is the view from above** —
the categories are defined in advance and the model learns the boundary. **Clustering is
the view from inside** — the model proposes the categories. Association analysis looks for
things that appear together; link analysis looks for position in a network. The last two
sound similar and are not: one operates on baskets, the other on graphs.

| Paradigm | Definition |
| :--- | :--- |
| **Supervised** | Requires labelled data |
| **Unsupervised** | No labels; structure is discovered |
| **Semi-supervised** | A little labelled data plus a lot of unlabelled data |
| **Reinforcement** | Learns a policy from a reward signal |

### 1.3 Five worldviews

This is a map rather than an algorithm. Its value is that it names *why* you would believe
a given model's answer.

| School | Worldview | Representative | Character |
| :--- | :--- | :--- | :--- |
| **Symbolist** | Effects have definite causes, and the rules can be recovered | **Decision tree** | Deduction and logic; built for explanation |
| **Bayesian** | Causes are probabilistic, written `P(A|B)` | **Naive Bayes** | Strong on uncertainty |
| **Analogist** | Knowledge transfers between similar things | **KNN, SVM kernels** | Requires a stated similarity measure |
| **Connectionist** | Imitate the neuron | **Neural networks** | Pattern lives in the connections |
| **Evolutionary** | Selection finds the fit | **Genetic algorithms** | No prior model structure needed |

The schools mix freely, and the point of listing them is not taxonomy:

> **Five schools are five different answers to "why should I believe this result",
> not five different algorithms.**

That framing runs through the whole module. A decision tree's answer is reproducible by
construction. A Bayesian answer carries a probability and never claims certainty. A neural
network's answer is neither, which is why sections 12 and 13 spend their time on
verification rather than on architecture.

### 1.4 Classification and regression are one thing seen twice

| | Classification | Regression |
| :--- | :--- | :--- |
| **Output** | A discrete label | A continuous value |
| **Usual loss** | Cross-entropy | Mean squared error |

Three connections matter more than the difference:

- **A classifier's output is a probability.** Turning it into a label requires a
  threshold, and that threshold is a business decision, not a model property. Section 10
  measures exactly how much moves when you change it — and what refuses to move.
- **Many models switch tasks by changing only the output layer.** Softmax for classes, a
  linear unit for values. Naive Bayes, trees, SVM, logistic regression and neural networks
  all do this.
- **A regression target can be binned into classes**, at the cost of precision. The vehicle
  pricing case below stays a regression for that reason.

> Both are learning a map `f(X) → Y` and minimising the gap between prediction and truth.
> **They differ in the shape of the output space and share everything else.**

---

## 2. Choosing a model

### 2.1 The task picks the family, not you

Vehicle price is a real number, so the case in sections 6 to 11 is a regression. That is
not a preference; nothing else fits.

### 2.2 Within a family, three axes

| If you need | Pick |
| :--- | :--- |
| **Explainability** | Linear or logistic regression, a shallow decision tree |
| **Accuracy** | XGBoost / LightGBM / CatBoost |
| **Capacity on a lot of data** | A neural network |

The criteria are identical for classification and regression; only the linear model's name
changes.

**The trade is between accuracy and stability.** A complex model has a higher ceiling and
a larger overfitting risk; a simple model has a lower ceiling and holds its behaviour
steady across refreshes. Anything whose deliverable is a *report* should lean simple: a
report has to be explainable this month and reproducible next month.

### 2.3 The real criterion for reaching for a network

Not the row count, and not the industry:

> **How complicated is the regularity you are trying to capture?**

| The regularity is | Better served by |
| :--- | :--- |
| Simple, clear, expressible as rules | A tree; it will find it faster |
| Complex, implicit, full of interactions | A network; it is built for crossed features |

Two corollaries worth stating:

- **Ten to twenty features is not network territory.** With that few inputs and a
  moderate number of rows, a boosted tree will beat a network and be explainable as well.
- **Tens of thousands of features is not tree territory.** A tree evaluates every feature
  at every split, so the feature count sits inside its cost. A network's first layer is one
  matrix multiply; more features just makes that matrix wider.

### 2.4 A rule of thumb for sample size

```
required samples  ∝  (number of features)²

  10 features →   1 000 samples
 100 features → 100 000 samples
```

Read it backwards and it becomes useful advice: **with a thousand rows, do not put a
hundred features in front of a model.** That is handing it an unlimited number of ways to
memorise the training set. Section 5.5 treats the same fact from the other end.

### 2.5 Depth is the explainability dial

"Make it explainable" is a wish until it becomes a parameter. For a decision tree it is
`max_depth`, and the useful range is **3 to 7, with 4 as the working default**. Depth 4
means every rule is at most four conditions long, which is the length a person can read
aloud.

### 2.6 Underfitting has two different cures

> For a **linear model**, underfitting means **add features**.
> For a **tree ensemble**, it means **add trees**.

This is the hinge that explains section 5. If the way to improve a boosted model is to
grow more trees, and every tree costs time and memory, then *making each tree cheaper to
build* becomes the valuable problem. All three algorithms in section 5.3 exist to attack
that one cost.

### 2.7 Where the large model goes

> **Large models are better at explaining a result than at producing one.**

Prediction on structured tables still belongs to the tree ensembles. The large model's
place in this pipeline is writing the code and narrating the output — a division of labour
that shows up three separate times in the sections below.

---

## 3. Industry context

The techniques in this module are not exercises; they are the working stock of several
industries. This section is a short map, kept short deliberately — none of it is
implemented in the scripts.

### 3.1 Finance is where structured prediction is densest

Accounts, transactions and credit histories are already tables, and the business questions
are already "predict a number" or "decide a class". A single institution's departments map
onto the same handful of algorithms with a different target variable each time:
personal banking (churn, propensity, pricing), corporate banking (credit assessment),
channel operations (siting, activity), risk management (default, fraud), treasury
(rate and price forecasting), finance (cash-flow forecasting).

> **Seven departments are not seven algorithms. They are seven different `y`.**

### 3.2 The two-window framing

Most of those targets share one construction, and it is worth knowing by name:

```
        observation window            index date         performance window
  ├────────────────────────────┤          │        ├────────────────────────────┤
   behaviour over a fixed span                       outcome over a fixed span
        →  features X                                       →  label Y
```

Train on history to learn `X → Y`; at inference, take the last N periods of behaviour and
predict the next N. **Swap the label and the same frame answers a different question** —
churn, purchase propensity, default, campaign response. The framing's one assumption is
that the historical regularity still holds; when it stops holding, accuracy drops and
nothing in the pipeline announces it.

### 3.3 Segment-level personalisation

Per-user personalisation needs a rich catalogue and frequent interaction. Where a business
has ten products and a user who appears twice a month, individual profiles cannot be
estimated and **group-level recommendation is the correct fallback** — cluster the users,
target the clusters. It is a real technique with real effects, not a consolation prize.

### 3.4 Scorecards: when the deliverable is a table

Credit scorecards are the clearest case of a model chosen for its *output shape* rather
than its accuracy. The deliverable is a printed table that a person applies by hand
— *postgraduate degree, +10; local residency, +10; debt ratio under 10%, +9* — so the
model has to be one that decomposes into addition. A network cannot produce "+12 points
for a bachelor's degree" at all, and is out on form, not on score.

Two details from real scorecards are more instructive than the technique:

- **A mortgage often scores higher than an outright-owned home.** It looks backwards until
  you notice that a mortgage means the applicant already passed someone else's credit check
  and has a payment record to inspect.
- **The dynamic range is narrow.** A typical card runs 630–700 with a base of 650. It is a
  fine-grained ranking instrument inside a narrow band, not a wide-range predictor.

### 3.5 Turning a curve into a row

The most transferable idea in the industrial applications: a bolt-tightening trace, a
torque curve over time, gets converted into **a row of features** — count of consecutive
rising points, count of drops beyond 30%, presence and length of the plateau — and then a
tree model classifies it.

Not a sequence model. Not image recognition. The reason is that the deliverable is a
**revised inspection standard**, and a standard has to be something a person can write
down and sign off. The same instinct governs the feature work in section 8.

### 3.6 What decides which tool

Across these settings one division keeps recurring, and it is worth stating once:

| Job | Tool | Because |
| :--- | :--- | :--- |
| **The decision** (will they default, will they answer) | **A small model** — logistic regression, a tree | Must be explainable and stable |
| **Building the model** | A large model | Writes the code |
| **Understanding the request, explaining the result** | A large model | Natural language |

---

## 4. The classic models

Everything in this section is exercised by `05_classifier_toolbox_and_thresholds.py`
except the Bayesian material, which is here because it explains a measurement in that same
script.

### 4.1 Bayes, and why a 99.9% test is not a 99.9% answer

Bayes attacked the **inverse** problem. Forward probability knows the contents of the bag
and asks about the draw. Inverse probability sees the draws and asks about the bag. Real
work is nearly always the second one.

| Term | Meaning |
| :--- | :--- |
| **Prior** | What you believed before seeing this evidence |
| **Posterior** | What you believe after |
| **Conditional** | `P(A\|B)`, the probability of A once B has happened |
| **Likelihood** | Training a probabilistic model = estimating its parameters |

```
posterior  ∝  prior × likelihood
              ─────────────────────
                  evidence
```

**The example worth memorising.** A condition affects 1 in 10 000 people. A test is
99.9% accurate with a 0.1% false positive rate. Someone tests positive. How likely are
they to have it?

Count instead of deriving:

- Of 10 000 people, the 0.1% false positive rate produces about **10 healthy positives**
- The 1 person who has it is detected with probability 99.9% → about **1 true positive**
- So there are ~11 positives, of whom 1 is real ⇒ **about 9%**

```
              P(Bᵢ) · P(A|Bᵢ)
P(Bᵢ|A) = ——————————————————————
            Σⱼ P(Bⱼ) · P(A|Bⱼ)
```

**No amount of test accuracy survives a small enough prior.** The base rate decides how
many false positives there are in absolute terms, and they drown the true ones.

> **This is not a curiosity. It is the same arithmetic as section 10.1**, where predicting
> that nobody ever leaves scores **0.8356 accuracy** on the attrition table without fitting
> anything. When a class is rare, every proportion quoted about it has to be read against
> the base rate first.

**What "naive" means.** The model assumes the input variables are independent given the
class, so a joint conditional becomes a product:

```
P(a₁, a₂, a₃ | c) = P(a₁|c) · P(a₂|c) · P(a₃|c)
```

The assumption is usually false and the model usually works anyway, because the alternative
is estimating a high-dimensional joint distribution from data that cannot support it.

**Its failure mode is worth knowing**: if a feature value never appears with a class in
training, that class's product is **exactly zero** and one unseen value vetoes everything.
Laplace smoothing (`alpha=1`) is the standard fix.

**Which variant to use is decided by the feature type**, and this is the part most often
mixed up:

| Class | Feature type | Typical use |
| :--- | :--- | :--- |
| `GaussianNB` | **Continuous**, roughly normal | Heights, lengths, measurements |
| `MultinomialNB` | **Counts**, multinomial | Word counts or TF-IDF values |
| `BernoulliNB` | **Booleans** | Whether a word occurs at all |

One more property, which section 10.3 measures: **feature scaling does nothing for Naive
Bayes**, because it works with probabilities rather than distances.

### 4.2 Decision trees

> **A decision tree is prior experience written down as reproducible rules.**

You do not design the shape of the tree. You define **one rule for choosing the split
feature**, and the tree grows itself.

**Information → entropy → information gain.**

```
I(X = xᵢ) = − log₂ p(xᵢ)                          information

H(X) = − Σᵢ p(xᵢ) · log p(xᵢ)                     entropy, the expectation of information

g(D, A) = H(D) − H(D|A)                           information gain
```

Entropy measures uncertainty; the gain of a feature is how much uncertainty it removes.
Pick the feature with the largest gain, split, repeat.

> **Lower entropy means a more decided outcome; higher gain means the feature contributes
> more to the decision.**

```python
from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifier(max_depth=4, random_state=SEED)
```

> The call above is the one in `05_classifier_toolbox_and_thresholds.py`. It leaves
> `criterion` at its default, which is **gini, not the entropy the derivation above uses** —
> asking for information gain means passing `criterion='entropy'` explicitly.

| Parameter | Note |
| :--- | :--- |
| `criterion` | `gini` or `entropy` — **the default is gini**, so entropy has to be asked for |
| `max_depth` | The explainability dial from section 2.5 |
| `min_samples_split` | Minimum samples for an internal node to split further |
| `class_weight` | Class weighting |

The property that puts trees in the symbolist column: **the same input always produces the
same output, with no probabilistic step in between.** That is exactly where they part
company with Naive Bayes.

### 4.3 Random forests

**Bagging**: train many weak learners on resampled data and combine them. Classification
votes; **regression averages**.

Two independent randomisations, and both are required:

| Randomisation | How | Effect |
| :--- | :--- | :--- |
| **Rows** | Draw N samples **with replacement** for each tree (a bootstrap sample) | Every tree sees a different training set, with repeats inside it |
| **Columns** | Fix `m << M`; at each split choose the best among a random `m` features | Stops one strong feature from dominating every tree |

Trees are grown to full depth without pruning.

**The error rate is pulled by two forces that fight each other:**

- **Correlation between any two trees** — higher correlation, higher error
- **Strength of the individual tree** — stronger trees, lower error

And `m` moves both at once. Raising it makes each tree stronger *and* makes the trees more
alike. Choosing `m` is choosing where on that trade to sit; `sqrt(n_features)` is the
usual starting point, and 100 trees is the usual starting count.

**Why redundancy can beat precision.** Ten cheap components where at least six must fail
simultaneously give a failure probability around `2.1 × 10⁻²⁸`, at a quarter of the cost of
one precise component. **The whole argument depends on the failures being independent.**

> That caveat is not decoration. **Section 11 measures it and the result goes the other
> way**: three gradient boosting models on the same data agree on their predictions at
> **0.9955** and on their *mistakes* at **0.8319**, and averaging them lands **41.61 MAE
> worse** than the best one alone. Redundancy pays only to the extent the errors are
> independent, and that is a number you can compute before you build the ensemble.

**Out-of-bag error** is a free by-product: each tree misses about a third of the rows, so
those rows serve as its validation set with no extra split.

### 4.4 Support vector machines

**The idea is lifting.** Some sets that no line separates in low dimensions are separated
by a plane in higher ones. Map the features up with a non-linear function and the problem
becomes linear.

**The kernel is the trick that avoids doing it.** The high-dimensional space is only ever
used to compute inner products, so compute the inner product directly:

```
K(xᵢ, xⱼ) = φ(xᵢ) · φ(xⱼ)
```

RBF, polynomial and sigmoid kernels are different implied mappings. **The kernel is a
hyperparameter — it is chosen by hand, not learned**, which makes the analogy to picking an
activation function a fair one.

**Where SVMs sit historically** is the answer to "why learn this now": an SVM is
effectively **a single-hidden-layer network with the kernel playing the part of the
activation**. Deep networks displaced it for large problems; it remains a strong choice for
small samples and for settings that need determinism.

| Class | Note |
| :--- | :--- |
| `SVC` | The standard implementation |
| `NuSVC` | Same, with a parameter controlling the number of support vectors |
| `LinearSVC` | Linear kernel only |

`SVC` classifies; `SVR` regresses.

| Parameter | Meaning |
| :--- | :--- |
| `C` | **Penalty. Larger C punishes misclassification harder** — reduce it when overfitting, raise it when underfitting |
| `kernel` | `rbf` (default), `linear`, `poly`, `sigmoid` |
| `gamma` | Kernel coefficient. **Larger means a more complex boundary** and more overfitting risk |
| `degree` | Polynomial order when `kernel='poly'`, default 3 |
| `probability` | Needed for `predict_proba`; costs extra fitting |

Two practical notes:

- **`nSV`, the support vector count, is worth a glance.** Support vectors are the samples
  sitting on the boundary. If nearly every sample is one, `C` is too small or the features
  were never scaled.
- **`LinearSVC` has no `predict_proba`.** Threshold work (section 10.6) requires
  `SVC(kernel='rbf', probability=True)`.

**Choosing a kernel** follows one comparison:

| Situation | Reading | Choice |
| :--- | :--- | :--- |
| Fewer samples than features | The feature work is already done | **Linear kernel** |
| More samples than features | Crosses are still to be found | **Non-linear kernel** |

The logic: if you already have more features than rows, lifting only creates more ways to
memorise noise. If features are scarce and rows are plentiful, the lift is doing your
feature crossing for you.

**And one hard prerequisite.** An SVM computes distances, so **the features must be
scaled**. Section 10.3 puts a number on skipping it: **AUC 0.4973 unscaled against 0.7789
scaled** — not a degradation, a model that does not work.

### 4.5 Logistic regression

Four clauses, and they form one chain — distributional assumption, objective, optimiser,
task:

> **Assume a Bernoulli distribution → maximise the likelihood → solve by gradient descent
> → produce a binary classification.**

The second assumption is that the positive-class probability comes from a sigmoid. The
consequence is that **the decision boundary stays linear**; curved boundaries are the
kernel methods' territory, which is precisely the division of labour with section 4.4.

The derivation runs: write the probability of both classes as one expression → take the
likelihood → take the log-likelihood → maximise it, which is minimising its negative by
gradient descent.

```python
from sklearn.linear_model import LogisticRegression

LogisticRegression(max_iter=2000, random_state=SEED)              # the toolbox run
LogisticRegression(max_iter=4000, C=1.0, random_state=SEED)       # the coefficient read-back
```

| Parameter | Meaning |
| :--- | :--- |
| `penalty` | `l1` or `l2`, default `l2` |
| `C` | **The inverse of the regularisation strength.** Smaller C means *more* regularisation |
| `max_iter` | Default 100, which is often too few for unscaled features |
| `tol` | Stop when the change falls below this |

> **`C` being an inverse is the single most reversed intuition in this API.**

**`predict` and `predict_proba` are the whole of section 10.** For a binary task
`predict_proba` returns two columns, so working code is always `predict_proba(X)[:, 1]`.
**Holding the probability instead of the label is what makes a threshold available to
move**, and moving it turns out to matter more than the choice of model.

**Reading the coefficients is a second, separate use of the model** — not "how accurate is
it" but "what did it find". Positive coefficient, the feature pushes the event; negative,
it holds it back. Two prerequisites decide whether that reading is legitimate:

- **Encoding.** Label-encoding a job role to 0–8 turns "positive coefficient" into
  *"higher job-role number means more attrition"*, and that ordering is alphabetical.
  **One-hot encoding is required if the coefficients are going to be read.**
- **Scale.** Fitted on standardised columns, a coefficient is a weight **per standard
  deviation**, not per unit. Section 10.8 measures what happens when that is ignored: the
  heaviest raw weight in the generating formula lands **twelfth** in the fitted ranking,
  purely because it is a yes-or-no column whose entire range is one step.

**Strengths and limits.**

| Strengths | Limits |
| :--- | :--- |
| Simple form; **highly interpretable** | Lower accuracy ceiling |
| Feature weights are directly readable | Struggles with class imbalance |
| **A good engineering baseline** — with decent features it is not far off | Cannot fit non-linear data unaided |
| Fast; cost scales with feature count only | Cannot select features by itself — pair it with a tree model for that |
| **Outputs probabilities**, so the cut point stays adjustable | Multiclass needs softmax |

> ⚠️ **Take "lower accuracy" with a measurement.** In section 10.1 logistic regression has
> **the highest AUC of all nine models on the attrition table (0.8078)**, above every
> gradient boosting library. A low ceiling only shows up when the data has enough structure
> to reach it.

Finally, the name. "Regression" came from the observation that tall parents have sons who
*regress* toward the mean height. It described that statistical phenomenon first and was
borrowed later for fitting numeric relationships — which is why a classifier ended up
carrying it.

---
## 5. Ensembles and gradient boosting

All three libraries in this section are used by the scripts: CatBoost in section 8,
all of XGBoost, LightGBM, CatBoost and NGBoost in section 10, and the first three plus a
nearest-neighbour model in section 11.

### 5.1 Three ways to combine

> **Combine weak learners into a strong one.** The interesting word is *combine* — the
> three families differ only in how.

**Bagging.** Resample the data with replacement, train a model per sample, then vote
(classification) or average (regression). The samples overlap, which is the point: the
models are independent enough for their errors to cancel.

**Stacking.** Train **different** models on the full training data, collect their
predictions, and **train a second model on those predictions**.

```
Classifier 1 ──prediction──┐
Classifier 2 ──prediction──├──►  meta model  ──►  final output
Classifier 3 ──prediction──┘
```

**Boosting.** Same weak learner, fitted **in sequence**, each one addressing what the
previous ones got wrong.

| Algorithm | Mechanism |
| :--- | :--- |
| **AdaBoost** | Analyse the errors and **give misclassified samples more weight** |
| **Gradient boosting** | Minimise the loss by **gradient descent** |

```
train a weak learner and add it to the ensemble
update the training set (weights, or targets) based on what the ensemble already predicts
```

The parenthesis maps exactly onto the two rows: **AdaBoost updates sample weights, gradient
boosting updates the target.**

The clearest illustration of the second one: predicting an age of 24, the first tree says
18, the second corrects by +5, the third by +0.5, and the answer is **18 + 5 + 0.5 = 23.5**.

> Three things follow from that one line:
> 1. **Each tree learns the residual**, not the label.
> 2. **The corrections shrink** — there is progressively less left to learn.
> 3. **The output is a sum, not a vote.** That is why a boosted classifier's base learners
>    are still *regression* trees: labels cannot be added, scores can.

**Bagging and boosting attack different halves of the error.**

| | **Bagging** | **Boosting** |
| :--- | :--- | :--- |
| Structure | **Parallel** | **Sequential** |
| Training sets | **Independent** | **Depend on the previous model** |
| Reduces | **Variance** | **Bias** |

The reason, in plain terms: bagging's members are all equally short-sighted (same bias) but
they miss in different directions, so averaging cancels the jitter and leaves the
short-sightedness. Boosting spends every step correcting what the last step could not see,
which lowers bias, but the members' errors accumulate in one direction, so the jitter
stays.

> ⇒ **Boosting is intrinsically the more overfitting-prone of the two.** That is why
> XGBoost writes a regularisation term into its objective at all, and why section 5.5
> exists.

### 5.2 XGBoost: from an objective function to a split criterion

The derivation is the intellectual centre of the whole family. The chain is:
**define the objective → expand it to second order → regroup by leaf → solve for the
optimal leaf weight → substitute back to get a structure score → derive the split gain.**

**The base learner is a regression tree, and each leaf carries a score** (`+2`, `+0.1`,
`−1`), because those scores are going to be summed.

**The prediction and the objective:**

```
ŷᵢ = Σ(k=1..K) f_k(xᵢ)

Obj = Σᵢ l(yᵢ, ŷᵢ) + Σ_k Ω(f_k)
       └── fit the data ──┘  └── punish complexity ──┘
```

A single tree is written `f_t(x) = w_q(x)`, and the notation is worth unpacking:
**`q` is the structure** — which leaf a sample lands in — and **`w` is the table of leaf
scores**. A tree is an assignment function plus a lookup table, and the derivation solves
for the table.

```
Ω(f_t) = γT + ½ λ Σⱼ wⱼ²
```

| Symbol | Meaning |
| :--- | :--- |
| **T** | Number of leaves |
| **γ** | The cost of introducing one more leaf |
| **λ** | L2 coefficient on the leaf scores |

For a three-leaf tree scoring `+2 / +0.1 / −1`, that is `γ·3 + ½λ(4 + 0.01 + 1)`.
**Two different penalties in one expression: `γ·3` charges for how finely the tree is cut,
`½λ·5.01` charges for how extreme its predictions are.**

**One tree is added at a time**, so at round *t*:

```
Obj⁽ᵗ⁾ = Σᵢ l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾ + f_t(xᵢ)) + Ω(f_t) + constant
```

The constant collects the earlier trees' regularisation, which is already fixed.

**Second-order Taylor expansion** treats the new tree's output as the increment:

```
gᵢ = ∂ l(yᵢ, ŷ⁽ᵗ⁻¹⁾) / ∂ŷ⁽ᵗ⁻¹⁾               first-order gradient
hᵢ = ∂² l(yᵢ, ŷ⁽ᵗ⁻¹⁾) / ∂(ŷ⁽ᵗ⁻¹⁾)²           second-order (Hessian)

Obj⁽ᵗ⁾ ≈ Σᵢ [ gᵢ·f_t(xᵢ) + ½ hᵢ·f_t(xᵢ)² ] + Ω(f_t) + constant
```

**What the gradient is, concretely.** For squared error it is `2(ŷ − y)`. Picture descending
a hill: **steep near the top, shallow near the bottom**. A larger absolute gradient means a
larger error on that sample, which means more room left to learn from it.

**And the Hessian:** it is the rate at which the gradient changes, so it governs step size.
**First order tells you which way to go; second order tells you how far you dare.**

> Hold on to the gradient statement — **GOSS in section 5.3 is built entirely on it.**

**Regroup by leaf.** With `Iⱼ` the set of samples in leaf *j*:

```
Gⱼ = Σ_{i∈Iⱼ} gᵢ          Hⱼ = Σ_{i∈Iⱼ} hᵢ

Obj⁽ᵗ⁾ = Σⱼ [ Gⱼwⱼ + ½ (Hⱼ + λ) wⱼ² ] + γT
```

Differentiate and solve:

```
wⱼ* = − Gⱼ / (Hⱼ + λ)

Obj = − ½ Σⱼ Gⱼ² / (Hⱼ + λ) + γT
```

> **This is the heart of the algorithm.** Once the structure is fixed, **every leaf's score
> has a closed form** — no further iteration. Tree building therefore splits cleanly into
> two jobs: **the structure has to be searched; the leaf values are a formula.**
>
> And note **where λ sits: in the denominator.** Regularisation is not a pruning step
> applied afterwards; it lives inside the optimal solution.

**The structure score** is that objective, and **smaller is better** — the negative sign
means a larger `Gⱼ²/(Hⱼ+λ)` drives the objective further down.

**The split criterion follows from it.** Exhaustive search over structures is impossible, so
be greedy and score one split at a time:

```
Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ
         └─ left ─┘      └─ right ─┘     └─ not splitting at all ─┘   └ new leaf's cost
```

**Split only when the gain is positive**, and take the largest.

> **The two regularisation parameters have physical positions in this formula.**
> `λ` is in all three denominators — **it compresses leaf scores**.
> `γ` is the `− γ` at the end — **it is the admission price of a split**. A split that
> genuinely lowers the loss, but by less than `γ`, should not be made.

**Why this is a real departure from a classic decision tree**: CART splits on Gini or
information gain, **which have nothing to do with the loss being optimised**. XGBoost's
gain is derived from the objective itself, so it measures the only thing that matters —
how much this particular cut lowers the thing you are minimising.

**The approximate split.** Enumerating every distinct value of a continuous feature is
expensive and encourages overfitting, so bucket the values and consider only bucket
boundaries. Ten thousand candidates become a hundred and twenty-seven; cost drops from
`O(n)` to `O(buckets)`.

> ⚠️ **Histogram-based splitting is XGBoost's own** (`tree_method='hist'`), from the
> original paper. It is frequently misattributed. The accurate statement is that
> **LightGBM made it the default and only path and then stacked two more algorithms on
> top of it.**

**What the algorithm gets you, and what it costs:**

- Complexity is **inside the objective**, so overfitting is resisted by construction
- Second-order expansion **speeds up optimisation**
- Splits use an **approximate greedy** search
- Base learners can be trees **or linear models**
- **Parallelism is over features, not over trees.** Boosting itself must stay sequential;
  what runs in parallel is the per-feature gain computation inside one tree, backed by a
  pre-sorted block layout that is reused every round
- **Cost: a lot of parameters, and tuning is slow**

> That pre-sorted block is exactly what LightGBM replaces next.

### 5.3 LightGBM: three cuts at one product

**The problem it was built for.** Neural networks train in mini-batches, so their data
never has to fit in memory at once. **Gradient boosting has to sweep the whole training set
at every split** to know where to cut. Hold it in memory and the data size is capped; stream
it and the repeated reads dominate the runtime. That dilemma — not accuracy — is what the
three algorithms address.

**Published comparisons** put memory at roughly a sixth and training time at roughly a
tenth, with accuracy comparable or slightly better. One detail in those tables is more
instructive than the headline: **the approximate-split variant of XGBoost uses no less
memory than the exact one.** Approximating the split does not shrink the data. What shrinks
it is storing a bucket index instead of a float.

| Aspect | Result |
| :--- | :--- |
| Accuracy | **Comparable** |
| Training speed | **~10× faster** |
| Memory | **~6× smaller** |
| Missing values | **Both handle them** |
| **Categorical features** | **XGBoost needs one-hot; LightGBM supports them directly** |

That last row is a thread: **XGBoost cannot, LightGBM can via `categorical_feature`,
CatBoost made it the headline feature.**

**The cost model turns optimisation into arithmetic:**

```
model cost      = number of trees × leaves per tree × cost per leaf
cost per leaf   = number of features × number of candidate splits × number of samples
```

| Algorithm | Which factor it attacks |
| :--- | :--- |
| **Histogram** | **candidate splits** |
| **GOSS** | **samples** |
| **EFB** | **features** |

```
LightGBM = XGBoost + Histogram + GOSS + EFB
```

> The first two factors — tree count and leaf count — **are yours to set, and shrinking
> them costs accuracy directly.** The remaining three are the only ones that can be
> attacked without paying for it, and each has an algorithm. This is not three unrelated
> tricks; it is one product with each factor pressed once.

**Histogram: store a bucket index, not a float.**

Continuous values are discretised into `k` bins (`max_bin`, commonly 255), a histogram of
width `k` is accumulated in one pass, and splits are searched over bins.
**Candidates drop from `n − 1` to `k − 1`.**

A worked instance — eight samples, three bins:

| i | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **xᵢ** | 0.1 | 2.1 | 2.5 | 3.0 | 3.0 | 4.0 | 4.5 | 5.0 |
| **gᵢ** | 0.01 | 0.03 | 0.06 | 0.05 | 0.04 | 0.7 | 0.6 | 0.07 |
| **hᵢ** | 0.2 | 0.04 | 0.05 | 0.02 | 0.08 | 0.02 | 0.03 | 0.03 |

```
bin1 = {1,2,3}   G = 0.10   H = 0.29
bin2 = {4,5,6}   G = 0.79   H = 0.12
bin3 = {7,8}     G = 0.67   H = 0.06
```

**Three gains from one change:**

1. Candidate splits collapse from thousands to `k − 1`
2. **A bucket index fits in one byte where a float needs four** — this is where the
   memory reduction actually comes from
3. **The sums are additive**, so the right child's histogram is the parent's minus the
   left's; no second pass

The cost is that splits can only fall on bin edges. That loses a little precision and buys
a little regularisation in exchange — no single extreme value gets its own cut.

**GOSS: sample only the part that has been learned.**

Keep every large-gradient sample; sample the small-gradient ones at a fixed rate. The
justification is section 5.2's statement about gradients: **the gain comes overwhelmingly
from samples the model is still getting wrong.**

On the same eight samples, with a threshold of 0.1 and a 1/3 sampling rate:

- Kept whole: **i6 (g=0.7), i7 (g=0.6)**
- Sampled in: **i2, i4**
- **Eight samples become four**

**Sampled small-gradient samples are then multiplied by 3** — one stands for three:

| bin | bin1 | bin2 | bin3 |
| :--- | :--- | :--- | :--- |
| **N** | **3 = 0 + 1×3** | **4 = 1 + 1×3** | **1 = 1 + 0×3** |
| **G** | 0.09 | 0.85 | 0.6 |
| **H** | 0.12 | 0.08 | 0.03 |

> **Read `0 + 1×3` correctly**: the left term counts retained large-gradient samples, the
> right counts sampled small-gradient ones times the compensation factor. **That factor is
> the reciprocal of the sampling rate, and it exists to keep the expected gradient sum
> unchanged** — otherwise the gains of a sampled branch would be systematically smaller and
> incomparable with an unsampled one.

**And GOSS is not a one-off downsample.** Every boosting round recomputes every gradient,
so a sample dropped this round can be a large-gradient sample next round.
**It is not discarding unimportant rows; it is discarding rows that have already been
learned** — which changes from round to round.

**EFB: bundle features that are never non-zero together.**

Sparse features are mostly zero. If two of them are never non-zero on the same row, they
can be **re-encoded into one column** by offsetting one of them past the other's range.
One-hot columns are exclusive by construction, which is why LightGBM can map a category
directly onto a bin and skip one-hot entirely.

A worked instance — two sparse features over 100 rows:

```
feature1:  0 → 90 rows    1 → 6 rows    2 → 4 rows
feature2:  0 → 95 rows    1 → 4 rows    2 → 1 row
```

Drop feature2's zeros, offset its values by 2 so they become 3 and 4:

| bundled value | 0 | 1 | 2 | 3 | 4 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **rows** | **90 − 5 = 85** | 6 | 4 | 4 | 1 |

Total 100, conserved. **The 5 rows subtracted are feature2's non-zero rows**, which sat at
zero in feature1 and now carry a 3 or a 4 instead.

> **Two easy misreadings**: `90 − 5 = 85` is a **row count**, not a percentage of zeros;
> and the **dimensionality went from 2 to 1**, not from 100 to 85.

**EFB is the one of the three with a precondition.** Strict exclusivity is rare in
practice, so implementations accept a conflict rate rather than demanding it.
Bundling correlated features costs accuracy.

| | Factor removed | Method | Cost |
| :--- | :--- | :--- | :--- |
| **Histogram** | candidates (`n−1` → `k−1`) | Discretise into `k` bins | Splits land on bin edges only |
| **GOSS** | samples (8 → 4) | Keep large gradients, sample and compensate the rest | Sampling variance, held in check by the factor |
| **EFB** | features (2 → 1) | Offset and merge mutually exclusive sparse columns | **Requires near-exclusivity** |

### 5.4 CatBoost

**CatBoost = Categorical + Boost.** Three properties separate it from the other two:

- **Target statistics.** Categories are replaced not by one-hot columns and not by an
  arbitrary integer, but by **a statistic of the target within that category**.
- **Ordered boosting.** Computing that statistic from the whole dataset leaks the label
  into the feature. CatBoost computes it using **only the rows that precede the current one
  in a permutation** — that is what "ordered" means.
- **Oblivious trees.** Every node at a given depth uses the **same split condition**, so the
  tree is symmetric. That caps capacity — a built-in regulariser — and makes inference very
  fast.

> **The first two are the same problem section 9 measures.** Target encoding's danger is
> not the technique, it is the number of rows sharing a key: from **0.00 MAE of leakage at
> 3 000 rows per key to 2945.40 at 1.4 rows per key.** Ordered boosting is the principled
> answer to that.

**A published three-way comparison on a dataset with real categorical columns:**

| | XGBoost | LightGBM (with cat. index) | CatBoost (with cat. index) |
| :--- | ---: | ---: | ---: |
| **Train AUC** | 0.999 | 0.999 | **0.887** |
| **Test AUC** | 0.789 | 0.772 | **0.816** |
| **Gap** | **0.210** | **0.227** | **0.071** |
| Prediction time | 184 s | 156 s | **2 s** |

> **The column to read is the gap, not the winner.** Two of the three memorised the
> training set almost perfectly and generalised at 0.78. The third scored lower on training
> and higher on test. **That is what "resists overfitting" means concretely, and it is the
> best possible introduction to section 5.5: a high training score is not good news; the
> distance between training and test is the news.**

**With two conditions attached**, both of which are usually omitted when this result is
quoted:

- The advantage appears **only when there are categorical features**, and only when
  `one_hot_max_size` has been tuned
- Without exploiting them, the same library **comes last** on the same data

On a dataset of raw pixel values — no categorical columns at all — the ordering reverses
and CatBoost is the least accurate of the three. **The claim is conditional, and the
condition is in the data.**

| Parameter | Note |
| :--- | :--- |
| `iterations` / `depth` / `learning_rate` | Tree count, depth, step |
| `l2_leaf_reg` | L2 coefficient |
| **`one_hot_max_size`** | **Categories at or below this go one-hot; above it, target statistics** |
| `loss_function` | RMSE by default for regression, Logloss for classification |
| **`cat_features`** | **Column positions, not names** |
| `od_type` / `od_wait` | Early stopping, named as an **overfitting detector** |

Two operational notes:

- **`cat_features` taking positions rather than names** means the list silently goes stale
  whenever an upstream `drop` changes the column order. It is an implicit dependency on the
  preprocessing sequence.
- **Naming early stopping "overfitting detector"** is not cosmetic — it is the only one of
  the three libraries that treats the check as a component with a name.

**Where each library belongs:**

- **Many categorical features → CatBoost**
- **Throughput matters → LightGBM**
- **All three have large parameter surfaces; start from the defaults**
- Because it resists overfitting, CatBoost tolerates a **larger `n_estimators`**

### 5.5 Controlling overfitting

**What overfitting looks like** is one comparison: **training MAE 400, test MAE 800**.
The model memorised structure that does not generalise.

The version that lands harder comes from a submission log rather than a definition:

| | Value |
| :--- | ---: |
| Best single model, validation MAE | **493.31** |
| Ensembled, validation MAE | **492.49** |
| The two corresponding scores on held-out data | **780.21 → 826.21** |

**The ensemble improved validation by 0.82 and made the held-out score 46 points worse.**

> **Tenths of a point squeezed out of a validation set are frequently just a fit to that
> validation set.** Section 9 is the systematic treatment of that failure; this is the
> anecdote that motivates it.

**Regularisation, by model family:**

| Family | Lever | Working range |
| :--- | :--- | :--- |
| **Trees** | `max_depth`, minimum samples per leaf | **depth 5–8**, **10–20 samples per leaf** |
| **Networks** | Dropout plus L2 | **dropout 0.2–0.5** |
| **Linear** | L1 / L2 strength | `C` smaller means stronger |

A conservative boosted-tree configuration looks like:

```python
params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "learning_rate": 0.01,     # smaller is more conservative
    "max_depth": 6,
    "subsample": 0.8,          # row sampling per tree
    "colsample_bytree": 0.8,   # column sampling per tree
    "seed": 42,
}
```

**A cross-library caution on the minimum-leaf lever.** scikit-learn's trees take
`min_samples_leaf`; XGBoost has no such parameter and its nearest equivalent is
`min_child_weight`, a **minimum sum of sample weights**. For regression that is
approximately a sample count. **For classification the weight is `p(1−p)`**, so a sample
the model already predicts confidently contributes almost nothing — the same
`min_child_weight=1` can correspond to dozens of rows. The two are not interchangeable.

**Early stopping** is a regularisation technique: **stop when validation stops improving
rather than training to convergence.** Set the round budget high and let it stop —
`num_boost_round=10000` with `early_stopping_rounds=200` is a standard pairing; for
networks, `patience` of 10–20 epochs.

> ⚠️ **One default differs between the two worlds and it matters.**
> Boosted-tree libraries **roll back to the best iteration by themselves**. Keras stops
> where it stopped — on the already-degrading final epoch — **unless
> `restore_best_weights=True` is passed.** Same concept, opposite default.

**Three layers, not one.** Regularisation is the model layer. The **data** layer — more
data, honest cross-validation, and **removing leaked features** — is treated separately in
section 9, because it turns out to be where the large errors live. The **business** layer
— plausible output ranges, monotonicity constraints — is outside this module's scope but
belongs on the list.

---
## 6. Datasets built from a known formula

`01_build_tabular_datasets.py` writes four tables. Every later result in this document is
scored against what this script put into them.

### 6.1 Why generate rather than download

A downloaded dataset can tell you a model scored 801. It cannot tell you whether 801 is
close to the best achievable, whether the model found the real drivers, or whether a
feature that ranked highly deserved to. **A generated dataset can, because the answer was
written down first.**

The four tables:

| File | Shape | Purpose |
| :--- | :--- | :--- |
| `vehicle_listings.csv` | **30 000 × 30** | Regression; the case that runs through sections 7–11 |
| `vehicle_holdout.csv` | **10 000 × 30** | Never used for any fit; the arbiter in section 9 |
| `employee_attrition.csv` | **1 800 × 33** | Imbalanced binary classification, **16.6% positive** |
| `speaker_acoustics.csv` | **3 200 × 21** | Near-separable binary classification, balanced |
| `property_valuation.csv` | **506 × 13** | Small dense regression for the hand-written network |

### 6.2 The pricing formula

`price` is produced by a formula that the script prints on every run:

| Term | Value |
| :--- | :--- |
| Brand tier base | 6 500 – 34 000 across ten tiers |
| Depreciation | **value halves every 6 years** |
| `power` | **+41 per unit**, capped at 400 |
| `odometer_km` | **−430 per unit** |
| `gearbox` automatic | **+1 900** |
| `damage_flag == 0` | **× 0.82** |
| `v_0` … `v_4` | weights **2600, −1800, 1200, 900, −650** |
| **`v_5` … `v_14`** | **no effect at all — pure noise columns** |
| Residual noise | **σ = 900** |

Two consequences are used repeatedly:

- **Fifteen anonymous columns look identical and only five of them matter.** Sections 7.5
  and 8.4 both check whether that is recovered.
- **σ = 900 sets a floor.** No model can beat `900 × √(2/π) ≈ 718` MAE on this data.
  Section 8.1's measured 801.35 is therefore **11% above the best achievable**, which is a
  statement, where "801" alone is not.

### 6.3 The columns

```
listing_id  reg_date  list_date  brand  model_code  body_type  fuel_type  gearbox
power  odometer_km  damage_flag  region_code  seller  offer_type
v_0 … v_14  price
```

| Group | Columns | What they need |
| :--- | :--- | :--- |
| Numeric | `power`, `odometer_km` | Outlier treatment, scaling for distance models |
| Categorical | `brand`, `model_code`, `body_type`, `fuel_type`, `gearbox`, `region_code` | Encoding |
| Temporal | `reg_date`, `list_date` | Parse, then derive age and calendar parts |
| Binary | `damage_flag`, `seller`, `offer_type` | `offer_type` is constant by construction |
| Anonymous | `v_0` … `v_14` | Unknown meaning; five of them are real |

### 6.4 The classification tables are generated the same way

`employee_attrition.csv` comes from a stated log-odds model:

```
OverTime               +1.25        YearsAtCompany      −0.085
MaritalStatus Single   +0.85        MonthlyIncome       −0.000105
BusinessTravel Freq.   +0.55        JobSatisfaction     −0.24
DistanceFromHome      +0.030        JobInvolvement      −0.20
NumCompaniesWorked    +0.11         WorkLifeBalance     −0.17
Age                   −0.019        StockOptionLevel    −0.22
```

**The intercept is solved for, not guessed.** A first attempt at hand-picking it produced a
1.1% positive class, because every centred term drags the log-odds down; bisecting for a
target rate of 0.16 lands the file at **16.6% positive**. The class balance is a parameter
of the file rather than an accident of the coefficients.

Two columns are constant on purpose — `EmployeeCount` is always 1, `StandardHours` always
80 — so that section 10.5 can measure what a model does with a feature carrying no signal.

`speaker_acoustics.csv` is the deliberate contrast: **`mean_fundamental` alone nearly
separates the two classes** (0.1709 against 0.1162). Section 10.2 uses it to show that the
ceiling belongs to the data.

`property_valuation.csv` has **twelve numeric predictors on scales four orders of magnitude
apart**, eleven linear terms and **one saturating `tanh` term on `vacancy_rate`**, so a
linear model cannot be perfect and section 12.6 has something non-obvious to recover.

### 6.5 The file layout carries a trap on purpose

The vehicle files are written **space-separated, with a missing value written as nothing at
all** — two adjacent spaces. The row keeps the correct number of separators. That is the
subject of the next section.

---

## 7. Auditing the load

`02_eda_that_silently_lies.py`. The finding here is not that a bad separator raises an
error. It is that it does not.

### 7.1 The same file, two readers

```python
correct   = pd.read_csv(path, sep=" ")        # one space is one separator
collapsed = pd.read_csv(path, sep=r"\s+")     # a run of whitespace is one separator
```

**Everything normally checked agrees:**

| Check | `sep=" "` | `sep=r"\s+"` |
| :--- | :--- | :--- |
| Shape | (30000, 30) | (30000, 30) |
| Column names identical | yes | yes |
| Sum of first column | 450015000 | 450015000 |
| `price` mean | 12317.58 | 12304.77 |
| Exception raised | no | no |

**A gap of 13 on a mean of 12 300 asks nobody to investigate.**

### 7.2 A check with a known right answer

| Column | Should have | `sep=" "` | `sep=r"\s+"` |
| :--- | ---: | ---: | ---: |
| `gearbox` | **2** | **2** | **264** |
| `damage_flag` | **2** | **2** | **1484** |
| `offer_type` | **1** | **1** | **1844** |

**A gearbox is manual or automatic. A reading that finds 264 kinds has not found a
surprise in the data.**

### 7.3 The mechanism

A missing value is an empty field between two spaces. Split on one space and it is a `NaN`.
Split on a run of whitespace and **it is not there at all** — so from that gap onward,
**every column shifts one place left**:

```
                 body_type  fuel_type   gearbox     power  odometer_km  damage_flag  region_code
one space              2.0        4.0       nan       155         12.9            0         1833
whitespace run         2.0        4.0     155.0      12.9          0.0         1833            0
```

The engine power became the gearbox. The odometer became the power.

**1 888 of 30 000 rows (6.29%) shift.**

### 7.4 Counting missing values catches nothing either

**Both readings report exactly 1 925 missing values.** What changed is which column holds
them:

| Column | `sep=" "` | `sep=r"\s+"` |
| :--- | ---: | ---: |
| `body_type` | 632 | 0 |
| `fuel_type` | 627 | 0 |
| `gearbox` | 666 | 0 |
| **`price`** | **0** | **1888** |
| `v_14` | 0 | 37 |

**The shift propagates to the end of the row, so under the wrong reading the gaps land in
the last column — and the last column is `price`, the label.**

**1 888 training rows lose their target**, and shape, column names, exception behaviour and
total missing count are all unchanged.

> ⇒ **An exploratory pass has to contain at least one count with a known correct answer.**
> Shape, column names, missing totals and means all failed this test.

### 7.5 What a correct read shows

| Item | Measured |
| :--- | :--- |
| Columns with gaps | `body_type` 632 (2.11%), `fuel_type` 627 (2.09%), `gearbox` 666 (2.22%) |
| `price` | min 250, median 11274, max 49311, **skew 0.784** |
| Five anonymous columns most correlated with `price` | **`v_0`, `v_1`, `v_2`, `v_3`, `v_4`** |
| Five that were actually paid into the price | **`v_0` – `v_4`** — **exact match** |
| Strongest noise column | `v_9`, at **0.0049** |

**A correlation sort with no model involved picks the five real drivers out of fifteen
identical-looking columns.** That is what exploratory work is for: producing a claim that
can be refuted before anything is trained.

---

## 8. Feature engineering, and what the model actually used

`03_feature_engineering_and_boosting.py`. CatBoost, 1 200 rounds, depth 8, scored on the
untouched holdout file.

### 8.1 The result, against the floor

| Metric | Measured |
| :--- | ---: |
| Best iteration | 1174 of 1200 |
| **Holdout MAE** | **801.35** |
| RMSE | 1011.17 |
| **R²** | **0.9824** |
| **Floor implied by σ = 900** | **718** |

### 8.2 The pipeline

Time features (age in days and years, calendar parts, season, `km_per_year`, an age
segment), interaction features (`power_per_year`, `power_times_km`, `brand_model`,
`latent_mean`, `latent_spread`), missing and outlier flags, then group statistics on
`brand` computed **from the training rows only**, and frequency encoding.

**Group statistics are the one place a rule has to be enforced by hand.** Recomputing
`brand_price_mean` over a concatenation of train and holdout would put holdout prices into
a training feature. Section 9 measures what that costs.

### 8.3 Two silent defects the pipeline produced

**A binning accident, in two reasonable steps.**

`1 683` rows carry a listing date earlier than the registration date. That is a data entry
error, so the age is clipped at zero — which leaves **1 690 rows sitting at exactly 0.0**.
The age is then cut into five segments.

| Bin edges | Rows falling outside every segment |
| :--- | ---: |
| `[0, 1, 3, 5, 10, 100]` | **1 690** |
| `[-0.01, 1, 3, 5, 10, 100]` | **0** |

`pandas.cut` excludes the leftmost edge, so an age of exactly 0.0 belongs to no interval.

> **Neither step is wrong.** Clipping bad dates is right; binning age is right.
> **What is missing is anyone looking at the pile of zeros between them.**
> Nothing raises, nothing warns — the column quietly gains nulls in **5.6% of rows**, and
> they are precisely the rows that were just repaired.
>
> ⇒ **Print the count falling outside the bins, every time. One line.**

**Nineteen flags, eighteen of them constant.**

"A missing indicator preserves information that imputation destroys" is true and has a
precondition. Measured:

| Flag | Rows marked |
| :--- | ---: |
| `power_outlier` | **150** |
| The other 18 (17 `*_missing` plus `odometer_outlier`) | **0** |

The gaps in this file are in three *categorical* columns; the missing flags were built over
the *numeric* ones. The odometer clip was set at 31.5 while the data maxes out at 15, so
that line never touched a row.

> **A missing indicator on a column that is never missing is a column of zeros with a
> descriptive name.** Count `isna().sum()` before building the flag and `nunique()` after.

### 8.4 The audit

| Item | Measured |
| :--- | ---: |
| Features handed to the model | **68** |
| **Importance exactly 0.0** | **21 (31%)** |
| Importance below 0.05 | 8 |
| **Features carrying 95% of total importance** | **18** |

The twenty-one never used:

```
17 × *_missing   odometer_outlier   offer_type   seller   is_new
```

**Did it find the real drivers?** The twelve strongest:

| Feature | Importance | Traces to a term in the formula |
| :--- | ---: | :--- |
| `power_per_year` | 16.272 | derived from `power` |
| **`v_0`** | 13.312 | **yes** (weight 2600, the largest) |
| **`v_1`** | 8.225 | **yes** (−1800) |
| `brand_price_median` | 7.475 | restatement of the brand tier |
| **`brand`** | 6.302 | **yes** |
| `brand_price_std` | 4.997 | restatement of the brand tier |
| **`damage_flag`** | 4.889 | **yes** (× 0.82) |
| **`odometer_km`** | 4.488 | **yes** (−430) |
| `brand_price_mean` | 3.990 | restatement of the brand tier |
| **`v_2`** | 3.788 | **yes** (1200) |
| `brand_model` | 3.563 | derived from `brand` |
| `age_segment` | 3.441 | derived from depreciation |

**Eight of twelve map onto a term in the generating formula or restate one.**

And the ten anonymous columns that were never paid into the price rank
**33, 27, 38, 30, 35, 31, 37, 29, 32, 36** — every one of them below 27th.
**The five real ones sit near 2nd, 3rd and 10th.** Fifteen identically-shaped columns did
not fool it.

A control was planted as well: a meaningless combination adding engine power to a model
identifier scores **0.1311**, outside the top forty. **A feature does not get used because
you built it.**

> **"I engineered seventy features" is a statement about the pipeline.
> The string of zeros in `get_feature_importance()` is the statement about the model.**

---

## 9. Leakage and split discipline

`04_leakage_and_split_discipline.py`. Three leaks, each run twice — honest and leaky — and
each scored on a holdout that took part in no fit.

### 9.1 An honest baseline

| | Validation MAE | Holdout MAE | Gap |
| :--- | ---: | ---: | ---: |
| Honest baseline | 1014.90 | 1034.77 | **−19.87** |

**The two agree. That is what a validation score is for**, and the gap is the column to
watch in everything below.

### 9.2 Fitting the scaler before the split — a negative result

The scaler is fitted on train and holdout together, then the split happens. Trees are
indifferent to scale, so this is measured with a nearest-neighbour model.

| | Validation MAE | Holdout MAE |
| :--- | ---: | ---: |
| Honest | 4880.29 | 4957.72 |
| **Leaky** | 4880.06 | 4957.81 |

**The leak is worth 0.23 MAE.**

> **Worth reporting exactly because it is small.** A minimum and a maximum are two numbers
> per column; handing them over leaks almost nothing. **This is the leak most often caught
> in review, and it is not the one inflating anyone's score.**

### 9.3 Target encoding, at four cardinalities

Encoding a key by the mean price within that key. If the mean includes the row being
encoded, part of that row's own label flows back into its features — and **how large a part
is decided by how many rows share the key.**

| Key | Distinct keys | Rows/key | Honest val | Leaky val | **Leaked** | Leaky holdout |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `brand` | 10 | 3000.0 | 1014.90 | 1014.90 | **0.00** | 1034.77 |
| `model_code` | 120 | 250.0 | 1018.82 | 1014.12 | **4.70** | 1036.41 |
| `region_code` | 3997 | 7.5 | 1034.97 | 1011.90 | **23.08** | 1052.15 |
| **`region_code × brand`** | **21085** | **1.4** | 4087.09 | **1141.69** | **2945.40** | **3338.63** |

**From zero to 2 945 MAE of leakage, over the same line of code.**

> **Two separate conclusions:**
> 1. **The danger is a property of the key, not of the technique.** Target encoding on
>    `brand` is harmless. On `region_code × brand` it copies the answer into the feature.
> 2. **The extreme row also exposes something else.** Its *honest* validation score is
>    4 087 — four times worse than not encoding at all. **That feature was a bad idea at
>    that cardinality regardless of leakage. What leakage did was dress it up in a good
>    score.**

### 9.4 The same record on both sides of the split

A quarter of the listings are duplicated, then split at random.

| | Measured |
| :--- | ---: |
| `listing_id` values present in both halves | **2 406** of 7 500 validation rows |
| Validation MAE | **942.62** |
| Holdout MAE | 1043.94 |
| **Gap** | **−101.32** |

Against the baseline's −19.87, **the gap widened fivefold**. No encoding mistake, no
feature error: **the split was over rows, and the rows were not independent.**

### 9.5 Side by side

| Setup | Validation | Holdout | Gap | Model |
| :--- | ---: | ---: | ---: | :--- |
| Honest baseline | 1014.90 | 1034.77 | −19.87 | boosted trees |
| Target encoding leak (extreme key) | 1141.69 | 3338.63 | **−2196.94** | boosted trees |
| Duplicate rows | 942.62 | 1043.94 | −101.32 | boosted trees |
| Scaler leak | 4880.06 | 4957.81 | −77.75 | nearest neighbours |

The last row runs on a different model; only its **gap** is comparable with the rows above.

### 9.6 Three conclusions

**Leakage usually damages the report, not the model.** For the scaler, the duplicates and
the two mild encodings, the holdout column barely moves. **The leak did not build a better
model; it built a better report about the same model — and the report is what gets acted
on.**

**The extreme case is the exception and should be said plainly.** There the holdout went
from 1 034 to 3 338, so the model really is worse. But its honest twin scored 4 087, so
**leakage hid a bad feature behind a good score** rather than creating a good one.

**None of the three raised an error, printed a warning, or produced an implausible number.**
The only thing separating them from the baseline is one column: **the distance between the
validation score and an untouched holdout. A validation score that beats an untouched
holdout by a wide margin is describing the split, not the model.**

---
## 10. The classifier toolbox, and the number no model chose

`05_classifier_toolbox_and_thresholds.py`. Nine classifiers, two tables, one split
(75/25, stratified).

### 10.1 A score available without fitting anything

**On the attrition test set, predicting that nobody ever leaves scores 0.8356 accuracy.**
That is the bar every model below has to clear first.

| Model | AUC | Accuracy | Flagged at 0.5 |
| :--- | ---: | ---: | ---: |
| **logistic regression** | **0.8078** | 0.8511 | 35 |
| catboost | 0.7818 | 0.8378 | 29 |
| ngboost | 0.7806 | 0.8378 | 23 |
| xgboost | 0.7784 | 0.8378 | 37 |
| lightgbm | 0.7728 | 0.8356 | 40 |
| gradient boosting | 0.7718 | 0.8378 | 33 |
| decision tree | 0.7653 | 0.8289 | 49 |
| random forest | 0.7653 | 0.8333 | 17 |
| **svm rbf** (unscaled) | **0.4973** | 0.8356 | **0** |

Two numbers carry the section:

- **Only five of the nine beat the constant guess on accuracy.**
- **The SVM flagged nobody at all, and scored 0.8356** — identical to the constant guess.
  **A model that learned nothing produced the same accuracy as a model that was never
  fitted.**

> ⇒ **On a 16.6% positive class, accuracy is mostly a report of the class balance.**
> The ranking metric is what separates the models. This is the arithmetic of section 4.1
> arriving in a results table.

### 10.2 The same nine, a different table

| Model | AUC | Accuracy |
| :--- | ---: | ---: |
| catboost | **0.9952** | 0.9675 |
| xgboost | 0.9950 | 0.9613 |
| random forest | 0.9942 | 0.9688 |
| lightgbm | 0.9937 | 0.9637 |
| gradient boosting | 0.9937 | 0.9650 |
| ngboost | 0.9926 | 0.9663 |
| logistic regression | 0.9835 | 0.9263 |
| decision tree | 0.9690 | 0.9487 |
| **svm rbf** (unscaled) | **0.5129** | 0.5075 |

| | Best AUC | Spread across models, excluding the unscaled SVM |
| :--- | ---: | ---: |
| Attrition | **0.8078** | 0.0425 |
| Acoustics | **0.9952** | 0.0262 |

> **Nine identical calls, identical splitting code, two completely different ceilings.**
> **The ceiling belongs to the data.** Choosing among the nine moves the result far less
> than choosing the table does — a fact worth knowing before a week is spent on model
> selection.

### 10.3 Which models care about scale

Min-max scaled, same split, same models:

| Model | Raw AUC | Scaled AUC | Change |
| :--- | ---: | ---: | ---: |
| **svm rbf** | **0.4973** | **0.7789** | **+0.2816** |
| logistic regression | 0.8078 | 0.8097 | +0.0019 |
| gradient boosting | 0.7718 | 0.7722 | +0.0004 |
| decision tree | 0.7653 | 0.7653 | 0.0000 |
| xgboost | 0.7784 | 0.7784 | 0.0000 |
| catboost | 0.7818 | 0.7818 | 0.0000 |
| random forest | 0.7653 | 0.7649 | −0.0004 |
| ngboost | 0.7828 | 0.7806 | −0.0022 |
| lightgbm | 0.7728 | 0.7608 | −0.0120 |

`MonthlyIncome` spans **1749 to 16985**. `JobSatisfaction` spans **1 to 4**. A distance in
that raw space is a distance in monthly income with a rounding error attached.

> **The tree rows are the control group** — they never compute a distance, and they move in
> the fourth decimal place. **Scaling is not a universal step. For the models that need it,
> it is not an improvement — it is the difference between working and not.**

### 10.4 Label encoding against one-hot

| Encoding | Columns | AUC |
| :--- | ---: | ---: |
| Label | 31 | 0.7728 |
| One-hot | 48 | 0.7654 |

**A difference of −0.0074.** Label encoding puts `JobRole` on an ordering it does not have,
and a deep enough tree can carve that axis back into the right pieces — which is why the
difference is small rather than absent.

> **It stops being small the moment coefficients are read.** See 10.8.

### 10.5 The two columns that never vary

`EmployeeCount` has one distinct value; so does `StandardHours`.
**Removing both changes AUC by +0.0000.**

> A column with no variance cannot split anything, so dropping it is housekeeping rather
> than a fix. It is still worth doing, **so the next reader does not have to check.**

### 10.6 Moving the threshold

Best-ranking model, one number changed:

| Threshold | Flagged | Precision | Recall | F1 | Accuracy | **AUC** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 210 | 0.2905 | **0.8243** | 0.4296 | 0.6400 | **0.8078** |
| 0.16 | 163 | 0.3558 | 0.7838 | 0.4895 | 0.7311 | **0.8078** |
| 0.20 | 138 | 0.3986 | 0.7432 | **0.5189** | 0.7733 | **0.8078** |
| 0.30 | 87 | 0.4483 | 0.5270 | 0.4845 | 0.8156 | **0.8078** |
| 0.50 | 35 | **0.6000** | 0.2838 | 0.3853 | **0.8511** | **0.8078** |
| 0.70 | 12 | 0.5833 | 0.0946 | 0.1628 | 0.8400 | **0.8078** |

**The last column does not move.** AUC reads the ordering of the scores, and sliding a cut
through a fixed ordering cannot change an ordering.

Everything to its left moves violently: **recall from 0.82 to 0.09, precision from 0.29 to
0.60.**

> ⇒ **Those are business decisions, not model performance.** Reporting a classifier as
> "85% accurate" without stating the threshold states nothing.

### 10.7 Forcing the flagged rate to match the base rate

The training base rate is 0.1659, so flag the top 75 of 450 scores.
**The threshold that does that is 0.3466, not 0.5.**

| Cut | Flagged | Caught | Missed | False alarms | Precision | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default 0.5 | 35 | 21 | 53 | 14 | **0.6000** | 0.2838 |
| Forced rate | 75 | **35** | **39** | 40 | 0.4667 | **0.4730** |

**Same fitted model, same scores, one number changed by hand.** Fourteen more people
caught, twenty-six more people disturbed.

> **Which row is better depends on the cost of one conversation against the cost of losing
> one person — and no metric in the script knows that.**
> **0.5 is a default, not a conclusion.**

### 10.8 Reading coefficients back against the generator

The labels came from a stated log-odds model, so "did it learn the truth" is checkable.

**One correction has to be applied first.** The fit runs on standardised columns, so each
coefficient is a weight **per standard deviation**; the generator states weights **per
unit**. Multiplying each true weight by its own column's spread puts them on one footing.

| Term | Raw weight | × spread | True rank | Fitted rank | Sign |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `YearsAtCompany` | −0.0850 | −0.9899 | 1 | **1** | ok |
| `OverTime` | **+1.2500** | 0.5481 | 2 | 12 | ok |
| `MaritalStatus_Single` | +0.8500 | 0.3928 | 3 | 7 | ok |
| `MonthlyIncome` | −0.0001 | −0.3802 | 4 | 9 | ok |
| `NumCompaniesWorked` | +0.1100 | 0.3198 | 5 | 10 | ok |
| `JobSatisfaction` | −0.2400 | −0.2671 | 6 | 14 | ok |
| `StockOptionLevel` | −0.2200 | −0.2496 | 7 | 2 | ok |
| `DistanceFromHome` | +0.0300 | 0.2488 | 8 | **8** | ok |
| `Age` | −0.0190 | −0.2271 | 9 | 4 | ok |
| `JobInvolvement` | −0.2000 | −0.2195 | 10 | 5 | ok |
| `BusinessTravel_Travel_Frequently` | +0.5500 | 0.2177 | 11 | 15 | ok |
| `WorkLifeBalance` | −0.1700 | −0.1916 | 12 | 29 | ok |

**Signs recovered: 12 of 12. Fitted ranks landing within three of the true rank: 2 of 12.**

> **Two findings, both worth stating:**
> 1. **The heaviest raw weight in the generator is `OverTime` at +1.25, and it comes out
>    twelfth.** It is a yes/no column whose whole range is one step; `YearsAtCompany` runs
>    over decades. **Weight-per-unit and weight-per-standard-deviation are different
>    questions, and a fitted coefficient only ever answers the second.**
> 2. **Even on the same footing the ordering only half survives.** Twelve overlapping
>    effects and 1 800 rows are enough to recover **directions**, not a league table.

---

## 11. Ensembling, measured

`06_ensembling_blend_vs_stack.py`. Four learners, 30 000 training rows, 10 000 holdout,
4-fold out-of-fold predictions.

### 11.1 Each learner alone

| Learner | Out-of-fold MAE | Holdout MAE |
| :--- | ---: | ---: |
| xgboost | 1032.60 | 1004.33 |
| lightgbm | 1018.95 | 997.17 |
| **catboost** | **858.42** | **858.26** |
| neighbours | 4821.15 | 4892.54 |

### 11.2 How much they agree

| | Prediction correlation | **Error correlation** |
| :--- | ---: | ---: |
| Among the three boosted models | **0.9955** | **0.8319** |
| Neighbours against the three | 0.575 | **0.2454** |

> **Predictions correlate because they are mostly the price. The errors are what is left
> when the price is removed — and averaging can only cancel what is left.**

### 11.3 Four ways to combine

| Combination | Holdout MAE | Gain over the best single model | Error correlation |
| :--- | ---: | ---: | ---: |
| **Best single model (catboost)** | **858.26** | — | — |
| Simple average of the three boosted | 899.87 | **−41.61** | 0.8319 |
| Weighted by inverse holdout error | 894.98 | **−36.72** | 0.8319 |
| **Stack over the three boosted** | **856.12** | **+2.13** | 0.8319 |
| Simple average of all four | 1527.98 | **−669.72** | 0.5386 |
| Stack over all four | 856.20 | +2.06 | 0.5386 |

The stack's learned weights:

```
xgboost   0.0132        lightgbm   0.0564
catboost  0.9368        neighbours 0.0050
```

### 11.4 Four things this table says

**A simple average loses when the members are unequal.**
Averaging the three boosted models gives 899.87 — **41.61 worse than the best of them**.
The best scores 858, the worst 1 004; **an equal vote spends the good model's accuracy on
carrying the other two.**

**Weighting by holdout error is tuning on the set being predicted.**
It recovers part of the loss (894.98) with **one parameter per model read off the holdout**.

**Adding a weak but different model splits the two methods completely.**
The neighbour model's error is **5.7× the best model's**, but its error correlation with
the boosted models is only 0.2454.

- **A simple average must take it at full strength** ⇒ 899.87 collapses to **1527.98**
- **A stack may weight it freely** ⇒ it chose **0.0050**, and the result barely moves

**And the honest size of the win is 0.25%.**
The best combination beats the best single model by **2.13 MAE**. Four models, a fold loop
and a meta model, for a quarter of a percent — **and two of the six combinations came out
worse than doing nothing.**

> ⇒ **Before adding a model to a blend, the question is not how good it is. It is how
> differently it is wrong — and that is a number, not a judgement.**
> **Three models that make the same mistakes are one model that took three times as long
> to train.**

### 11.5 Blending and stacking

Both learn the weights instead of assuming them; they differ in where the meta-features
come from.

| | **Blending** | **Stacking** |
| :--- | :--- | :--- |
| Meta-features from | **One hold-out split** | **Cross-validation** |
| Base models see | Part of the training data | Effectively all of it |
| Complexity | **Simple** | Higher, and easier to leak with |

With a hold-out split of 70/30 inside a 70/30 split, the actual allocation is
**49% base models / 21% meta model / 30% final evaluation** — a detail that is easy to
lose track of, and the real cost of the simpler method.

Three cautions that apply to either:

1. **A plain linear meta model constrains neither sign nor sum.** Negative weights are
   common; `LinearRegression(positive=True)` or manual normalisation if that matters.
2. **The meta model must be fitted on predictions made for rows the base models did not
   see**, or it will reward whichever base model overfits hardest.
3. **The base models' outputs have to be on comparable scales**, or a linear combination of
   them means nothing.

### 11.6 A one-line weighting rule worth keeping

When training a meta model is not worth it, weight by **inverse error, cross-multiplied**:

```
model1 mae = 510        w1 = 490 / (510 + 490) = 0.49
model2 mae = 490        w2 = 510 / (510 + 490) = 0.51
```

**Each model's weight is the other's error.** The better model gets more weight, and the
whole thing is one line with no extra data.

### 11.7 Where ensembling sits

> **Feature engineering first. Ensembling is a late-stage move. Hyperparameter search is
> last.**

Three figures from these scripts, all scored on the same untouched holdout:

| Step | Holdout MAE | Configuration |
| :--- | ---: | :--- |
| A plain feature set, boosted trees | **1034.77** | 25 raw columns, LightGBM, 400 rounds |
| Same data, a stronger model | **858.26** | 26 columns, CatBoost, 400 rounds, depth 7 |
| **Engineered features** | **801.35** | **68 features, CatBoost, 1200 rounds, depth 8** |
| Best ensemble over four models | **856.12** | stack over out-of-fold predictions |

> These are not a controlled ablation — the model and the round budget change along with
> the feature set, so the rows cannot be differenced cleanly. What they do show is the
> **order of magnitude of each move**: engineering the features and strengthening the model
> are worth tens to hundreds of MAE, while the ensemble that came last is worth **2.13**.

---
## 12. A network with nothing but arrays

`07_neural_net_from_scratch.py`. No framework appears anywhere in this script — the point
is that every quantity in it can be checked against something.

### 12.1 A neuron is a decision, and the activation is the switch

A neuron receives weighted signals and decides whether what it received is important
enough to pass on. The simplest possible version of that decision is a step: above a
threshold, emit 1; below it, emit 0.

```
z = Σ wᵢxᵢ + b        a = f(z)
```

> **A simple linear model plus a non-linear function.**

That one line explains why a logistic regression can be described as the last layer of a
network: **replace `f` with a sigmoid and the neuron literally is one.** Everything the
rest of this section compares — sigmoid, tanh, ReLU — is the feel of that switch: hard,
soft, or open on one side and shut on the other.

**Structure conventions worth fixing before the code:**

- **The input layer is layer 0 and does not count toward depth.** A `2-3-2-2` network has
  two hidden layers.
- **Depth and width are set by a person; the weights are learned.** That is the whole line
  between hyperparameter and parameter.
- `w_ij^(l)` puts the **destination index first**, which looks backwards until you notice
  it makes `z = W·a + b` line up with matrix multiplication — and makes back-propagation
  come out as the transpose of the same matrix.
- **The bias is the weight on a constant input of 1**, which is why it updates under the
  same gradient rule as everything else. Section 12.7 measures what happens when it does
  not.

### 12.2 Why the non-linearity is not optional

Without it, every layer is a linear function of the previous one, so
`W₂(W₁x) = (W₂W₁)x` — **the whole stack collapses to a single layer.** A deep network with
no activation is a linear regression with extra notation.

With it, the network can approximate essentially any function. It is the same problem a
kernel solves in section 4.4: low-dimensional data that no straight boundary separates.
**The kernel maps once; a network maps layer after layer.**

### 12.3 Three activations, measured

| Activation | Largest slope anywhere | Where |
| :--- | ---: | ---: |
| sigmoid | **0.2500** | x = 0.00 |
| tanh | **1.0000** | x = 0.00 |
| relu | **1.0000** | everywhere x > 0 |

| x | sigmoid | slope | tanh | slope | relu | slope |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6.0 | 0.0025 | 0.0025 | −1.0000 | 0.0000 | 0.0000 | 0.0000 |
| −0.5 | 0.3775 | 0.2350 | −0.4621 | 0.7864 | 0.0000 | 0.0000 |
| **0.0** | 0.5000 | **0.2500** | 0.0000 | **1.0000** | 0.0000 | 0.0000 |
| 0.5 | 0.6225 | 0.2350 | 0.4621 | 0.7864 | 0.5000 | **1.0000** |
| 6.0 | 0.9975 | 0.0025 | 1.0000 | 0.0000 | 6.0000 | **1.0000** |

**Sigmoid.** `1/(1+e^(-x))`, output in (0, 1). Its derivative has a form worth knowing:

```
f'(x) = f(x) · (1 − f(x))
```

**The derivative is computable from the output alone** — no exponential is recomputed
during the backward pass. That was a real engineering reason for its early popularity.

Its two weaknesses: **the derivative peaks at 0.25**, and **the output is never negative**,
so for a given neuron every weight's gradient carries the same sign and the whole row can
only move up together or down together. That zig-zag is what tanh's zero-centring fixes.

**tanh.** Odd, range (−1, 1), **derivative 1 at the origin**. It cures the non-zero-mean
problem and **not** the saturation: both tails still flatten.

**ReLU.** `max(0, x)`. **The gradient is exactly 1 wherever the input was positive**, so
error propagates backward undiminished. The price is that a unit pushed permanently
negative has gradient 0 forever and never returns — which is what Leaky ReLU exists to
address.

> **The three in one line:** sigmoid collapses in depth because its slope is capped at a
> quarter; tanh raises the centre to 1 and still dies in the tails; ReLU makes the positive
> half exactly 1 **and gives up the negative half entirely.**

### 12.4 What multiplying those slopes does

Back-propagation carries the error through one derivative per layer, multiplying as it
goes. The table below uses **each activation's best possible slope**, so it is a ceiling —
a real network only does worse.

| Layers | sigmoid | tanh | relu |
| ---: | ---: | ---: | ---: |
| 1 | 2.500e-01 | 1.000e+00 | 1.000e+00 |
| 2 | 6.250e-02 | 1.000e+00 | 1.000e+00 |
| 5 | 9.766e-04 | 1.000e+00 | 1.000e+00 |
| **10** | **9.537e-07** | 1.000e+00 | 1.000e+00 |
| 20 | 9.095e-13 | 1.000e+00 | 1.000e+00 |
| 40 | 8.272e-25 | 1.000e+00 | 1.000e+00 |

**Even in the best case, ten sigmoid layers deliver a millionth of the error to the first
one.** That is not slow training — **the early layers are not training.** The network
behaves like the last few layers alone.

The two right-hand columns are flat for one reason: **ReLU's positive-side slope is exactly
1, so nothing is attenuated at all.**

> ⇒ **The diagnostic follows from the mechanism.** Total loss going down is not evidence
> that a deep network is learning; the later layers can produce that on their own.
> **Look at the gradient norm of the layers nearest the input.**

### 12.5 The forward pass, by hand

Six weight matrices of evenly spaced small decimals, chosen so the whole thing can be
verified on a calculator:

```
input                       [1.  0.5]
layer 1 weighted sum        [0.3 0.7 1.1]
    first unit by hand:     1.0 x 0.1 + 0.5 x 0.2 + 0.1 = 0.3000
layer 1 after sigmoid       [0.574443 0.668188 0.75026 ]
layer 2 weighted sum        [0.51616  1.214027]
layer 2 after sigmoid       [0.626249 0.771011]
output, no activation       [0.316827 0.696279]
```

**Nothing above is approximate.** A network this size is two matrix products with a squash
between them. The shapes of the weight matrices *are* the architecture: 2×3, 3×2, 2×2 reads
as **2 → 3 → 2 → 2**.

**The output layer has no activation** because this is a regression; classification would
put a sigmoid or a softmax there, paired with cross-entropy instead of squared error.

### 12.6 Back-propagation, and the one check that proves it

**The chain rule, made concrete.** A car costs 50, two are bought, tax is 10%, the total is
110. Differentiating right to left gives `1 × 1.1 × 2 = 2.2`. Verify it directly: raise the
unit price to 51 and the total becomes `51 × 2 × 1.1 = 112.2`, exactly +2.2.

> **A gradient is not an abstract slope. It is "if this number moves by one, how much does
> that number move".**

**Distributing the error backward.** A hidden unit feeding two outputs is responsible for
part of each output's error, in proportion to its weight on that path. Written as a matrix
and with the normalising denominators dropped:

```
E_h = Wᵀ · E_o
```

> **Forward propagation carries the input through the weight matrix; back-propagation
> carries the error through its transpose.**
> Dropping the denominators changes the absolute size of the error but not the relative
> proportions, and the absolute size is absorbed by the learning rate. That is why the real
> formula is a clean `Wᵀ` with nothing underneath it.

**The implementation, with both halves using the same aggregation:**

```python
z1 = x @ w1 + b1
a1 = np.maximum(z1, 0.0)                  # ReLU
prediction = a1 @ w2 + b2

residual = prediction - y
loss = np.mean(residual ** 2)

d_pred = 2.0 * residual / n               # gradient of the same mean
grad_w2 = a1.T @ d_pred
grad_b2 = d_pred.sum(axis=0)
d_a1 = d_pred @ w2.T
d_z1 = d_a1 * (z1 > 0)                    # ReLU's derivative
grad_w1 = x.T @ d_z1
grad_b1 = d_z1.sum(axis=0)
```

Two details that are easy to get wrong:

- **`d_z1` tests `z1`, the pre-activation sum — not the activated output.**
- **Taking the loss as a mean and the gradient as a sum leaves them off by the batch size.**
  Nothing raises; the learning rate is silently multiplied by however many rows are in the
  batch. Both halves here use the mean.

**The check.** A derived gradient and a finite difference of the loss must produce the same
number. Ten parameters sampled:

| Parameter | Derived | Finite difference | Relative error |
| :--- | ---: | ---: | ---: |
| `w1[241]` | 0.53008212 | 0.53008205 | 1.42e-07 |
| `w1[250]` | 4.55895805 | 4.55895815 | 2.28e-08 |
| `b1[5]` | 0.96778989 | 0.96778990 | 1.43e-08 |
| `b1[12]` | −21.79938403 | −21.79938394 | 4.09e-09 |
| `w2[1]` | −41.73640272 | −41.73640286 | 3.53e-09 |
| `w2[10]` | −50.39624449 | −50.39624455 | 1.12e-09 |
| `b2[0]` | −62.41542020 | −62.41542019 | 2.03e-10 |

**Worst relative error: 1.42e-07.**

> **This check has no substitute.** A wrong gradient still trains — it just trains toward
> somewhere else. The loss goes down, the curve looks fine, nothing raises.
> **Perturb a weight, watch the loss, divide by twice the perturbation.** No framework
> required.

### 12.7 Declaring a bias and never updating it

A common shape of bug: the biases exist, participate in the forward pass, and never receive
a gradient. Same data, same initial weights, same 4 000 epochs:

| Run | First loss | Final loss | Test MAE |
| :--- | ---: | ---: | ---: |
| **Biases updated** | 1049.4831 | **4.3988** | **2.4799** |
| **Biases left at zero** | 1049.4831 | **15.5387** | **4.7898** |

| | Measured |
| :--- | ---: |
| Final output-layer bias | **5.8580 against 0.0000** |
| Parameters frozen | **25 of 337 (7.4%)** |

> **Not a crash and not a warning.** Only 7.4% of the parameters were frozen — but they are
> the ones carrying the offset. With the inputs standardised to a mean of zero, **the
> weights alone cannot produce an average output of 31.5.**
> **Both loss curves descend.** Only side by side is the factor of three visible.

### 12.8 Optimisers, briefly

The scripts here use plain gradient descent, because the point is the mechanism. In
practice the progression is worth knowing:

| Optimiser | Problem it solves | New state it keeps |
| :--- | :--- | :--- |
| SGD | Full-batch descent is slow | none |
| Momentum | SGD's direction oscillates | **velocity** |
| Adagrad | One learning rate for all parameters | **accumulated squared gradients** |
| RMSprop | Adagrad's rate decays too fast | a **moving average** of them |
| **Adam** | Both at once | **first and second moments** |

⇒ **Adam is momentum for the direction plus RMSprop for the step size**, which is why it
became the default that nobody tunes.

> A caution that generalises past optimisers: **a typical value is not a universal
> default.** A learning rate interacts with the data size *and* with how the loss is
> aggregated — a loss summed rather than averaged inflates every gradient by the batch
> size, and the learning rate must shrink to match. The two settings are coupled.

### 12.9 What the network recovered

The target came from eleven linear terms plus **one saturating `tanh` term on
`vacancy_rate`**, with noise σ = 2.2.

**How much a term actually moves the target is the spread of that term over the data**, not
its coefficient — for linear terms that is `|coefficient| × column spread`; the saturating
term has to be evaluated, because `tanh` caps its own swing.

| Feature | Term spread | True rank | Network rank |
| :--- | ---: | ---: | ---: |
| `vacancy_rate` (saturating) | 6.405 | 1 | **1** |
| `rooms` | 3.429 | 2 | 3 |
| `floor_area` | 2.170 | 3 | 6 |
| `distance_to_centre` | 2.013 | 4 | 12 |
| `pupil_teacher_ratio` | 1.724 | 5 | **5** |
| `crime_index` | 1.552 | 6 | 7 |
| `school_rating` | 1.141 | 7 | 8 |
| `tax_rate` | 1.078 | 8 | 11 |
| `lot_size` | 0.856 | 9 | 4 |
| `transit_index` | 0.847 | 10 | **10** |
| `noise_level` | 0.681 | 11 | 9 |
| `build_year` | 0.520 | 12 | 2 |

**Two of the three strongest terms land in the network's own top three. Test MAE 2.4799
against a noise floor of 2.20.**

> One hidden ReLU layer and a `for` loop. Nobody told it there were twelve features or
> that one of them saturates — **and it put the saturating one first.**

### 12.10 A note on the data

The dataset traditionally used for this problem was withdrawn from scikit-learn on
ethical grounds:
one of its columns is **the proportion of Black residents in a district**, used directly as
a predictor of house price. These scripts therefore use a **synthesised table of the same
shape** — 506 rows, twelve numeric features, all of them property and geography
attributes — which also has the advantage that its generating coefficients are known, and
section 12.9 could not exist otherwise.

Two facts from the original are worth carrying forward regardless:

- **Sixteen of its rows sit exactly at the price cap**, because values above the ceiling
  were recorded as the ceiling. Those rows' true values are unknowable and a model must
  underestimate them. Systematic under-prediction at the top of a range is worth checking
  for truncation before it is blamed on the model.
- **Its feature scales span five orders of magnitude**, so without standardisation the
  gradient is dominated by one column. The synthesised table preserves that property
  deliberately — `noise_level` around 0.55 against `lot_size` around 9 412.

**And a baseline that is usually not printed**: the MSE of always predicting the mean is
the variance of the target. Together with the noise floor it brackets the result — **what
you get for doing nothing, and the best anyone can do.** The model's score is only
meaningful between those two lines.

---

## 13. The framework ladder

`08_framework_abstraction_ladder.py`. One network — **12 → 10 → 1, ReLU, MSE** — written
four times.

### 13.1 What each rung takes over

| Rung | Implementation | You write | The framework writes |
| :--- | :--- | :--- | :--- |
| **0** | numpy | **structure, forward, loss, gradients, update** | matrix multiplication |
| **1** | PyTorch | structure, training loop | **gradients, optimiser** |
| **2** | TensorFlow + tape | structure, training loop | **gradients** |
| **3** | Keras `fit` | **structure only** | **loop, gradients, optimiser, logging** |

| Step | numpy | PyTorch |
| :--- | :--- | :--- |
| Structure | four arrays | `nn.Sequential(nn.Linear(12,10), nn.ReLU(), nn.Linear(10,1))` |
| Forward | three lines | `model(x)` |
| **Gradients** | **five lines of chain rule** | **`loss.backward()`** |
| Update | four lines | `optimizer.step()` |
| Clear gradients | not needed | **`zero_grad()` — required, they accumulate** |

> `loss.backward()` walks the same `E_h = Wᵀ·E_o` chain from section 12.6; it just keeps
> the graph itself. **`zero_grad()` is the line most often forgotten when moving off
> hand-written gradients**, because nothing in the numpy version corresponds to it.

Rung 3 removes the training loop entirely: `model.fit(...)` absorbs forward pass, loss,
gradient clearing, backward pass, update and logging, and `history.history["loss"]` is the
list that the loop used to build.

### 13.2 Do the four agree?

**Same initial weights** (drawn once with numpy, then loaded into both frameworks), **same
300 full-batch steps**, **learning rate 0.05, no momentum, no shuffling**, everything in
float32.

| Implementation | First loss | Final loss |
| :--- | ---: | ---: |
| numpy, hand-derived gradients | 1002.466614 | **5.064641** |
| PyTorch, autograd | 1002.466553 | **5.064640** |
| TensorFlow, gradient tape | 1002.466614 | **5.064638** |

| Pair | Largest gap over the whole run | Final gap |
| :--- | ---: | ---: |
| numpy vs PyTorch | 1.556e-03 | 4.768e-07 |
| numpy vs TensorFlow | 1.392e-03 | 2.861e-06 |
| PyTorch vs TensorFlow | 1.556e-03 | 2.384e-06 |

| Parameter | numpy vs PyTorch | numpy vs TensorFlow |
| :--- | ---: | ---: |
| `w1` | 1.907e-05 | 2.003e-05 |
| `b1` | 1.669e-05 | 2.158e-05 |
| `w2` | 2.217e-05 | 2.408e-05 |
| `b2` | 1.907e-06 | 7.629e-06 |

And the top rung: 300 epochs of `model.fit` land **4.768e-07** from the hand-written run,
test MAE **2.0224**.

> **Four implementations agreeing to the last few digits a float32 can hold.**
> **Automatic differentiation removed the writing of the derivatives, not the derivatives.**
> Two thirds less code, the same arithmetic.

> ⚠️ **One detail can quietly destroy this comparison**: numpy defaults to float64 and both
> frameworks default to float32. Without pinning the dtype the curves separate, **and the
> separation has nothing to do with the abstraction level.**

### 13.3 Graph mode

A computation graph is nodes (operations) and edges (tensors).
`c = a + b` **does not add anything** — it adds an addition node to a graph. The work
happens when the graph runs, which is precisely what gives the framework a chance to fuse,
reorder and place operations first.

Eager execution is the default and behaves like ordinary Python, which is what makes it
debuggable. `@tf.function` traces a Python function into a graph.

**Measured, 200 steps each, with the tracing call excluded from the timing:**

| Mode | 200 steps |
| :--- | ---: |
| Eager | **0.299 s** |
| Compiled graph | **0.038 s** |
| **Ratio** | **7.9×** |

> **Read that ratio correctly.** What is saved is the **Python overhead between
> operations**. On a two-layer network with ten hidden units, that overhead is almost the
> entire runtime. On a model large enough that the arithmetic dominates, the same switch
> saves proportionally far less.
> **The speed-up is a function of model size, not a property of the framework.**

And a second, less obvious consequence: **a graph can be serialised and a Python function
cannot.** `@tf.function` is not only a performance switch — it is the precondition for
exporting the model at all.

### 13.4 Data parallelism

| | Data parallel | Model parallel |
| :--- | :--- | :--- |
| Split | **the data** | **the model's layers** |
| Each device holds | **a full copy of the weights** | one piece |
| Used when | the model fits on one device | **it does not** |
| Constraint | none | **layers with dependencies still run in sequence** |

> **Model parallelism has a ceiling built into it**: layer 2 waits for layer 1, so two
> devices are serial and only the memory is shared. **It solves "does not fit", not "is
> slow"** — which is why data parallelism is the mainstream option.

The conversion cost is three lines: create the strategy, put the model construction and
`compile` inside `strategy.scope()`, leave training, prediction and plotting untouched.

**Measured on this machine:**

| Item | Measured |
| :--- | :--- |
| Visible devices | **['CPU']** |
| **Replicas in sync** | **1** |
| Final loss inside the scope | 5.064640 |
| Final loss outside it | 5.064640 |
| **Difference** | **0.000e+00** |

> **This run had one replica, so nothing was split and nothing ran in parallel. Reporting
> it as a speed result would be reporting a measurement that was never taken.**
>
> What it does demonstrate is the programming model. **Mirrored means every device holds a
> full copy of the weights and receives a slice of the batch, and the gradients are summed
> across devices before the update** — which is exactly why the result does not depend on
> how many devices there are, and why that difference is zero rather than merely small.

### 13.5 Serving

Export produces a **SavedModel** — the graph and the weights, not the Python — which a
serving process exposes over REST or gRPC.

Two details worth keeping:

- **The version is a directory name.** Dropping a new numbered subdirectory in place is the
  update mechanism; the process does not restart.
- **Training and serving are different problems.** Training cares about throughput and
  convergence; serving cares about per-request latency, concurrency and version switching.
  The "train once, predict many times, so split the scripts" habit from section 8 reaches
  its final form here: **inference stops being a script and becomes a service.**

---
## 14. The scripts

Eight scripts. **`01` builds the data; the rest are independent of each other.**

| # | Script | Sections | What it establishes |
| :--- | :--- | :--- | :--- |
| **01** | `build_tabular_datasets.py` | 6 | Four tables from stated formulas — seeded, reproducible byte for byte, and the reason every later number has something to be checked against |
| **02** | `eda_that_silently_lies.py` | 7 | One file, two separators: identical shape, identical column names, identical missing-value total — and **1 888 rows lose their label** |
| **03** | `feature_engineering_and_boosting.py` | 8 | 68 features into CatBoost, then the audit: **21 of them are never used**, 18 carry 95% of the decision |
| **04** | `leakage_and_split_discipline.py` | 9 | Three leaks, honest against leaky, all scored on an untouched holdout; target encoding's damage runs from **0.00 to 2945.40** with the key |
| **05** | `classifier_toolbox_and_thresholds.py` | 10 | Nine classifiers over two tables; scaling is worth **+0.2816 AUC** to one of them and nothing to six; AUC does not move when the threshold does |
| **06** | `ensembling_blend_vs_stack.py` | 11 | Six ways to combine four models — **two of them lose**; the stack recovers by giving one model a weight of **0.9368** |
| **07** | `neural_net_from_scratch.py` | 12 | Activation slopes, gradient decay by depth, a hand-checkable forward pass, a gradient check at **1.42e-07**, and a bias ablation |
| **08** | `framework_abstraction_ladder.py` | 13 | The same network in numpy, PyTorch, TensorFlow and Keras, agreeing to **1.5e-03**; graph mode **7.9×** eager; replica count reported as the **1** it was |

### 14.1 Running them

```bash
python 01_build_tabular_datasets.py     # writes data/, required by everything else
python 02_eda_that_silently_lies.py
python 03_feature_engineering_and_boosting.py
python 04_leakage_and_split_discipline.py
python 05_classifier_toolbox_and_thresholds.py
python 06_ensembling_blend_vs_stack.py
python 07_neural_net_from_scratch.py
python 08_framework_abstraction_ladder.py
```

Every script writes its findings to stdout. `07` also writes two plots to `outputs/`.
**A cold run of all eight takes about three and a half minutes on CPU**, the bulk of it in
`03`.

`data/` and `outputs/` are regenerated on demand and are not tracked.

### 14.2 Dependencies

```
numpy  pandas  scikit-learn  matplotlib
xgboost  lightgbm  catboost  ngboost
torch  tensorflow-cpu
```

**One installation note.** NGBoost declares no pandas requirement of its own, but depends
on lifelines, which caps pandas below 3.0 — so installing it will downgrade pandas unless
pandas is pinned. **The cap is precautionary rather than a real incompatibility**: lifelines
imports and NGBoost fits and predicts normally on pandas 3.0.5, which is the version every
script here was verified against. Holding pandas at 3.x leaves one entry in `pip check`
about that declared bound and nothing else.

The reason to hold it there rather than accept the downgrade is that **pandas 3.0 removed
conversions that 2.x only warned about**, so a script that runs on 2.x is not yet known to
run on 3.x. Calling `float()` on a one-element Series is the case this module actually
tripped over.

---

## 15. What the module is really about

None of the code here is difficult to write. Every one of these models is a handful of
lines, and the frameworks make the neural network shorter still.

**The difficulty is in reading the number that comes out.**

Which is why the same shape of question keeps returning in every section:

- **What is the floor?** `801.35` means nothing until `718` is next to it.
- **How many features did the model actually use?** Sixty-eight went in; **twenty-one were
  never touched, and eighteen made the decision.**
- **How far is the validation score from an untouched holdout?** −19.87 is a working split.
  −2196.94 is a description of the split, not of the model.
- **Who chose the 0.5?** Not the model. Move it and recall goes from 0.82 to 0.09 while the
  ranking metric does not move at all.
- **How differently are these models wrong?** Error correlation 0.8319, and the average of
  three of them is **41.61 worse** than the best one alone.
- **Is this gradient the gradient?** Perturb the weight, watch the loss, divide.
  **1.42e-07.**
- **How many replicas were actually in sync?** **One.** So there is no speed result to
  report.

Every one of those is a number rather than a judgement, and every one of them can be
computed before anyone is asked to trust the result.
