# Fine-tuning with low-rank adaptation

Scripts 01 and 02 show the low-rank idea on plain matrices: truncated SVD on an image, and
alternating least squares on a sparse preference matrix. Script 03 measures LoRA's premise on a
real update. Scripts 04, 05 and 07 train LoRA adapters three ways: supervised fine-tuning, GRPO
with rule-based rewards, and a vision-language model reading rendered panels. Script 06 changes
no weights and controls how long a reasoning model thinks. This document explains what each
script does and the ideas it relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_svd_image_compression.py` | Truncated SVD, paired sign flips, rank-k rebuilds, and why energy share understates the error |
| 02 | `02_als_low_rank_factorization.py` | Alternating least squares on a masked sparse matrix, the objective against the RMSE, and how much data a rank needs |
| 03 | `03_lora_low_rank_hypothesis.py` | A LoRA layer by hand, where adapters attach, and the spectrum of a full-rank update against two controls |
| 04 | `04_lora_sft_instruction_tuning.py` | Supervised fine-tuning with LoRA on rule-generated labels, against a prompted baseline, then save, reload and merge |
| 05 | `05_grpo_reward_shaping.py` | GRPO with five rule-based rewards, a k3 KL penalty with the adapter switched off, and the cold start |
| 06 | `06_thinking_budget_control.py` | Thinking budget control: capping and extending the thinking phase by decoding one token at a time |
| 07 | `07_vision_lora_gauge_reading.py` | Vision LoRA on the language tower of a small vision-language model, scored field by field |

## Shared setup (scripts 03 to 07)

*   Every number below comes from a run on one laptop GPU, an RTX 5070 Ti Laptop with 12 GB
    (Blackwell, sm_120). Scripts 01 and 02 run on the CPU in about 10 and 30 seconds.
*   Scripts 03 to 06 load DeepSeek-R1-Distill-Qwen-1.5B from the weights folder of module 01,
    or from `HF_CACHE_DIR` when it is set, so they download nothing new. Script 07 downloads
    SmolVLM-256M-Instruct, 518 MB.
*   Adapters come from `peft` (`LoraConfig` and `get_peft_model`), models from `transformers`,
    and the training loops are written out with `torch`. Blackwell needs CUDA 12.8 or newer,
    and the accelerated fine-tuning stacks and the 4-bit quantisation library were unverified or
    unsupported with it on Windows. Written out, the loops also show the prompt mask, the
    advantages and the KL term that a wrapper would hide.
*   Models load in bfloat16, except in script 03 (float32, see there). Every script seeds its
    generators, so the numbers reproduce on the same hardware.
*   Beyond the module's requirements, 03 to 06 need `peft`, and 07 also needs `torchvision`
    for the image processor.

    | Script | Peak VRAM reserved | Time |
    | :--- | ---: | :--- |
    | `03` | 9.75 GB | 78 s for 4 runs of 40 steps |
    | `04` | 6.13 GB | 15.8 s for 120 steps, about 2 min 50 s in all |
    | `05` | 7.07 GB | 400.3 s for 24 steps |
    | `06` | 3.78 GB | about 3 min, no training |
    | `07` | 4.43 GB | 210.3 s for 150 steps |

## Script 01: Truncated SVD

SVD writes any matrix as a weighted sum of rank-1 terms, `A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ...`, with the
singular values as weights. Keeping the k largest terms gives the best rank-k matrix, and its
error depends only on the weights left out. LoRA rests on the same idea: a weight update whose
singular values fall fast can be stored as a few terms.

*   Part 1 decomposes a 3 × 2 matrix. The singular values are 2.618 and 0.382, and their
    squares, 6.854 and 0.146, equal the eigenvalues of AᵀA. The two rank-1 terms add back up to
    A with a largest deviation of 8.88e-16.
*   Part 2 shows that signs flip in pairs. Negating column 1 of U alone gives a different
    matrix, 3.79 away from A at the worst entry. Negating row 1 of Vᵀ as well restores A. Every
    number in a decomposition can be right and the product still wrong.
*   Parts 3 and 4 draw a 512 × 512 grayscale test image (a diagonal gradient, a ring, a
    rectangle, stripes, a triangle, the word RANK and noise) and rebuild it from the top k
    terms. A rank-k factorisation stores `k × (rows + columns + 1)` numbers:

    | k | Relative error | Stored | Share of 262,144 |
    | ---: | ---: | ---: | ---: |
    | 1 | 35.47% | 1,025 | 0.39% |
    | 5 | 12.24% | 5,125 | 1.96% |
    | 20 | 6.08% | 20,500 | 7.82% |
    | 50 | 3.78% | 51,250 | 19.55% |
    | 100 | 2.51% | 102,500 | 39.10% |

*   What comes back first depends on the direction of an edge, not on how fine the detail is.
    SVD splits a matrix into rows and columns, so the stripes and the rectangle are sharp at
    k=2. The ring needs about 10 terms and the word RANK about 20, and the diagonal triangle is
    still blurred at k=50.
*   Part 5 compares energy share with the error. The first term holds 87.42% of the energy, yet
    the k=1 rebuild is 35.47% off and shows nothing. Energy is the square of the magnitude, so
    `relative error = sqrt(1 - energy share)`:

    | k | Energy kept | Relative error |
    | ---: | ---: | ---: |
    | 1 | 87.42% | 35.47% |
    | 2 | 92.64% | 27.12% |
    | 3 | 95.59% | 21.01% |
    | 8 | 99.06% | 9.70% |

    A rank chosen at 90% energy still leaves about a third of the magnitude wrong, so k is
    chosen by the error the task can accept.
*   Part 6 reads two reference matrices the same way:

    | Matrix | Rank | σ₁ / σ₂₀₀ | Energy at k=8 | Error at k=8 |
    | :--- | ---: | ---: | ---: | ---: |
    | Gaussian noise (floor) | 512 | 2.0 | 5.83% | 97.04% |
    | Test image | 512 | 596.6 | 99.06% | 9.70% |
    | Gradient background (ceiling) | 2 | none past σ₂ | 100.00% | 0.00% |

    The gradient background alone holds 58% of the image's energy, so the image sits close to
    the ceiling.
*   Part 7 builds a 512 × 512 stand-in update as a rank-12 product plus noise. Its singular
    values drop from 118.09 at term 12 to 1.73 at term 13, and rank 12 stores 12,300 of 262,144
    numbers (4.69%). The matrix was built with that shape, so it shows what LoRA assumes and
    tests nothing. Script 03 measures a real update.

## Script 02: Alternating least squares

ALS models each cell of a user-by-item matrix as the dot product of a user vector and an item
vector, here of length 3. It minimises, over the observed cells only,
`Σ (r_ui - x_uᵀy_i)² + λ(Σ‖x_u‖² + Σ‖y_i‖²)`. With the item vectors held fixed, each user vector
is a ridge regression with a closed form, solved with one `np.linalg.solve` per row; then the
roles swap. A mask marks the observed cells, so a missing cell never enters the loss and is
never read as a zero.

*   Part 1 shows the data: 12 users and 9 items in three groups. Each user touched two of the
    three items in their group, 24 of 108 cells, so a good fit recommends the third.
*   Part 2 compares two spectra. With every group filled in, the complete matrix has rank 3.
    The observed matrix is full rank,
    because the unobserved cells read as zeros, but its singular values drop after the third:
    2.000 then 1.276.
*   In part 3, over 20 iterations the objective falls every time, 4.663 to 1.066, while the
    plain RMSE on the observed cells rises from 0.0085 to 0.0169: the solver trades a little
    training error for much smaller factors, which is what the penalty asks for.
*   The loss cannot say when to stop, so parts 4 and 5 score group agreement: whether each
    user's top unseen item lies in their own group, after 20 iterations and after 2.

    | Fit | Objective | RMSE | Top item in the right group |
    | :--- | ---: | ---: | ---: |
    | After 2 iterations (lowest RMSE) | 3.106 | 0.0073 | 7/12 |
    | After 20 iterations | 1.066 | 0.0169 | 12/12 |

*   Over ten seeds, stopping at the lowest-RMSE iteration cost accuracy in 5. The converged fit
    reached 12/12 in 8; seeds 404 and 8080 end at 75%.
*   Part 6 generates a 300 × 120 matrix from a known rank-3 signal and hides most cells. The
    error on the hidden cells is divided by the error of answering zero everywhere, so above
    100% is worse than no model:

    | Density | Fewest cells in a row | Mean cells per row | Train RMSE | Hidden-cell RMSE | Share |
    | ---: | ---: | ---: | ---: | ---: | ---: |
    | 0.03 | 0 | 3.6 | 0.2053 | 2.0076 | 123% |
    | 0.05 | 1 | 6.1 | 0.3987 | 2.0704 | 126% |
    | 0.08 | 3 | 9.6 | 0.2951 | 1.0219 | 62% |
    | 0.15 | 7 | 18.0 | 0.0244 | 0.0481 | 3% |
    | 0.30 | 22 | 36.1 | 0.0099 | 0.0148 | 1% |

*   At 0.05 even the rows with 11 or more observed cells fail, so the whole matrix lacks data,
    not a few rows. A stronger penalty softens the failure (64% at penalty 3) but does not
    replace the data.
*   A rank is a claim about how much data each row needs. The same holds for an adapter's rank:
    capacity the data cannot support memorises.

## Script 03: LoRA's low-rank premise

LoRA freezes a weight W and trains two thin matrices beside it, so the layer computes
`h = Wx + BAx · α/r`. A (the down projection, r × in) starts random and B (the up projection,
out × r) starts at zero. BA is the update, and its rank can never exceed r. This only works if
the update a full fine-tuning would make is nearly low rank, and the script measures that.

*   Parts 1 and 2 write the layer by hand. With B at zero it answers exactly like the frozen
    layer, so training starts from the original model; once B has values the update has rank 8,
    the chosen r. The `α/r` scale keeps the added term about the same size when r changes. An
    adapter holds `r × (in + out)` parameters: 49,152 at r=16 on a 1536 × 1536 matrix, 2.08%.
*   Part 3 surveys where adapters can attach. The model has 28 layers of seven projections.
    `k_proj` and `v_proj` are 256 × 1536 instead of 1536 × 1536: grouped-query attention lets
    several query heads share one key and value head. The trainable share of three selections:

    | Selection | Modules | r=1 | r=4 | r=8 | r=16 |
    | :--- | ---: | ---: | ---: | ---: | ---: |
    | q_proj and v_proj | 56 | 0.008% | 0.031% | 0.061% | 0.123% |
    | All four attention projections | 112 | 0.015% | 0.061% | 0.123% | 0.245% |
    | All seven projections | 196 | 0.065% | 0.260% | 0.520% | 1.039% |

    The often quoted "about 1% of the parameters" is the bottom right corner.
*   Part 4 trains q_proj and v_proj of the last four layers with no rank limit: 8 matrices,
    11,010,048 parameters (0.620% of the model), AdamW for 40 steps on eight fixed pairs. The
    loss goes from 3.4988 to 0.0466. Turning weight decay off changed none of the direction
    counts below.
*   The model loads in float32 here: the update is far smaller than the weights, and bfloat16
    would round part of it away.
*   The layer count is set by memory. The float32 weights take 7.11 GB, and each trained layer
    adds about 0.15 GB at peak: 4 layers reach 9.12 GB allocated in about 12 s for 40 steps,
    8 reach 9.71 GB and 16 reach 10.92 GB. All 28 would need about 12.7 GB, more than the
    12 GB card holds.
*   Parts 5 and 6 read the singular values of layer 24's q_proj update against two controls of
    the same shape. The controls make this a measurement: fast decay is not a property of every
    matrix.

    | 1536 × 1536 matrix | Directions for 50% of energy | 90% | 99% | σ₁ / σ₆₄ |
    | :--- | ---: | ---: | ---: | ---: |
    | Trained update ΔW | 3 | 36 | 493 | 22.3 |
    | Frozen weight W | 156 | 612 | 1063 | 1.9 |
    | Random noise, same scale | 279 | 783 | 1184 | 1.1 |

*   Part 7 truncates that update to rank r:

    | r | Energy kept | Relative error left | Adapter parameters |
    | ---: | ---: | ---: | ---: |
    | 1 | 30.82% | 83.18% | 3,072 |
    | 4 | 66.61% | 57.79% | 12,288 |
    | 8 | 80.09% | 44.62% | 24,576 |
    | 16 | 85.13% | 38.57% | 49,152 |
    | 64 | 92.66% | 27.10% | 196,608 |

    r=16 keeps 85% of the energy, and 39% of the magnitude is still wrong, for the reason shown
    in script 01.
*   Part 8 widens the task. Ten short input-output tasks are defined, and the tiers are nested
    prefixes of them. Pool size, batch, steps and learning rate stay fixed, and each tier starts
    again from the original weights. Averaged over the eight matrices:

    | Kinds of task | Final loss | 50% of energy | 90% | 99% |
    | ---: | ---: | ---: | ---: | ---: |
    | 1 | 0.0359 | 2.2 | 20.0 | 255.4 |
    | 3 | 0.0490 | 2.6 | 25.8 | 321.6 |
    | 10 | 0.2538 | 4.1 | 37.6 | 359.4 |

*   Seven of the eight matrices need more directions at every tier. At ten kinds, 90% of the
    energy still fits in about 38 of 1536 directions, so the concentration holds, but the count
    depends on the task. Tier 10 ends at five times the loss, so part of its rise may be that it
    is less converged.
*   What the measurement covers: eight pairs trained to a loss of 0.05, closer to memorisation
    than to a skill, and only q_proj and v_proj in layers 24 to 27. The reading supports "this
    kind of fine-tuning concentrates its update in few directions", not "weight updates are low
    rank".

## Script 04: Supervised fine-tuning with LoRA

The script teaches the base model one answer line, `TIER: C | ACTION: decline`, for a policy
application. The labels come from a rule on age and prior claims: tier C under 25 or
with 3 or more claims, tier A with no claims at 30 or older, tier B otherwise. vehicle_value
takes no part in the rule. Because a rule makes every label, each answer is plainly right or
wrong.

*   Part 1 builds 540 applications and splits them by (age, claims) pair, so none of the 15
    pairs in the 60 evaluation cases appears in the 480 training cases. Evaluation holds 12 of
    tier A, 8 of tier B and 40 of tier C.
*   Split at random instead, every evaluation pair also sat in training under another vehicle
    value, so a memorised pair looked like a learned rule.
*   Part 2 renders each example in the Alpaca instruction template (instruction, input,
    response).
    The end-of-sequence token is appended to the answer, or the model never learns to stop.
    Prompt tokens carry the label -100, so the loss counts only the answer. In the printed
    example 9 of 104 tokens carry a label: `TIER: C | ACTION: decline<｜end▁of▁sentence｜>`.
*   In part 4, PEFT attaches rank-8 adapters (α 16, dropout 0.05) to the four attention
    projections: 2,179,072 of 1,779,267,072 parameters, 0.1225%. The frozen weights still sit in
    memory for the forward pass but need no gradients and no optimiser state, which is why
    training fits in 6.13 GB.
*   Part 3 scores the base twice before training, and part 5 scores the adapter after 120 steps
    (batch 4, learning rate 2e-4). The loss goes from 1.6322 to 0.0179, the mean of the last ten
    steps.

    | Setting | Answers in the required shape | Exactly correct | By tier (A, B, C) |
    | :--- | ---: | ---: | :--- |
    | Always answering tier C | | 66.7% | 0/12, 0/8, 40/40 |
    | Base, instruction only | 8.3% | 0.0% | |
    | Base, rule written into the prompt | 100.0% | 66.7% | 0/12, 0/8, 40/40 |
    | Adapter, rule learned from examples | 100.0% | 93.3% | 12/12, 8/8, 36/40 |

*   Shape and correctness are counted apart, since a well-shaped line can hold the wrong tier.
    With the rule in the prompt, the base answers tier C to all 60 cases and scores exactly what
    always answering C scores. The adapter is the only setting that applies the rule. Its four
    misses are all tier C cases; the one printed, age 24 with one claim, came out as tier B.
    Seeds 11 and 42 gave the same 56 of 60.
*   Part 6 checks three things that fail in different ways. The saved directory holds only the
    adapter, 8.75 MB, and its config names the base, which must be present wherever it is
    loaded. Reloaded onto a fresh base, it reproduces the trained answers on the first four
    evaluation cases. Merged, W + BA becomes an ordinary weight: the same four answers, no
    adapter modules left, no extra cost at inference, and nothing left to swap out.

## Script 05: GRPO with rule-based rewards

GRPO (group relative policy optimisation) is reinforcement learning with no written answers.
The model samples a group of answers to one problem, rules score each one, and each answer is
judged against the rest of its group. Here the model is the 1.5B base with a rank-16 adapter on
the four attention projections, and the task is 60 subtraction problems, 12 held out.

*   The advantage of each answer is its reward minus the group mean, divided by the group's
    standard deviation. When every answer in a group scores the same, the advantages are all
    zero and the step changes nothing.
*   PPO, the usual RLHF algorithm, computes the advantage as reward minus a critic's prediction,
    so it trains a value network beside the policy. GRPO drops the critic and uses the group as
    the baseline. Dividing by the standard deviation makes a problem count for how its answers
    rank: rewards of [0, 1, 2, 3] and [0, 100, 200, 300] both become [-1.16, -0.39, +0.39,
    +1.16].
*   PPO also clips the probability ratio. Here each step samples fresh answers and updates once,
    so the ratio is 1 and the script leaves the clip out.
*   Five rules score each answer, and part 2 tries them on two hand-written samples:

    | Reward | Test | Maximum |
    | :--- | :--- | ---: |
    | Correct | The extracted answer equals the expected one | 2.0 |
    | Integer | A bare integer sits inside the answer tags | 0.5 |
    | Strict format | The exact line layout, newlines included | 0.5 |
    | Soft format | Tags in the right order, whitespace ignored | 0.5 |
    | Tag count | 0.125 for each of four tags that appears exactly once | 0.5 |

    A correct, well-formed sample scores 4.0 and plain prose 0.0. Correctness is worth as much
    as all the format rules together, so an answer that only looks right earns half at most.
*   The first run scored 0 for ten steps. Asked for tagged output, the base writes prose ("Okay,
    so I have this problem about beads in a jar..."), so every sample in every group scored zero
    and no gradient existed. The prompt now ends with `<reasoning>\n`. Some samples then beat
    their siblings, which is the one thing a group-relative method needs to start.
*   A KL penalty keeps the policy near the base. The base's log probabilities come from the same
    model with `disable_adapter()`, so no second copy is loaded. The penalty is the k3
    estimator, `p_ref/p - log(p_ref/p) - 1`, which is never negative and pulls toward the base;
    the plain log ratio has zero expected gradient.
*   Part 5 trains for 24 steps, each sampling six answers to each of two problems. Parts 4 and 6
    score the 12 held-out problems before and after, with greedy decoding:

    | | Before | After 24 steps |
    | :--- | ---: | ---: |
    | Answers holding the tag structure | 0/12 | 11/12 |
    | Right integer in the answer tags | 0/12 | 10/12 |
    | Right number anywhere in the text | 7/12 | 11/12 |
    | Cut off at 160 tokens | 6/12 | 1/12 |

*   Before training the base already writes the right number in 7 of 12 answers, for example
    `130 minus 85, which equals 45`, but never opens an answer tag. Most of 0/12 to 10/12 is
    the model learning where to put the answer, not learning to subtract.
*   The end token has to stay in the loss. This tokenizer pads with its end token, and an
    earlier mask dropped every padding position, the real stop included, so the trained model
    kept writing after `</answer>`. Single-change runs:

    | Run | Tag structure | Right integer | Cut off | Stops after `</answer>` |
    | :--- | ---: | ---: | ---: | :--- |
    | Plain log ratio, end token masked (old) | 8/12 | 7/12 | not printed | no |
    | Plain log ratio, end token kept | 10/12 | 7/12 | 4/12 | yes |
    | k3, end token masked | 8/12 | 7/12 | 3/12 | no |
    | k3, end token kept (current) | 11/12 | 10/12 | 1/12 | yes |

    Keeping the end token is what makes answers stop. The rise to 10/12 came only with both
    fixes, and each row is one run, so it is not credited to either.
*   The mean reward rose from 0.821 over the first five steps to 1.508 over the last five, but
    the curve is noisy, and 24 steps say nothing about convergence.
*   Sampling takes almost all of each step: over four groups, 30.9 s against 2.5 s for the log
    probabilities and the backward pass. That is why 24 steps took 400.3 s.
*   The log probabilities need full logits, 151,936 per position. Backpropagating the group of
    six in one pass peaked at 11.67 GB allocated, and in chunks of two at 6.56 GB, so chunking
    is what fits the 12 GB card.

## Script 06: Thinking budget control

Test-time compute control changes no weights. A reasoning model writes its deliberation between
`<think>` and `</think>` before it answers. The script decodes one token at a time with the cache
kept, so it can act in the middle of that phase; a single `generate` call would finish first.

*   Part 1 finds that `<think>` and `</think>` are single tokens, ids 151648 and 151649, so
    `</think>` can be banned in the logits.
*   To cap the thinking, the script writes `</think>` into the stream once the budget is spent,
    which pushes the model into its answer. To extend it, the script bans `</think>` and appends
    "Wait, let me check that again." each time the model is about to stop. The nudge tokens
    count toward the budget, so the extended setting runs under the widest cap, 400.
*   Part 2 asks eight questions with one numeric answer (four letter counts, four arithmetic),
    with greedy decoding. The answer is read from `\boxed{}` when the model writes one,
    otherwise as the first integer. Reading the first integer alone turned `4 kilograms (empty
    box) + 36 kilograms (12 bags) = \boxed{40}` into 4.

    | Setting | Correct | Mean thinking tokens | Stopped on its own | Seconds |
    | :--- | ---: | ---: | ---: | ---: |
    | Cap 24 | 1/8 | 24.0 | 0/8 | 16.7 |
    | Cap 64 | 2/8 | 64.0 | 0/8 | 25.3 |
    | Cap 160 | 5/8 | 110.9 | 7/8 | 33.0 |
    | Cap 400 | 5/8 | 112.9 | 8/8 | 34.7 |
    | Cap 400 and two nudges | 5/8 | 225.5 | 8/8 | 58.1 |

*   At caps of 24 and 64 no answer stopped on its own, so every answer there was forced, and
    truncation cost answers. These questions need about 110 thinking tokens: raising the cap
    from 160 to 400 adds 2 tokens on average and no correct answer.
*   Two nudges doubled the thinking and left accuracy at 5/8. Both nudges were used on every
    question and every answer still stopped on its own, so the extension was not cut short.
*   The four arithmetic questions are right from a cap of 160 up. Three of the four letter
    counts are wrong in every setting: raspberry reads 2 instead of 3, possessions reads 3 (the
    model copies the word as "possession") and beekeeper reads 3. Part 3 follows raspberry
    through every setting: with two nudges it spends 275 thinking tokens and still answers "The
    letter 'r' appears twice".
*   On this 1.5B distilled model "think again" did not recover a wrong answer. Published results
    come from much larger models.

## Script 07: Vision LoRA

A vision-language model is an image encoder and a language model joined by a connector. The
encoder turns pixels into embeddings, the connector maps them to the language model's width, and
the language model reads them alongside the text. The script fine-tunes SmolVLM-256M-Instruct to
answer one line, `GEAR: R | LAMP: off | NEEDLE: low | ODO: 226355`, about a rendered panel.

*   Part 1 draws each panel from its labels, so the label cannot disagree with the image and no
    photographs need labelling. The needle angle and its zone come from one number, with zones
    split at 0.34 and 0.67. The four fields differ in kind: gear is one of four, the lamp is on
    or off, the needle zone is a threshold on a continuous value, and the odometer is six
    digits.
*   112 panels at 384 × 384: 96 for training, 16 held out. All 16 held-out combinations of gear,
    lamp and zone also occur in training, but the fields are drawn independently and read from
    different places, and no held-out odometer value appears in training.
*   Part 2 counts the linear layers by tower:

    | Tower | Linear layers | Parameters | Share |
    | :--- | ---: | ---: | ---: |
    | Vision | 72 | 84,934,656 | 33.1% |
    | Language | 211 | 134,553,600 | 52.5% |
    | Connector | 1 | 7,077,888 | 2.8% |

*   At rank 16, adapters on the four attention projections of the language tower take 120
    modules and 1,843,200 parameters, 0.719% of the base model.
*   Part 4 attaches the adapter by full module path, because selecting by suffix over-attaches.
    The vision tower's attention layers also have q_proj, k_proj and v_proj, so asking for those
    suffixes adds 12 layers × 3 = 36 modules in the encoder: 156 modules, 2,727,936 parameters,
    1.064%. Counted after attaching: 120 modules, 0 of them in the image encoder.
*   Part 5 trains with the prompt masked out of the loss. The image placeholder expands to image
    tokens, so the script finds the mask length by encoding the prompt alone with the same
    image: 1189 prompt tokens of 1216. Training runs 150 steps, each adding up the gradients of
    two single examples. The loss goes from 1.9049 to 0.0292, the mean of the last ten steps.
    The end token, `<end_of_utterance>`, stays in the loss; the pad token is a different one.
*   Parts 3 and 6 score the 16 held-out panels field by field, before and after:

    | Field | Before | After |
    | :--- | ---: | ---: |
    | Answer in the required shape | 0/16 | 16/16 |
    | Gear | 0/16 | 16/16 |
    | Lamp | 0/16 | 16/16 |
    | Needle zone | 0/16 | 14/16 |
    | Odometer | 0/16 | 16/16 |
    | Right odometer digits anywhere in the text | 7/16 | 16/16 |

*   Before training the base already writes the right odometer digits in 7 of 16 answers, so
    for the odometer the zero is a format failure, and most of 0/16 to 16/16 is the adapter
    teaching the answer line.
*   Only the needle zone falls short, and its two misses (0.826 and 0.587, both read as low) are
    not near a boundary, while the four panels closest to one were read correctly. The training
    zones are balanced, so with 16 panels the misses may be chance.
*   Adapting the language tower alone brought three of four fields to 16/16, so in this run the
    encoder did not need adapting for them. The saved adapter is 7.42 MB.
