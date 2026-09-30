# Machine learning and deep learning foundations

Script 01 generates the module's datasets from stated formulas, so every later score can be
checked against a known answer. Scripts 02 to 06 work on those tables with the classic tabular
toolkit: loading, feature engineering, gradient boosting, leakage, nine classifiers and model
ensembling. Scripts 07 and 08 train one small network, first with nothing but numpy arrays and
then through PyTorch, TensorFlow and Keras. This document explains what each script does and the
ideas it relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_build_tabular_datasets.py` | Synthetic tables with known ground truth: vehicle pricing, employee attrition, speaker acoustics and property valuation |
| 02 | `02_eda_that_silently_lies.py` | Exploratory data analysis: one file read with two separators, where every usual check agrees and 1888 rows lose their label |
| 03 | `03_feature_engineering_and_boosting.py` | Feature engineering into CatBoost, then a feature importance audit against the generating formula |
| 04 | `04_leakage_and_split_discipline.py` | Data leakage from a scaler, from target encoding at four cardinalities and from duplicate rows, each scored on an untouched holdout |
| 05 | `05_classifier_toolbox_and_thresholds.py` | Nine classifiers on two tables, feature scaling and encoding, the decision threshold, and coefficients read back against the generator |
| 06 | `06_ensembling_blend_vs_stack.py` | Model ensembling: simple averaging, inverse-error weighting and stacking over out-of-fold predictions |
| 07 | `07_neural_net_from_scratch.py` | A network in numpy: activation slopes, a forward pass by hand, backpropagation with a gradient check, and a bias ablation |
| 08 | `08_framework_abstraction_ladder.py` | The same network in numpy, PyTorch, TensorFlow and Keras, graph mode, and a data-parallel strategy |

## Shared setup

*   Script 01 writes everything the others read into `data/`, so it runs first. The other seven
    are independent of each other. `data/` and `outputs/` are rebuilt by a rerun and are not
    tracked.
*   Everything runs on CPU and no script calls a hosted model. A cold run of all eight takes
    about three and a half minutes, most of it in script 03.
*   The dependencies are numpy, pandas, scikit-learn, matplotlib, xgboost, lightgbm, catboost,
    ngboost, torch and tensorflow-cpu. NGBoost depends on lifelines, which caps pandas below
    3.0, so installing it downgrades pandas unless pandas is pinned. The cap is precautionary:
    NGBoost fits and predicts normally on pandas 3.0.5.

## Script 01: Datasets from a known formula

A downloaded dataset can tell you a model scored 801. It cannot tell you how close 801 is to the
best achievable, or whether a feature that ranked highly deserved to. A generated dataset can,
because the answer was written down first.

| File | Shape | Purpose |
| :--- | :--- | :--- |
| `vehicle_listings.csv` | 30 000 × 30 | Regression, read by scripts 02, 03, 04 and 06 |
| `vehicle_holdout.csv` | 10 000 × 30 | Never used for any fit; the holdout of scripts 03, 04 and 06 |
| `employee_attrition.csv` | 1 800 × 33 | Imbalanced binary classification, 16.6% positive |
| `speaker_acoustics.csv` | 3 200 × 21 | Near-separable binary classification, balanced |
| `property_valuation.csv` | 506 × 13 | Small dense regression for scripts 07 and 08 |

*   Part 1 prices each vehicle with a formula, which part 6 prints:

    | Term | Value |
    | :--- | :--- |
    | Brand tier base | 6 500 to 34 000 across ten tiers |
    | Depreciation | value halves every 6 years of age, counted in calendar days over 365 |
    | `power` | +41 per unit, capped at 400 |
    | `odometer_km` | −430 per unit |
    | `gearbox` automatic | +1 900 |
    | `undamaged_flag == 0` | × 0.82 (the flag is 1 for an undamaged car) |
    | `v_0` to `v_4` | weights 2600, −1800, 1200, 900, −650 |
    | `v_5` to `v_14` | no effect, pure noise columns |
    | Residual noise | σ = 900 |

*   Fifteen anonymous columns look alike and only five of them enter the price. Scripts 02 and
    03 check whether that is recovered.
*   The noise sets a floor: no model can beat 900 × √(2/π) ≈ 718 MAE on this data. Script 03's
    787.57 is therefore 10% above the best achievable, which is a statement where 787 alone is
    not.
*   Part 2 writes the vehicle files space separated, with a missing value written as nothing at
    all: two adjacent spaces. The row keeps the right number of separators. Script 02 shows what
    that does to a reader.
*   Part 3 draws attrition from a stated log-odds model:

    | Term | Weight | Term | Weight |
    | :--- | ---: | :--- | ---: |
    | OverTime | +1.25 | YearsAtCompany | −0.085 |
    | MaritalStatus Single | +0.85 | MonthlyIncome | −0.000105 |
    | BusinessTravel Frequently | +0.55 | JobSatisfaction | −0.24 |
    | DistanceFromHome | +0.030 | JobInvolvement | −0.20 |
    | NumCompaniesWorked | +0.11 | WorkLifeBalance | −0.17 |
    | Age | −0.019 | StockOptionLevel | −0.22 |

*   The terms are not centred and average −3.19 on this table, so an intercept of 0 would give
    only 9.2% leavers. The script bisects the intercept for a target rate of 0.16, and the file
    lands at 16.6%. `EmployeeCount` (always 1) and `StandardHours` (always 80) are constant on
    purpose, for script 05.
*   In part 4, speaker acoustics is the contrast: `mean_fundamental` alone nearly separates the
    two classes (0.1709 against 0.1162). One cut at the midpoint, 0.1436, sorts 96.1% of rows.
*   Part 5's property table has twelve numeric predictors on scales four orders of magnitude
    apart, eleven linear terms and one saturating `tanh` term on `vacancy_rate`, so a linear
    model cannot be perfect. The target is clipped to 5 to 50.
*   The property table stands in for the housing dataset scikit-learn withdrew on ethical
    grounds: one of its columns was the proportion of Black residents in a district, used as a
    price predictor. A synthesised table also has known coefficients, which script 07 needs.

## Script 02: Auditing the load

The finding here is not that a wrong separator raises an error. It is that it does not.

*   Part 1 reads the same file twice: `sep=" "` takes one space as one separator, and
    `sep=r"\s+"` takes a run of whitespace as one. In part 2, the checks people usually run
    agree:

    | Check | `sep=" "` | `sep=r"\s+"` |
    | :--- | :--- | :--- |
    | Shape | (30000, 30) | (30000, 30) |
    | Column names identical | yes | yes |
    | Sum of first column | 450015000 | 450015000 |
    | `price` mean | 12302.45 | 12288.41 |
    | Exception raised | no | no |

*   A gap of 14 on a mean of 12 300 asks nobody to investigate.
*   Part 3 runs a check with a known right answer, the number of distinct values, and it does
    not agree:

    | Column | Should have | `sep=" "` | `sep=r"\s+"` |
    | :--- | ---: | ---: | ---: |
    | `gearbox` | 2 | 2 | 264 |
    | `undamaged_flag` | 2 | 2 | 1484 |
    | `offer_type` | 1 | 1 | 1844 |

*   Part 4 shows the mechanism on the first affected row. Split on one space, the empty field is
    a `NaN`. Split on a run of whitespace, it is not there at all, so from that gap onward every
    column shifts one place left. The engine power becomes the gearbox, and the odometer becomes
    the power.
*   In part 5, 1 888 of 30 000 rows (6.29%) shift. Both readings report exactly 1 925 missing
    values; only the column holding them changes:

    | Column | `sep=" "` | `sep=r"\s+"` |
    | :--- | ---: | ---: |
    | `body_type` | 632 | 0 |
    | `fuel_type` | 627 | 0 |
    | `gearbox` | 666 | 0 |
    | `price` | 0 | 1888 |
    | `v_14` | 0 | 37 |

*   The shift runs to the end of the row, so the gaps land in the last column, and the last
    column is `price`, the label. The 37 rows with two holes lose `v_14` as well.
*   An exploratory pass needs at least one count with a known right answer. Shape, column
    names, missing totals and means all failed this test.
*   Part 6 profiles the correct read:

    | Item | Measured |
    | :--- | :--- |
    | Columns with gaps | `body_type` 632 (2.11%), `fuel_type` 627 (2.09%), `gearbox` 666 (2.22%) |
    | `price` | min 250, median 11272, max 49521, skew 0.782 |
    | Five anonymous columns most correlated with `price` | `v_0` to `v_4`, the five in the formula |
    | Weakest paid column | `v_4`, at 0.0834 |
    | Strongest noise column | `v_11`, at 0.0050 |

*   A correlation sort, with no model involved, picks the five real drivers out of fifteen
    identical-looking columns. That is a claim that can be refuted before anything is trained.

## Script 03: Feature engineering and the importance audit

*   Part 1 derives time features (age in days and years, calendar parts, season,
    `km_per_year`, an age segment), interaction features (`power_per_year`, `power_times_km`,
    `brand_model`, `latent_mean`, `latent_spread`), and missing and outlier flags.
*   1 683 rows carry a listing date earlier than the registration date. The age is clipped at
    zero, which leaves 1 690 rows at exactly 0.0: the 1 683, plus 7 listed on the day they were
    registered. Part 2 then cuts the age into five segments:

    | Bin edges | Rows outside every segment |
    | :--- | ---: |
    | `[0, 1, 3, 5, 10, 100]` | 1 690 |
    | `[-0.01, 1, 3, 5, 10, 100]` | 0 |

*   `pandas.cut` excludes the leftmost edge, so an age of exactly 0.0 belongs to no interval.
    Clipping bad dates is right and binning age is right. Nothing raises, and the column gains
    nulls in 5.6% of rows. Printing the count that falls outside the bins costs one line.
*   Part 3 checks the 19 flags for variance. Only `power_outlier` marks any rows (150). The gaps
    in this file are in three categorical columns, and the missing flags were built over the
    numeric ones. The odometer clip sits at 31.5 while the data maxes out at 15. A missing
    marker on a column that is never missing is a column of zeros with a descriptive name.
*   Part 4 adds group statistics on `brand`, frequency encoding and `brand_mean_ratio` (the
    brand mean over the training price mean), all computed from the training rows only.
    Computing `brand_price_mean` over train and holdout together would put holdout prices into a
    feature. Script 04 measures the same kind of leak.
*   Part 5 trains CatBoost for up to 1 200 rounds at depth 8 and scores the untouched holdout
    file:

    | Metric | Measured |
    | :--- | ---: |
    | Best iteration | 1031 of 1200 (early stopping) |
    | Holdout MAE | 787.57 |
    | RMSE | 994.78 |
    | R² | 0.9829 |
    | Floor implied by σ = 900 | 718 |

*   Part 6 counts what the model used. Of 68 features, 19 (28%) have importance exactly 0.0: the
    17 missing flags, `odometer_outlier` and `offer_type`. Ten more are below 0.05, and 18
    features carry 95% of the total importance. A control, engine power added to a model
    identifier, scores 0.1197.
*   Part 7 checks the twelve strongest against the formula:

    | Feature | Importance | Traces to the formula |
    | :--- | ---: | :--- |
    | `power_per_year` | 19.162 | computed from `power` and age |
    | `v_0` | 14.080 | yes (weight 2600, the largest) |
    | `v_1` | 8.003 | yes (−1800) |
    | `brand` | 6.637 | yes |
    | `brand_price_std` | 5.791 | computed from `brand` only |
    | `brand_model` | 5.513 | no: it mixes in `model_code`, which the formula never uses |
    | `undamaged_flag` | 4.795 | yes (× 0.82) |
    | `brand_mean_ratio` | 4.789 | computed from `brand` only |
    | `odometer_km` | 4.349 | yes (−430) |
    | `v_2` | 3.895 | yes (1200) |
    | `brand_price_mean` | 3.193 | computed from `brand` only |
    | `power` | 2.676 | yes (+41 per unit) |

*   Eleven of the twelve are a formula term or computed from formula terms only, and the
    exception still carries `brand` inside it. The five paid anonymous columns rank 2, 3, 10, 15
    and 19; the ten noise columns rank 27 to 38.
*   "I engineered 68 features" is a statement about the pipeline. The zeros in
    `get_feature_importance()` are the statement about the model.

## Script 04: Leakage and split discipline

Three leaks, each run as an honest arm and a leaky arm, and each scored on a holdout that took
part in no fit. The gap between validation and holdout is the column to watch.

*   In part 1, the honest baseline scores 955.56 on validation and 963.10 on the holdout, a gap
    of −7.53. The two agree, which is what a validation score is for.
*   Part 2 fits the scaler on train and holdout together. Both arms fit it on the whole training
    file before the internal split, so what they contrast is the holdout crossing into the
    transform. Trees ignore scale, so this runs on a nearest-neighbour model:

    | | Validation MAE | Holdout MAE |
    | :--- | ---: | ---: |
    | Honest | 4862.87 | 4947.71 |
    | Leaky | 4862.76 | 4947.83 |

*   The leak is worth 0.11 MAE. A minimum and a maximum are two numbers per column, and handing
    them over leaks almost nothing.
*   Part 3 encodes a key by the mean price within that key. If the mean includes the row being
    encoded, part of that row's own label flows back into its features, and how large a part
    depends on how many rows share the key:

    | Key | Distinct keys | Rows/key | Honest val | Leaky val | Leaked | Leaky holdout |
    | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
    | `brand` | 10 | 3000.0 | 955.56 | 955.56 | 0.00 | 963.10 |
    | `model_code` | 120 | 250.0 | 958.02 | 951.91 | 6.11 | 960.25 |
    | `region_code` | 3997 | 7.5 | 991.41 | 957.38 | 34.02 | 974.14 |
    | `region_code × brand` | 21085 | 1.4 | 4015.31 | 1117.83 | 2897.48 | 3316.20 |

*   The honest arm keeps validation prices out of the mean, but it still encodes each fit row
    with a mean that includes that row's own price. At 1.4 rows per key that mean is mostly the
    row's own label, and the tree learns to lean on it. A third arm encodes each fit row from
    the other four folds only:

    | Key | Out-of-fold val | Out-of-fold holdout |
    | :--- | ---: | ---: |
    | `brand` | 953.58 | 964.54 |
    | `model_code` | 955.41 | 961.56 |
    | `region_code` | 955.06 | 960.24 |
    | `region_code × brand` | 950.92 | 959.96 |

*   The danger belongs to the key, not the technique. Out of fold, the worst key beats the
    baseline's 955.56 and 963.10.
*   Part 4 duplicates a quarter of the listings, then splits at random. 2 406 of the 7 500
    validation rows have a `listing_id` also in the fit half. Validation MAE falls to 876.01
    while the holdout stays at 973.49, a gap of −97.48, thirteen times the baseline's. The split
    was over rows, and the rows were not independent.
*   Part 5 puts them side by side:

    | Setup | Validation | Holdout | Gap | Model |
    | :--- | ---: | ---: | ---: | :--- |
    | Honest baseline | 955.56 | 963.10 | −7.53 | boosted trees |
    | Target encoding leak (extreme key) | 1117.83 | 3316.20 | −2198.37 | boosted trees |
    | Duplicate rows | 876.01 | 973.49 | −97.48 | boosted trees |
    | Scaler leak | 4862.76 | 4947.83 | −85.07 | nearest neighbours |

*   Part 6 reads the gaps. For the scaler, the duplicates and the three milder encodings, the
    holdout barely moves: the leak made the report better and left the model where it was. The
    extreme key is the exception, where the holdout went from 963 to 3 316.
*   None of the leaks raised an error or printed an implausible number. The gap catches the
    target encoding and the duplicates. It misses the scaler: the two scaler arms' gaps differ
    by 0.23 (−84.84 against −85.07), so that row's gap comes from the nearest-neighbour model. A
    leak that small shows only when the two fits sit side by side, as in part 2.

## Script 05: Classifier toolbox and thresholds

Nine classifiers, two tables, one stratified 75/25 split. The run takes about 22 seconds.

*   In part 1, predicting that nobody ever leaves scores 0.8356 accuracy on the attrition test
    set. That is the bar every model has to clear first:

    | Model | AUC | Accuracy | Flagged at 0.5 |
    | :--- | ---: | ---: | ---: |
    | logistic regression | 0.8078 | 0.8511 | 35 |
    | catboost | 0.7818 | 0.8378 | 29 |
    | ngboost | 0.7806 | 0.8378 | 23 |
    | xgboost | 0.7784 | 0.8378 | 37 |
    | lightgbm | 0.7728 | 0.8356 | 40 |
    | gradient boosting | 0.7718 | 0.8378 | 33 |
    | decision tree | 0.7653 | 0.8289 | 49 |
    | random forest | 0.7653 | 0.8333 | 17 |
    | svm rbf (unscaled) | 0.4973 | 0.8356 | 0 |

*   Only five of the nine beat the constant guess on accuracy. The SVM flagged nobody and
    matched it exactly. On a 16.6% positive class, accuracy mostly reports the class balance,
    and the ranking metric is what separates the models.
*   NGBoost's default base tree takes no seed of its own, so the script passes it one. Without
    it, NGBoost's AUC came out 0.7806 or 0.7828 from one run to the next.
*   Part 2 runs the same nine on the acoustic table. The best AUC is 0.9952 (catboost) against
    0.8078 on attrition, while the spread across the eight usable models is 0.0261 and 0.0426.
    Changing the table moves the best AUC by 0.1874; choosing among the models moves it by less
    than 0.05. The ceiling belongs to the data.
*   Part 3 min-max scales the columns, `x' = (x - min) / (max - min)`, fitted on the training
    rows only:

    | Model | Raw AUC | Scaled AUC | Change |
    | :--- | ---: | ---: | ---: |
    | svm rbf | 0.4973 | 0.7789 | +0.2816 |
    | logistic regression | 0.8078 | 0.8097 | +0.0019 |
    | gradient boosting | 0.7718 | 0.7722 | +0.0004 |
    | decision tree | 0.7653 | 0.7653 | 0.0000 |
    | xgboost | 0.7784 | 0.7784 | 0.0000 |
    | ngboost | 0.7806 | 0.7806 | 0.0000 |
    | catboost | 0.7818 | 0.7818 | 0.0000 |
    | random forest | 0.7653 | 0.7649 | −0.0004 |
    | lightgbm | 0.7728 | 0.7608 | −0.0120 |

*   `MonthlyIncome` spans 1749 to 16985 and `JobSatisfaction` spans 1 to 4, so a distance in the
    raw space is a distance in monthly income. A tree compares values inside one column and
    never computes a distance. A single LightGBM tree scores 0.7307 on both versions, so its
    −0.0120 builds up over the 400 boosting rounds.
*   Standardisation, `z = (x - mean) / sd`, is the other common scaling. An outlier still shifts
    the mean and sd, but it does not pin the other rows against zero the way it pins them under
    min-max. The networks in scripts 07 and 08 are fed standardised columns.
*   In part 4, label encoding (31 columns) scores 0.7728 and one-hot (48 columns) 0.7654 on
    LightGBM. Over ten other splits, one-hot minus label averages +0.0001, from −0.0202 to
    +0.0124, and the sign changes from split to split. A tree can cut the false order label
    encoding puts on `JobRole` back into the right pieces.
*   In part 5, removing the two constant columns changes AUC by +0.0000. A column with no
    variance cannot split anything, so dropping it is housekeeping.
*   Part 6 moves the threshold on the best-ranking model:

    | Threshold | Flagged | Precision | Recall | F1 | Accuracy | AUC |
    | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | 0.10 | 210 | 0.2905 | 0.8243 | 0.4296 | 0.6400 | 0.8078 |
    | 0.16 | 163 | 0.3558 | 0.7838 | 0.4895 | 0.7311 | 0.8078 |
    | 0.20 | 138 | 0.3986 | 0.7432 | 0.5189 | 0.7733 | 0.8078 |
    | 0.30 | 87 | 0.4483 | 0.5270 | 0.4845 | 0.8156 | 0.8078 |
    | 0.50 | 35 | 0.6000 | 0.2838 | 0.3853 | 0.8511 | 0.8078 |
    | 0.70 | 12 | 0.5833 | 0.0946 | 0.1628 | 0.8400 | 0.8078 |

*   AUC reads the ordering of the scores, and sliding a cut through a fixed ordering cannot
    change it. Everything to its left follows the cut: recall from 0.82 to 0.09, precision from
    0.29 to 0.60. "85% accurate" without the threshold states nothing.
*   Part 7 flags the top 75 of 450 scores to match the training base rate of 0.1659. The
    threshold that does that is 0.3466:

    | Cut | Flagged | Caught | Missed | False alarms | Precision | Recall |
    | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
    | Default 0.5 | 35 | 21 | 53 | 14 | 0.6000 | 0.2838 |
    | Forced rate | 75 | 35 | 39 | 40 | 0.4667 | 0.4730 |

*   Same scores, one number changed by hand: fourteen more leavers caught and twenty-six more
    false alarms. Which row is better depends on what a conversation with a flagged employee
    costs against what losing one costs, and no metric in the script knows that.
*   Part 8 fits the logistic regression on all 1800 rows and reads its coefficients back against
    the generator. Two things have to be right first. The categorical columns are one-hot
    encoded with the first category dropped, which the generator gives no weight. The fit runs
    on standardised columns, so each coefficient is a weight per standard deviation, and each
    true weight is multiplied by its own column's spread to match:

    | Term | Raw weight | × spread | Fitted | True rank | Fitted rank |
    | :--- | ---: | ---: | ---: | ---: | ---: |
    | `YearsAtCompany` | −0.0850 | −0.9899 | −0.6203 | 1 | 1 |
    | `OverTime` | +1.2500 | 0.5481 | 0.5583 | 2 | 2 |
    | `MaritalStatus_Single` | +0.8500 | 0.3928 | 0.3628 | 3 | 4 |
    | `MonthlyIncome` | −0.0001 | −0.3802 | −0.2117 | 4 | 9 |
    | `NumCompaniesWorked` | +0.1100 | 0.3198 | 0.2062 | 5 | 10 |
    | `JobSatisfaction` | −0.2400 | −0.2671 | −0.1849 | 6 | 11 |
    | `StockOptionLevel` | −0.2200 | −0.2496 | −0.3842 | 7 | 3 |
    | `DistanceFromHome` | +0.0300 | 0.2488 | 0.2200 | 8 | 8 |
    | `Age` | −0.0190 | −0.2271 | −0.2685 | 9 | 5 |
    | `JobInvolvement` | −0.2000 | −0.2195 | −0.2621 | 10 | 6 |
    | `BusinessTravel_Travel_Frequently` | +0.5500 | 0.2177 | 0.2245 | 11 | 7 |
    | `WorkLifeBalance` | −0.1700 | −0.1916 | −0.0776 | 12 | 12 |

*   All 12 signs come back, and 5 of 12 fitted ranks land within three places of the true rank.
    Per unit, `OverTime` has the largest weight, but it is a yes-or-no column whose whole range
    is one step, while `YearsAtCompany` moves over decades. Per standard deviation, the
    generator itself ranks `YearsAtCompany` first, and so does the fit.
*   The three yes-or-no terms come back close to their true size (0.558 against 0.548, 0.363
    against 0.393, 0.225 against 0.218). Twelve overlapping effects and 1800 rows recover the
    directions, not a league table.

## Script 06: Ensembling, measured

Four learners, 30 000 training rows, 10 000 holdout rows, 4-fold out-of-fold predictions. The
run takes about 25 seconds.

*   Part 1 scores each learner alone:

    | Learner | Out-of-fold MAE | Holdout MAE |
    | :--- | ---: | ---: |
    | xgboost | 969.69 | 946.75 |
    | lightgbm | 949.77 | 933.75 |
    | catboost | 776.47 | 776.24 |
    | neighbours | 4805.38 | 4879.89 |

*   In part 2, the three boosted models' predictions correlate at 0.9956 and their errors at
    0.8092; the neighbour model's errors correlate with theirs at 0.2304. Predictions correlate
    because they are mostly the price. The errors are what is left once the price is taken out,
    and averaging can only cancel what is left.
*   Part 6 lists every combination against the best single model:

    | Combination | Holdout MAE | Gain | Error correlation |
    | :--- | ---: | ---: | ---: |
    | Best single model (catboost) | 776.24 | 0.00 | n/a |
    | Simple average of the three boosted | 830.85 | −54.61 | 0.8092 |
    | Weighted by inverse holdout error | 823.73 | −47.48 | 0.8092 |
    | Stack over the three boosted | 775.61 | +0.63 | 0.8092 |
    | Simple average of all four | 1498.64 | −722.40 | 0.5198 |
    | Stack over all four | 775.58 | +0.66 | 0.5198 |

*   In part 3, the simple average is 54.61 worse than catboost. The best member scores 776 and
    the worst 947, and an equal vote spends the good model's accuracy on carrying the other two.
    Weighting by inverse holdout error recovers part of that, with one parameter per model read
    off the set being predicted.
*   Part 4 learns the weights out of fold, so nothing about the holdout takes part in choosing
    them. The stack of three gives catboost 0.9462, lightgbm 0.0529 and xgboost 0.0072, with an
    intercept of −74.02.
*   Part 5 adds the neighbour model, whose error is 6.3 times the best model's. A simple average
    must take it at full strength, and 830.85 becomes 1498.64. The stack gives it a weight of
    0.0049, and the MAE moves by 0.03.
*   The best gain cannot be told apart from zero. The stack of four beats catboost by 0.66 MAE
    (0.08%), and resampling the 10 000 holdout rows 2 000 times puts that gain between −0.40
    and +1.74. The stack also rescales: catboost alone, rescaled by the same out-of-fold linear
    fit (intercept −56.04, slope 1.0049), scores 775.78, which is 0.46 of the 0.66.
*   Three of the five combinations came out worse than doing nothing. Whether a model helps a
    blend depends on two numbers: its error, and how much its errors overlap the others'.
*   Across scripts 03, 04 and 06, on the same untouched holdout:

    | Step | Holdout MAE | Configuration |
    | :--- | ---: | :--- |
    | A plain feature set, boosted trees | 963.10 | 22 raw columns, LightGBM, 400 rounds |
    | Same data, a stronger model | 776.24 | 26 columns, CatBoost, 400 rounds, depth 7 |
    | Engineered features | 787.57 | 68 features, CatBoost, 1200 rounds, depth 8 |
    | Best ensemble over four models | 775.58 | stack over out-of-fold predictions |

*   These are not a controlled ablation: the model, the round budget and the depth change along
    with the feature set. Switching to CatBoost and adding four columns is worth about 187 MAE.
    Script 03's engineered features did not beat it, and the ensemble added 0.66, inside the
    noise.

## Script 07: A network with nothing but arrays

No framework appears anywhere in this script, so every quantity in it can be checked by hand. A
neuron computes `a = f(Σ wᵢxᵢ + b)`: a linear model plus a non-linear function. With a sigmoid
as `f`, one neuron is a logistic regression. Without any `f`, `W₂(W₁x) = (W₂W₁)x`, and the whole
stack collapses to a single linear layer.

*   Part 1 measures the three activations:

    | x | sigmoid | slope | tanh | slope | relu | slope |
    | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | −6.0 | 0.0025 | 0.0025 | −1.0000 | 0.0000 | 0.0000 | 0.0000 |
    | −0.5 | 0.3775 | 0.2350 | −0.4621 | 0.7864 | 0.0000 | 0.0000 |
    | 0.0 | 0.5000 | 0.2500 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
    | 0.5 | 0.6225 | 0.2350 | 0.4621 | 0.7864 | 0.5000 | 1.0000 |
    | 6.0 | 0.9975 | 0.0025 | 1.0000 | 0.0000 | 6.0000 | 1.0000 |

*   The sigmoid's slope, `f(x)(1 − f(x))`, peaks at 0.25, and its output is never negative, so
    every weight into one neuron gets a gradient of the same sign. Tanh is centred on zero with
    slope 1 at the origin, but both tails still flatten. ReLU's slope is exactly 1 wherever the
    input was positive; a unit pushed permanently negative gets gradient 0 and never returns.
*   Backpropagation multiplies one slope per layer. Part 2 multiplies each activation's best
    slope and leaves the weights out, so it shows the activation's share of the shrinking, not
    a bound on the gradient:

    | Layers | sigmoid | tanh | relu |
    | ---: | ---: | ---: | ---: |
    | 1 | 2.500e-01 | 1.000e+00 | 1.000e+00 |
    | 5 | 9.766e-04 | 1.000e+00 | 1.000e+00 |
    | 10 | 9.537e-07 | 1.000e+00 | 1.000e+00 |
    | 40 | 8.272e-25 | 1.000e+00 | 1.000e+00 |

*   Even in the best case, ten sigmoid layers deliver a millionth of the error to the first.
    Tanh and ReLU both peak at 1, so their best case never shrinks; the difference is that tanh
    reaches 1 only at exactly zero (0.7864 at 0.5, 0.0707 at 2), and ReLU at every positive
    input. A falling total loss does not show that the early layers learn. Their gradient norm
    does.
*   Part 3 pushes the input [1.0, 0.5] through a 2 → 3 → 2 → 2 network of evenly spaced small
    weights, so every step can be checked on a calculator. The first hidden unit is
    1.0 × 0.1 + 0.5 × 0.2 + 0.1 = 0.3, and the output is [0.316827, 0.696279]. The output layer
    has no activation because this is a regression. The network trained below is a different
    one, with one hidden layer of 24 ReLU units.
*   Backpropagation carries the error through the transpose of the weight matrix that carried
    the input forward. Two details are easy to get wrong in the numpy code: ReLU's derivative
    tests the pre-activation `z1`, not the activated output, and a loss taken as a mean with a
    gradient taken as a sum silently multiplies the learning rate by the batch size.
*   Part 4 checks the derived gradients against a finite difference of the loss, on the
    training network at its starting weights and 64 rows:

    | Parameter | Derived | Finite difference | Relative error |
    | :--- | ---: | ---: | ---: |
    | `w1[241]` | 0.53008212 | 0.53008205 | 1.42e-07 |
    | `w1[250]` | 4.55895805 | 4.55895815 | 2.28e-08 |
    | `b1[12]` | −21.79938403 | −21.79938394 | 4.09e-09 |
    | `w2[10]` | −50.39624449 | −50.39624455 | 1.12e-09 |
    | `b2[0]` | −62.41542020 | −62.41542019 | 2.03e-10 |

*   The worst relative error is 1.42e-07, over 10 of the 337 parameters. A derivation that is
    wrong is usually wrong for a whole array at once, which a sample from each array finds. A
    wrong gradient still trains, just towards somewhere else.
*   Part 5 trains twice from the same weights for 4 000 epochs, once with the bias gradients
    and once with the biases left at zero:

    | Run | First loss | Final loss | Test MAE | Average house |
    | :--- | ---: | ---: | ---: | ---: |
    | Biases updated | 1049.4831 | 4.3988 | 2.4799 | 26.96 |
    | Biases left at zero | 1049.4831 | 15.5387 | 4.7898 | 0.00 |

*   25 of the 337 parameters (7.4%) are frozen, and the output bias ends at 5.8580 against
    0.0000. Without biases every layer maps zero to zero, so the average house, where every
    standardised input is zero, is priced at exactly 0.00 whatever the weights are. That costs
    3.5 times the final loss and 1.9 times the test MAE, with no crash and no warning.
*   The run with biases was lowest at epoch 3266 (4.2681) and ended at 4.3988, so plain gradient
    descent at this learning rate was not only going down when training stopped.
*   Part 6 ranks the features by `|w1| × |w2|` summed over every path to the output, against
    how much each term moves the target over the data (`|coefficient| × column spread`, and
    the saturating term evaluated):

    | Feature | Term spread | True rank | Network rank |
    | :--- | ---: | ---: | ---: |
    | `vacancy_rate` (saturating) | 6.405 | 1 | 1 |
    | `rooms` | 3.429 | 2 | 3 |
    | `floor_area` | 2.170 | 3 | 6 |
    | `distance_to_centre` | 2.013 | 4 | 12 |
    | `pupil_teacher_ratio` | 1.724 | 5 | 5 |
    | `crime_index` | 1.552 | 6 | 7 |
    | `school_rating` | 1.141 | 7 | 8 |
    | `tax_rate` | 1.078 | 8 | 11 |
    | `lot_size` | 0.856 | 9 | 4 |
    | `transit_index` | 0.847 | 10 | 10 |
    | `noise_level` | 0.681 | 11 | 9 |
    | `build_year` | 0.520 | 12 | 2 |

*   The rank correlation over all twelve is 0.25, and `build_year` (true 12, network 2) is the
    largest miss. Weight size is a loose reading of what a network uses.
*   The network's test MAE is 2.4799, a linear regression on the same split scores 2.0518, and
    a perfect model would score about 1.76, the MAE of noise with σ = 2.2 (`σ × √(2/π)`, not
    σ). On a table this close to linear, the network does not beat a linear regression.

## Script 08: The framework ladder

One network, 12 → 10 → 1 with ReLU and a squared-error loss, written four times. Each rung hands
more of the work to the framework:

| Rung | Implementation | You write | The framework writes |
| :--- | :--- | :--- | :--- |
| 1 | numpy | structure, forward, loss, gradients, update | matrix multiplication |
| 2 | PyTorch | structure, training loop, update | gradients |
| 3 | TensorFlow with a gradient tape | structure, training loop | gradients |
| 4 | Keras `fit` | structure only | loop, gradients, optimiser, loss history |

*   Part 1 draws the starting weights once with numpy and loads them into every rung. All three
    loops take the same 300 full-batch steps at learning rate 0.05, with no momentum, no
    shuffling, and everything in float32. Numpy defaults to float64; without pinning the dtype
    the curves separate for a reason that has nothing to do with the abstraction.
*   Parts 2 to 4 run the three loops. The numpy gradients are six lines of chain rule, and
    PyTorch replaces them with `loss.backward()`. PyTorch adds each backward pass onto the
    gradients already stored, so the loop clears them first, a step with no counterpart in the
    numpy version.
*   Part 5 lines the three up and scores each on the test rows:

    | Implementation | First loss | Final loss |
    | :--- | ---: | ---: |
    | numpy, hand-derived gradients | 1002.466614 | 5.064641 |
    | PyTorch, autograd | 1002.466553 | 5.064640 |
    | TensorFlow, gradient tape | 1002.466614 | 5.064638 |

    | Pair | Largest loss gap over the run | Final loss gap | Largest weight gap |
    | :--- | ---: | ---: | ---: |
    | numpy vs PyTorch | 1.556e-03 | 4.768e-07 | 2.217e-05 |
    | numpy vs TensorFlow | 1.392e-03 | 2.861e-06 | 2.408e-05 |
    | PyTorch vs TensorFlow | 1.556e-03 | 2.384e-06 | n/a |

*   Automatic differentiation removed the writing of the derivatives, not the derivatives: the
    three agree to the last few digits a float32 holds.
*   In part 6, Keras `compile` and `fit` replace the loop, and `history.history["loss"]` is the
    list the loop used to build. 300 epochs land 4.768e-07 from the hand-written run, at test
    MAE 2.0224.
*   Part 7 runs the same training step 200 times eagerly and 200 times as a graph traced by
    `@tf.function`, with the tracing call left out of the timing. Eager took 0.398 s and the
    graph 0.050 s, a ratio of 7.96x in the recorded run; the ratio varies from run to run. Both
    loops start from the same weights and end at loss 5.947921.
*   On a network this small, most of the runtime is Python overhead between operations, and
    that is what the graph saves. On a large model the arithmetic dominates, so the same switch
    saves far less.
*   Part 8 wraps the model in `MirroredStrategy`. The conversion is two lines: create the
    strategy, and build and compile the model inside `strategy.scope()`. Training and prediction
    stay as they were.

    | Item | Measured |
    | :--- | :--- |
    | Visible devices | ['CPU'] |
    | Replicas in sync | 1 |
    | Final loss inside the scope | 5.064640 |
    | Final loss outside it | 5.064640 |
    | Difference | 0.000e+00 |

*   With one replica nothing was split and nothing ran in parallel, so there is no speed result.
    Mirrored means each device holds a full copy of the weights, takes a slice of the batch, and
    the gradients are summed across devices before the update. In exact arithmetic the result
    does not depend on the device count. The difference here is zero for a simpler reason: one
    replica runs the same computation as no strategy at all.
