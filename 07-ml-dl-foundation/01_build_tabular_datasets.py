"""Generate the four tabular datasets the rest of this module trains on, from known ground truth.

Demonstrates how to build a dataset whose every claim can later be checked:
    1. Draw vehicle listings from an explicit price formula and keep the formula.
    2. Write them space separated, leaving a missing value as nothing at all.
    3. Draw an attrition table from an explicit log-odds model, including two dead columns.
    4. Draw a speaker table whose classes are almost separable, as a contrast case.
    5. Draw a small property table for the from-scratch network to fit.
    6. Print the coefficients that produced each file, so later scripts can be scored against them.

Module 07: Machine Learning and Deep Learning Foundations - Dataset Construction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
SEED = 20260824

VEHICLE_TRAIN_ROWS = 30000
VEHICLE_HOLDOUT_ROWS = 10000
ATTRITION_ROWS = 1800
SPEAKER_ROWS = 3200
# The share of employees who leave. The intercept of the log-odds model is
# solved for this number rather than guessed, so the class balance is a
# parameter of the file and not an accident of the coefficients.
TARGET_POSITIVE_RATE = 0.16
PROPERTY_ROWS = 506

# Categorical fields that are allowed to go missing in the vehicle file. These
# are the three that carry the separator trap in script 02: a missing value is
# written as nothing between two spaces, so a reader that collapses runs of
# whitespace shifts the whole rest of the row one column to the left.
VEHICLE_NULLABLE = ("body_type", "fuel_type", "gearbox")
MISSING_RATE = 0.022

BRAND_TIERS = {
    0: 34000.0, 1: 26500.0, 2: 21000.0, 3: 18000.0, 4: 15500.0,
    5: 13000.0, 6: 11000.0, 7: 9500.0, 8: 8000.0, 9: 6500.0,
}

# The price formula. Every number here is recovered or contradicted by a later
# script, so it is stated once and printed at the end of the run.
AGE_HALF_LIFE_YEARS = 6.0
POWER_COEFFICIENT = 41.0
ODOMETER_COEFFICIENT = -430.0
DAMAGE_PENALTY = 0.82
GEARBOX_AUTOMATIC_BONUS = 1900.0
LATENT_WEIGHTS = (2600.0, -1800.0, 1200.0, 900.0, -650.0)
NOISE_SIGMA = 900.0


def _write_space_separated(frame, path):
    """Write a frame with single spaces and empty fields for missing values.

    This is the layout the vehicle file ships in. It matters that a missing
    value becomes an empty field rather than a placeholder: the row still has
    the right number of separators, so the file looks intact to any reader that
    treats one space as one separator, and silently loses a column to any
    reader that treats a run of whitespace as one separator.
    """
    columns = list(frame.columns)
    lines = [" ".join(columns)]
    for row in frame.itertuples(index=False, name=None):
        fields = []
        for value in row:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                fields.append("")
            elif isinstance(value, float):
                fields.append(f"{value:.4f}")
            else:
                fields.append(str(value))
        lines.append(" ".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_vehicles(rng, rows, first_id):
    """Draw vehicle listings and price them with the formula above."""
    listing_id = np.arange(first_id, first_id + rows)

    reg_year = rng.integers(2005, 2020, rows)
    reg_month = rng.integers(1, 13, rows)
    reg_day = rng.integers(1, 29, rows)
    list_year = np.clip(reg_year + rng.integers(0, 9, rows), 2005, 2020)
    list_month = rng.integers(1, 13, rows)
    list_day = rng.integers(1, 29, rows)

    reg_date = reg_year * 10000 + reg_month * 100 + reg_day
    list_date = list_year * 10000 + list_month * 100 + list_day
    age_years = np.maximum((list_date - reg_date) / 10000.0, 0.0)

    brand = rng.integers(0, 10, rows)
    model_code = rng.integers(0, 120, rows)
    body_type = rng.integers(0, 8, rows).astype(float)
    fuel_type = rng.integers(0, 7, rows).astype(float)
    gearbox = rng.integers(0, 2, rows).astype(float)
    damage_flag = rng.integers(0, 2, rows)
    region_code = rng.integers(1, 4000, rows)
    seller = (rng.random(rows) < 0.002).astype(int)
    offer_type = np.zeros(rows, dtype=int)

    power = np.clip(rng.normal(120, 45, rows), 0, None).round().astype(int)
    # A handful of listings carry an implausible power reading. Script 03 asks
    # whether flagging them changes anything.
    outlier_idx = rng.choice(rows, size=max(1, rows // 200), replace=False)
    power[outlier_idx] = rng.integers(600, 20000, len(outlier_idx))
    odometer_km = np.clip(rng.normal(9.0, 3.4, rows), 0.05, 15.0).round(1)

    latent = rng.normal(0, 1, (rows, 15))

    depreciation = 0.5 ** (age_years / AGE_HALF_LIFE_YEARS)
    base = np.array([BRAND_TIERS[int(b)] for b in brand])
    price = base * depreciation
    price = price + POWER_COEFFICIENT * np.clip(power, 0, 400)
    price = price + ODOMETER_COEFFICIENT * odometer_km
    price = price + GEARBOX_AUTOMATIC_BONUS * gearbox
    price = price * np.where(damage_flag == 0, DAMAGE_PENALTY, 1.0)
    for i, weight in enumerate(LATENT_WEIGHTS):
        price = price + weight * latent[:, i]
    price = price + rng.normal(0, NOISE_SIGMA, rows)
    price = np.clip(price, 250, None).round(0).astype(int)

    frame = pd.DataFrame({
        "listing_id": listing_id,
        "reg_date": reg_date,
        "list_date": list_date,
        "brand": brand,
        "model_code": model_code,
        "body_type": body_type,
        "fuel_type": fuel_type,
        "gearbox": gearbox,
        "power": power,
        "odometer_km": odometer_km,
        "damage_flag": damage_flag,
        "region_code": region_code,
        "seller": seller,
        "offer_type": offer_type,
    })
    for i in range(15):
        frame[f"v_{i}"] = latent[:, i].round(4)
    frame["price"] = price

    # Knock holes in the three nullable categorical columns.
    for column in VEHICLE_NULLABLE:
        mask = rng.random(rows) < MISSING_RATE
        frame.loc[mask, column] = np.nan

    rows_with_a_hole = int(frame[list(VEHICLE_NULLABLE)].isna().any(axis=1).sum())
    return frame, rows_with_a_hole


def make_attrition(rng, rows):
    """Draw an HR table whose attrition probability follows a stated log-odds model."""
    age = rng.integers(19, 60, rows)
    business_travel = rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                                 rows, p=[0.10, 0.71, 0.19])
    department = rng.choice(["Research & Development", "Sales", "Human Resources"],
                            rows, p=[0.65, 0.30, 0.05])
    distance_from_home = rng.integers(1, 30, rows)
    education = rng.integers(1, 6, rows)
    education_field = rng.choice(
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"],
        rows, p=[0.41, 0.32, 0.11, 0.10, 0.06])
    gender = rng.choice(["Female", "Male"], rows, p=[0.4, 0.6])
    job_level = rng.integers(1, 6, rows)
    job_role = rng.choice(
        ["Sales Executive", "Research Scientist", "Laboratory Technician",
         "Manufacturing Director", "Manager", "Sales Representative"],
        rows, p=[0.22, 0.20, 0.18, 0.14, 0.13, 0.13])
    marital_status = rng.choice(["Single", "Married", "Divorced"], rows, p=[0.32, 0.46, 0.22])
    over_time = rng.choice(["No", "Yes"], rows, p=[0.72, 0.28])

    monthly_income = (2100 + 2450 * job_level + rng.normal(0, 900, rows)).round().astype(int)
    monthly_income = np.clip(monthly_income, 1009, None)
    daily_rate = rng.integers(102, 1500, rows)
    hourly_rate = rng.integers(30, 101, rows)
    total_working_years = np.clip(age - 20 + rng.integers(-3, 6, rows), 0, None)
    years_at_company = np.clip(total_working_years - rng.integers(0, 12, rows), 0, None)
    years_in_current_role = np.clip(years_at_company - rng.integers(0, 6, rows), 0, None)
    years_since_promotion = np.clip(years_at_company - rng.integers(0, 9, rows), 0, None)
    years_with_manager = np.clip(years_in_current_role - rng.integers(0, 4, rows), 0, None)

    job_involvement = rng.integers(1, 5, rows)
    job_satisfaction = rng.integers(1, 5, rows)
    relationship_satisfaction = rng.integers(1, 5, rows)
    work_life_balance = rng.integers(1, 5, rows)
    environment_satisfaction = rng.integers(1, 5, rows)
    stock_option_level = rng.integers(0, 4, rows)
    num_companies_worked = rng.integers(0, 10, rows)
    percent_salary_hike = rng.integers(11, 26, rows)
    performance_rating = np.where(percent_salary_hike > 20, 4, 3)
    training_times = rng.integers(0, 7, rows)

    # The generative model. Overtime and being single dominate; income and
    # tenure pull the other way. Script 05 asks which library recovers this.
    linear = (1.25 * (over_time == "Yes")
             + 0.85 * (marital_status == "Single")
             + 0.55 * (business_travel == "Travel_Frequently")
             - 0.085 * years_at_company
             - 0.000105 * monthly_income
             - 0.24 * job_satisfaction
             - 0.20 * job_involvement
             - 0.17 * work_life_balance
             + 0.030 * distance_from_home
             - 0.019 * age
             + 0.11 * num_companies_worked
             - 0.22 * stock_option_level
             + rng.normal(0, 0.32, rows))

    # Solve the intercept by bisection so that the expected positive rate lands
    # on TARGET_POSITIVE_RATE. Guessing an intercept instead produced a 1.1%
    # positive class, because every centred term above pulls the log-odds down.
    low, high = -20.0, 20.0
    for _ in range(80):
        intercept = (low + high) / 2.0
        rate = float((1.0 / (1.0 + np.exp(-(intercept + linear)))).mean())
        if rate < TARGET_POSITIVE_RATE:
            low = intercept
        else:
            high = intercept
    intercept = (low + high) / 2.0

    probability = 1.0 / (1.0 + np.exp(-(intercept + linear)))
    attrition = np.where(rng.random(rows) < probability, "Yes", "No")

    frame = pd.DataFrame({
        "employee_id": np.arange(1, rows + 1),
        "Age": age,
        "Attrition": attrition,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        # Two columns that never vary. They are kept on purpose: script 05
        # measures what a model does with a feature that carries no signal.
        "EmployeeCount": np.ones(rows, dtype=int),
        "StandardHours": np.full(rows, 80, dtype=int),
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": over_time,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_promotion,
        "YearsWithCurrManager": years_with_manager,
    })
    return frame, float((attrition == "Yes").mean()), intercept


def make_speaker_acoustics(rng, rows):
    """Draw a near-separable two-class table as the easy contrast to attrition."""
    labels = rng.choice(["female", "male"], rows, p=[0.5, 0.5])
    is_male = (labels == "male").astype(float)

    # The mean fundamental frequency alone almost separates the two classes.
    mean_fundamental = np.where(is_male == 1, rng.normal(0.116, 0.014, rows),
                                rng.normal(0.171, 0.017, rows))
    columns = {
        "mean_freq": rng.normal(0.181, 0.030, rows) - 0.011 * is_male,
        "sd": rng.normal(0.057, 0.017, rows) + 0.004 * is_male,
        "median_freq": rng.normal(0.186, 0.036, rows) - 0.013 * is_male,
        "q25": rng.normal(0.140, 0.049, rows) - 0.020 * is_male,
        "q75": rng.normal(0.225, 0.024, rows) - 0.007 * is_male,
        "iqr": rng.normal(0.084, 0.043, rows) + 0.014 * is_male,
        "skew": np.abs(rng.normal(3.14, 4.24, rows)),
        "kurtosis": np.abs(rng.normal(36.6, 134.9, rows)),
        "spectral_entropy": rng.normal(0.895, 0.045, rows) + 0.009 * is_male,
        "spectral_flatness": rng.normal(0.408, 0.177, rows) + 0.021 * is_male,
        "mode_freq": rng.normal(0.165, 0.077, rows) - 0.010 * is_male,
        "centroid": rng.normal(0.181, 0.030, rows) - 0.011 * is_male,
        "mean_fundamental": mean_fundamental,
        "min_fundamental": rng.normal(0.037, 0.019, rows) - 0.003 * is_male,
        "max_fundamental": rng.normal(0.259, 0.030, rows) - 0.004 * is_male,
        "mean_dominant": rng.normal(0.829, 0.525, rows) - 0.060 * is_male,
        "min_dominant": np.abs(rng.normal(0.052, 0.063, rows)),
        "max_dominant": rng.normal(5.05, 3.52, rows) - 0.24 * is_male,
        "dominant_range": rng.normal(4.99, 3.52, rows) - 0.24 * is_male,
        "modulation_index": np.clip(rng.normal(0.174, 0.119, rows), 0, 1),
    }
    frame = pd.DataFrame(columns)
    frame["label"] = labels
    return frame


def make_property_valuation(rng, rows):
    """Draw a small dense regression table for the from-scratch network.

    Twelve numeric predictors on deliberately different scales, so that script
    07 has something to say about normalisation, and a target built from a
    linear part plus one saturating term.
    """
    rooms = np.clip(rng.normal(6.3, 0.7, rows), 3.5, 8.8)
    build_year = rng.integers(1930, 2016, rows)
    lot_size = np.clip(rng.normal(9500, 4200, rows), 1200, None)
    floor_area = np.clip(rng.normal(1750, 520, rows), 500, None)
    distance_to_centre = np.clip(rng.normal(9.2, 5.1, rows), 0.4, None)
    transit_index = np.clip(rng.normal(5.5, 2.4, rows), 1, 10)
    school_rating = np.clip(rng.normal(6.4, 1.9, rows), 1, 10)
    crime_index = np.clip(rng.normal(3.6, 3.1, rows), 0.05, None)
    tax_rate = rng.integers(187, 712, rows)
    pupil_teacher_ratio = np.clip(rng.normal(18.5, 2.2, rows), 12, 23)
    noise_level = np.clip(rng.normal(0.55, 0.12, rows), 0.2, 0.9)
    vacancy_rate = np.clip(rng.normal(12.6, 7.1, rows), 1.0, None)

    target = (11.4
              + 4.9 * rooms
              + 0.021 * (build_year - 1930)
              + 0.00021 * lot_size
              + 0.0043 * floor_area
              - 0.42 * distance_to_centre
              + 0.38 * transit_index
              + 0.61 * school_rating
              - 0.55 * crime_index
              - 0.0072 * tax_rate
              - 0.79 * pupil_teacher_ratio
              # One saturating term, so a linear model cannot be perfect.
              - 9.5 * np.tanh((vacancy_rate - 12.6) / 6.0)
              - 6.1 * noise_level
              + rng.normal(0, 2.2, rows))
    target = np.clip(target, 5.0, 50.0)

    return pd.DataFrame({
        "rooms": rooms.round(3),
        "build_year": build_year,
        "lot_size": lot_size.round(1),
        "floor_area": floor_area.round(1),
        "distance_to_centre": distance_to_centre.round(3),
        "transit_index": transit_index.round(2),
        "school_rating": school_rating.round(2),
        "crime_index": crime_index.round(4),
        "tax_rate": tax_rate,
        "pupil_teacher_ratio": pupil_teacher_ratio.round(1),
        "noise_level": noise_level.round(3),
        "vacancy_rate": vacancy_rate.round(2),
        "market_value": target.round(2),
    })


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("--- 1. Vehicle listings, priced from a known formula ---")
    train, train_holes = make_vehicles(rng, VEHICLE_TRAIN_ROWS, 1)
    holdout, holdout_holes = make_vehicles(rng, VEHICLE_HOLDOUT_ROWS, 500001)
    print(f"train   {train.shape[0]} rows x {train.shape[1]} columns, "
          f"{train_holes} rows carry at least one empty field "
          f"({train_holes / len(train):.2%})")
    print(f"holdout {holdout.shape[0]} rows x {holdout.shape[1]} columns, "
          f"{holdout_holes} rows carry at least one empty field")
    print(f"price   min {train['price'].min()}, median {int(train['price'].median())}, "
          f"max {train['price'].max()}")

    print("\n--- 2. Written space separated, a missing value written as nothing ---")
    _write_space_separated(train, DATA / "vehicle_listings.csv")
    _write_space_separated(holdout, DATA / "vehicle_holdout.csv")
    sample = (DATA / "vehicle_listings.csv").read_text(encoding="utf-8").splitlines()
    first_hole = next(i for i, line in enumerate(sample) if "  " in line)
    line = sample[first_hole]
    print(f"data row {first_hole} carries an empty field. Counting its columns two ways:")
    print(f"    split on one space          {len(line.split(' '))} fields  (correct)")
    print(f"    split on runs of whitespace {len(line.split())} fields  (one column short)")
    print(f"    the row reads: {' '.join(line.split(' ')[:9])}")
    print("gearbox really has exactly "
          f"{int(train['gearbox'].nunique())} distinct values "
          "- script 02 uses that number as its check")

    print("\n--- 3. Employee attrition, drawn from a stated log-odds model ---")
    attrition, positive_rate, intercept = make_attrition(rng, ATTRITION_ROWS)
    attrition.to_csv(DATA / "employee_attrition.csv", index=False)
    print(f"{attrition.shape[0]} rows x {attrition.shape[1]} columns, "
          f"positive class {positive_rate:.1%} "
          f"(intercept solved to {intercept:.4f} for a {TARGET_POSITIVE_RATE:.0%} target)")
    print("EmployeeCount and StandardHours are constant on purpose: "
          f"{attrition['EmployeeCount'].nunique()} and "
          f"{attrition['StandardHours'].nunique()} distinct values")

    print("\n--- 4. Speaker acoustics, the near-separable contrast case ---")
    speaker = make_speaker_acoustics(rng, SPEAKER_ROWS)
    speaker.to_csv(DATA / "speaker_acoustics.csv", index=False)
    female = speaker.loc[speaker["label"] == "female", "mean_fundamental"].mean()
    male = speaker.loc[speaker["label"] == "male", "mean_fundamental"].mean()
    print(f"{speaker.shape[0]} rows x {speaker.shape[1]} columns, classes balanced")
    print(f"mean_fundamental separates them on its own: "
          f"female {female:.4f} against male {male:.4f}")

    print("\n--- 5. Property valuation, small and dense for the hand-written network ---")
    property_frame = make_property_valuation(rng, PROPERTY_ROWS)
    property_frame.to_csv(DATA / "property_valuation.csv", index=False)
    print(f"{property_frame.shape[0]} rows x {property_frame.shape[1]} columns")
    print("feature scales differ by four orders of magnitude: "
          f"noise_level around {property_frame['noise_level'].mean():.2f}, "
          f"lot_size around {property_frame['lot_size'].mean():.0f}")

    print("\n--- 6. The coefficients that produced these files ---")
    print("vehicle price:")
    print(f"    brand tier base        {min(BRAND_TIERS.values()):.0f} to "
          f"{max(BRAND_TIERS.values()):.0f}")
    print(f"    value halves every     {AGE_HALF_LIFE_YEARS} years")
    print(f"    power                  {POWER_COEFFICIENT:+.0f} per unit, capped at 400")
    print(f"    odometer_km            {ODOMETER_COEFFICIENT:+.0f} per unit")
    print(f"    automatic gearbox      {GEARBOX_AUTOMATIC_BONUS:+.0f}")
    print(f"    damage_flag == 0       x{DAMAGE_PENALTY}")
    print(f"    v_0 to v_4 weights     {LATENT_WEIGHTS}")
    print(f"    v_5 to v_14            no effect at all, they are noise columns")
    print(f"    residual noise sigma   {NOISE_SIGMA:.0f}")
    print("attrition log-odds: OverTime +1.25, Single +0.85, Travel_Frequently +0.55,")
    print("    YearsAtCompany -0.085, JobSatisfaction -0.24, StockOptionLevel -0.22")
    print("property value: linear in eleven features, saturating in vacancy_rate")

    print(f"\nAll five files written to {DATA}")
    print("Rerunning this script reproduces them byte for byte.")


if __name__ == "__main__":
    main()
