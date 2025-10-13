import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# --- Configuration ---
RSEED = 42
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# --- Data Loading and Preparation ---
print("--- Data Loading and Preparation ---")
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv"
df = pd.read_csv(url)

# Clean column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Define feature types and target
numerical_features = ['annual_income', 'interaction_count', 'number_of_courses_viewed', 'lead_score']
categorical_features = ['industry', 'location', 'employment_status', 'lead_source']
target_variable = 'converted'

# Handle Missing Values
for col in numerical_features:
    df[col] = df[col].fillna(0.0)
for col in categorical_features:
    df[col] = df[col].fillna('NA')

# Convert target to integer type
df[target_variable] = df[target_variable].astype(int)

# --- Question 1: Mode of 'industry' ---
print("\n--- Question 1: Mode of 'industry' ---")
industry_mode = df['industry'].mode().iloc[0]
print(f"The mode for 'industry' is: '{industry_mode}'")
print(f"Q1 Answer: {industry_mode}")

# --- Question 2: Biggest Correlation ---
print("\n--- Question 2: Biggest Correlation ---")
correlation_matrix = df[numerical_features].corr()

# Extract specified correlations
relevant_pairs = {
    'interaction_count and lead_score': correlation_matrix.loc['interaction_count', 'lead_score'],
    'number_of_courses_viewed and lead_score': correlation_matrix.loc['number_of_courses_viewed', 'lead_score'],
    'number_of_courses_viewed and interaction_count': correlation_matrix.loc['number_of_courses_viewed', 'interaction_count'],
    'annual_income and interaction_count': correlation_matrix.loc['annual_income', 'interaction_count']
}

# Find the pair with the highest absolute correlation
highest_corr_pair = max(relevant_pairs.items(), key=lambda item: abs(item[1]))

print(f"Correlation coefficients for specified pairs (absolute):\n{pd.Series({k: abs(v) for k, v in relevant_pairs.items()}).sort_values(ascending=False).to_string()}")
print(f"The pair with the biggest absolute correlation is: {highest_corr_pair[0]} ({highest_corr_pair[1]:.4f})")
print(f"Q2 Answer: {highest_corr_pair[0]}")


# --- Split the Data (60/20/20 with Stratification) ---
y = df[target_variable].copy()
X = df.drop(columns=[target_variable]).copy()

# Split X and y into train_full (80%) and test (20%) - Stratified
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RSEED, stratify=y
)

# Split X_train_full (80%) into train (60%) and validation (20%) - Stratified
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RSEED, stratify=y_train_full # 0.25 * 0.8 = 0.2
)

print(f"\n--- Data Split Check (Stratified) ---")
print(f"Train size: {len(X_train)} ({len(X_train)/len(df):.0%})")
print(f"Validation size: {len(X_val)} ({len(X_val)/len(df):.0%})")
print(f"Test size: {len(X_test)} ({len(X_test)/len(df):.0%})")


# --- Question 3: Mutual Information ---
print("\n--- Question 3: Mutual Information ---")

def calculate_mi(series):
    # mutual_info_score expects two Series/arrays
    return mutual_info_score(series, y_train)

# Calculate MI only on categorical features in the training set
mi_scores = X_train[categorical_features].apply(calculate_mi).round(2)
print("Mutual Information Scores (rounded to 2 decimals) on the training set:")
print(mi_scores.to_string())

biggest_mi_feature = mi_scores.idxmax()
print(f"The variable with the biggest MI score is: {biggest_mi_feature}")
print(f"Q3 Answer: {biggest_mi_feature}")


# --- Preprocessing for Logistic Regression (Q4-Q6) ---
dv = DictVectorizer(sparse=False)
train_dict = X_train.to_dict(orient='records')
val_dict = X_val.to_dict(orient='records')

# Fit DictVectorizer on training data and transform all sets
X_train_encoded = dv.fit_transform(train_dict)
X_val_encoded = dv.transform(val_dict)
all_features = dv.get_feature_names_out()


# --- Question 4: Baseline Logistic Regression Accuracy ---
print("\n--- Question 4: Logistic Regression Accuracy (C=1.0) ---")
# Train the model with C=1.0 (baseline)
baseline_C = 1.0
model = LogisticRegression(
    solver='liblinear',
    C=baseline_C,
    max_iter=1000,
    random_state=RSEED
)
model.fit(X_train_encoded, y_train)

# Calculate accuracy on validation set
y_pred_val = model.predict(X_val_encoded)
original_accuracy = accuracy_score(y_val, y_pred_val)
accuracy_rounded_q4 = round(original_accuracy, 2)

print(f"Validation Accuracy: {original_accuracy:.4f}")
print(f"Validation Accuracy (rounded to 2 decimals): {accuracy_rounded_q4}")
print(f"Q4 Answer: {accuracy_rounded_q4}")


# --- Question 5: Feature Elimination ---
print("\n--- Question 5: Feature Elimination ---")
features_to_check = ['industry', 'employment_status', 'lead_score']
accuracy_diffs = {}

for feature_name in features_to_check:
    # Identify indices of columns to drop for the current feature
    cols_to_drop_indices = [
        i for i, col in enumerate(all_features)
        if col == feature_name or col.startswith(f"{feature_name}=")
    ]

    # Create datasets without the feature(s)
    X_train_subset = np.delete(X_train_encoded, cols_to_drop_indices, axis=1)
    X_val_subset = np.delete(X_val_encoded, cols_to_drop_indices, axis=1)

    # Train a new model
    model_subset = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=RSEED)
    model_subset.fit(X_train_subset, y_train)

    # Calculate accuracy and difference
    accuracy_subset = accuracy_score(y_val, model_subset.predict(X_val_subset))
    difference = original_accuracy - accuracy_subset # How much the accuracy DROPPED
    accuracy_diffs[feature_name] = difference

    print(f"Feature removed: {feature_name:<17} | Diff: {difference:.4f} (Original: {original_accuracy:.4f}, New: {accuracy_subset:.4f})")

# Find the feature with the smallest difference (absolute value closest to zero)
smallest_diff_feature = min(accuracy_diffs.items(), key=lambda item: abs(item[1]))[0]

print(f"\nThe feature with the smallest difference is: '{smallest_diff_feature}'")
print(f"Q5 Answer: '{smallest_diff_feature}'")


# --- Question 6: Regularized Logistic Regression (C values) ---
print("\n--- Question 6: Regularized Logistic Regression (C Tuning) ---")
C_values = [0.01, 0.1, 1, 10, 100]
best_accuracy = -1.0
best_C = None
results = {}

for C in C_values:
    model_reg = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=RSEED)
    model_reg.fit(X_train_encoded, y_train)

    accuracy_reg = accuracy_score(y_val, model_reg.predict(X_val_encoded))
    results[C] = accuracy_reg

    # Check for best C (and select smallest C in case of a tie)
    if accuracy_reg > best_accuracy:
        best_accuracy = accuracy_reg
        best_C = C
    elif accuracy_reg == best_accuracy:
        # If tied, keep the smaller C value
        if best_C is None or C < best_C:
             best_C = C

print("Validation Accuracies:")
for C, acc in results.items():
    print(f" C={C:<6} -> accuracy={acc:.4f}")

print(f"\nThe best C value (smallest in case of a tie) is: {best_C} (Acc: {best_accuracy:.4f})")
print(f"Q6 Answer: {best_C}")
