import pandas as pd
import numpy as np

# Set the URL for the dataset
DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"

# Columns to use for the analysis
COLUMNS = [
    'engine_displacement',
    'horsepower',
    'vehicle_weight',
    'model_year',
    'fuel_efficiency_mpg'
]

# Rename columns to match the dataset for clarity
RENAME_MAP = {
    'mpg': 'fuel_efficiency_mpg',
    'displacement': 'engine_displacement',
    'weight': 'vehicle_weight',
    'model year': 'model_year',
}

# --- 1. Load Data and Initial Preprocessing ---
print("--- Data Loading & Preprocessing ---")
df = pd.read_csv(DATA_URL)
print(f"Original shape: {df.shape}")

# Rename and select columns
df.columns = df.columns.str.lower().str.replace('-', '_').str.replace(' ', '_')
df = df.rename(columns=RENAME_MAP)
df = df[COLUMNS]

# Convert 'horsepower' to numeric, handling missing values ('?')
# The standard Auto MPG dataset uses '?' for NAs, which is why we force numeric conversion.
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Target variable transformation
df['fuel_efficiency_mpg'] = np.log1p(df['fuel_efficiency_mpg'])

# --- EDA ---
# Q1: Missing values (The dataset columns are: engine_displacement, horsepower, vehicle_weight, model_year, fuel_efficiency_mpg)
missing_values = df.isnull().sum()
q1_missing_col = missing_values[missing_values > 0].index[0]
print(f"Q1. Column with missing values: {q1_missing_col}")

# Q2: Median Horsepower
q2_median_hp = df['horsepower'].median()
print(f"Q2. Median horsepower (raw): {q2_median_hp}")


# --- 2. Model Functions ---

def train_linear_regression(X, y):
    """Trains linear regression without regularization (r=0)"""
    # Prepare matrix X with bias term (column of 1s)
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    # Normal Equation: w = (X^T * X)^-1 * (X^T * y)
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]  # w0 (intercept), w (weights)

def train_linear_regression_reg(X, y, r=0.0):
    """Trains regularized linear regression (Ridge)"""
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    # Normal Equation with regularization: w = (X^T * X + r*I)^-1 * (X^T * y)
    XTX = X.T.dot(X)
    # Identity matrix (I) with regularization term (r)
    reg = r * np.eye(XTX.shape[0])
    XTX_reg = XTX + reg
    
    XTX_inv = np.linalg.inv(XTX_reg)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]

def rmse(y, y_pred):
    """Calculates Root Mean Squared Error"""
    error = y - y_pred
    mse = (error ** 2).mean()
    return np.sqrt(mse)

def prepare_X(df, fill_value=0):
    """Prepares the feature matrix (X) and fills NAs"""
    df = df.copy()
    features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
    X = df[features].fillna(fill_value).values
    return X


# --- 3. Prepare and Split the Dataset (Helper Function) ---

def split_data(df, seed):
    """Shuffles and splits the data into train, val, test (60/20/20)"""
    n = len(df)
    n_test = int(0.2 * n)
    n_val = int(0.2 * n)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx]

    df_train = df_shuffled[:n_train].copy()
    df_val = df_shuffled[n_train:n_train + n_val].copy()
    df_test = df_shuffled[n_train + n_val:].copy()

    y_train = df_train['fuel_efficiency_mpg'].values
    y_val = df_val['fuel_efficiency_mpg'].values
    y_test = df_test['fuel_efficiency_mpg'].values
    
    return df_train, df_val, df_test, y_train, y_val, y_test


# --- Q3: Missing Value Imputation Comparison (Seed=42) ---
print("\n--- Q3: Imputation Comparison (Seed=42) ---")
df_train_42, df_val_42, _, y_train_42, y_val_42, _ = split_data(df, seed=42)
mean_hp_train = df_train_42['horsepower'].mean()

# Option 1: Fill with 0
X_train_0 = prepare_X(df_train_42, fill_value=0)
X_val_0 = prepare_X(df_val_42, fill_value=0)
w0_0, w_0 = train_linear_regression(X_train_0, y_train_42)
y_pred_0 = w0_0 + X_val_0.dot(w_0)
rmse_0 = rmse(y_val_42, y_pred_0)
print(f"RMSE (Fill with 0): {rmse_0:.4f} -> {round(rmse_0, 2)}")

# Option 2: Fill with Mean
X_train_mean = prepare_X(df_train_42, fill_value=mean_hp_train)
X_val_mean = prepare_X(df_val_42, fill_value=mean_hp_train)
w0_mean, w_mean = train_linear_regression(X_train_mean, y_train_42)
y_pred_mean = w0_mean + X_val_mean.dot(w_mean)
rmse_mean = rmse(y_val_42, y_pred_mean)
print(f"RMSE (Fill with Mean): {rmse_mean:.4f} -> {round(rmse_mean, 2)}")


# --- Q4: Regularization (Fill with 0, Seed=42) ---
print("\n--- Q4: Regularization Check (r values) ---")
r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
best_rmse = float('inf')
best_r = -1

# Use the X_train_0 and X_val_0 from Q3 (NAs filled with 0)
for r in r_values:
    w0, w = train_linear_regression_reg(X_train_0, y_train_42, r=r)
    y_pred = w0 + X_val_0.dot(w)
    score = rmse(y_val_42, y_pred)
    rounded_score = round(score, 2)
    print(f"r={r:<4}: RMSE = {score:.4f} -> {rounded_score}")
    
    # Check for best RMSE (and smallest r in case of a tie)
    if rounded_score < best_rmse:
        best_rmse = rounded_score
        best_r = r
    elif rounded_score == best_rmse and r < best_r:
        best_r = r

print(f"Best r (smallest in case of tie): {best_r} with RMSE {best_rmse}")


# --- Q5: Seed Influence (NAs with 0, r=0) ---
print("\n--- Q5: Seed Influence (Std Dev) ---")
seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_scores = []

for seed in seed_values:
    df_train, df_val, _, y_train, y_val, _ = split_data(df, seed=seed)
    
    # Fill NAs with 0
    X_train = prepare_X(df_train, fill_value=0)
    X_val = prepare_X(df_val, fill_value=0)
    
    # Train Linear Regression (r=0)
    w0, w = train_linear_regression(X_train, y_train)
    
    # Evaluate on Validation
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    rmse_scores.append(score)
    print(f"Seed {seed}: RMSE = {score:.4f}")

std_rmse = np.std(rmse_scores)
print(f"RMSE Scores: {rmse_scores}")
print(f"Standard Deviation: {std_rmse:.5f} -> {round(std_rmse, 3)}")


# --- Q6: Final Model (Seed 9, Train+Val, r=0.001) ---
print("\n--- Q6: Final Model on Test Set (Seed=9) ---")
SEED_9 = 9
R_FINAL = 0.001

df_train_9, df_val_9, df_test_9, y_train_9, y_val_9, y_test_9 = split_data(df, seed=SEED_9)

# 1. Combine Train and Validation
df_full_train = pd.concat([df_train_9, df_val_9])
y_full_train = np.concatenate([y_train_9, y_val_9])

# 2. Prepare X (Fill NAs with 0)
X_full_train = prepare_X(df_full_train, fill_value=0)
X_test = prepare_X(df_test_9, fill_value=0)

# 3. Train Regularized Model (r=0.001)
w0_final, w_final = train_linear_regression_reg(X_full_train, y_full_train, r=R_FINAL)

# 4. Evaluate on Test
y_pred_test = w0_final + X_test.dot(w_final)
rmse_test = rmse(y_test_9, y_pred_test)

print(f"Test RMSE (r={R_FINAL}): {rmse_test:.3f}")

# --- Summary Answers ---
print("\n--- Final Answers ---")
print(f"Q1: Column with missing values: {q1_missing_col}")
print(f"Q2: Median 'horsepower' (closest to 99): {q2_median_hp}")
print(f"Q3: Better RMSE: Fill with **mean** ({round(rmse_mean, 2)} vs {round(rmse_0, 2)})")
print(f"Q4: Best r: **{best_r}** (RMSE: {best_rmse})")
print(f"Q5: Std Dev: {round(std_rmse, 3)}")
print(f"Q6: Test RMSE: {round(rmse_test, 2)}")