import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LEAD SCORING HOMEWORK - COMPLETE SOLUTION")
print("Random State: 42")
print("="*80)

# Load the data
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('course_lead_scoring.csv')

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nTarget variable 'converted' distribution:")
print(df['converted'].value_counts())
print(f"Conversion rate: {df['converted'].mean():.1%}")

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("\n" + "="*80)
print("DATA PREPARATION")
print("="*80)

# Check for missing values
print("\n[STEP 2] Checking missing values...")
missing = df.isnull().sum()
print(f"Total missing values: {missing.sum()}")
for col in df.columns:
    if missing[col] > 0:
        print(f"  {col}: {missing[col]} missing ({missing[col]/len(df)*100:.1f}%)")

# Identify categorical and numerical columns
print("\n[STEP 3] Identifying column types...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from numerical columns
if 'converted' in numerical_cols:
    numerical_cols.remove('converted')

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Fill missing values as per instructions
print("\n[STEP 4] Filling missing values...")
print("  - Categorical features: filling with 'NA'")
print("  - Numerical features: filling with 0.0")

for col in categorical_cols:
    df[col] = df[col].fillna('NA')

for col in numerical_cols:
    df[col] = df[col].fillna(0.0)

print(f"After filling, missing values: {df.isnull().sum().sum()}")

# Split the data with random_state=42
print("\n[STEP 5] Splitting data: 60% train / 20% validation / 20% test")
print("  Using train_test_split with random_state=42")

# First split: 80% (full_train) and 20% (test)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Second split: split full_train into 75% (train) and 25% (val)
# This gives us 60% train and 20% val of the original dataset
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

print(f"\nDataset sizes:")
print(f"  Total:      {len(df):5d} (100.0%)")
print(f"  Train:      {len(df_train):5d} ({len(df_train)/len(df)*100:5.1f}%)")
print(f"  Validation: {len(df_val):5d} ({len(df_val)/len(df)*100:5.1f}%)")
print(f"  Test:       {len(df_test):5d} ({len(df_test)/len(df)*100:5.1f}%)")
print(f"  Full Train: {len(df_full_train):5d} ({len(df_full_train)/len(df)*100:5.1f}%) [used for CV]")

# ============================================================================
# QUESTION 1: ROC AUC Feature Importance
# ============================================================================
print("\n" + "="*80)
print("QUESTION 1: ROC AUC Feature Importance")
print("="*80)

print("\nTask: Evaluate 4 numerical variables using them as prediction scores")
print("  - For each variable, compute AUC with 'converted' as ground truth")
print("  - If AUC < 0.5, invert the variable and recompute")
print("  - Find which has the highest AUC")

features_to_evaluate = ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']

y_train = df_train['converted'].values

print(f"\nUsing TRAINING SET (n={len(df_train)}):")
print("-"*80)

auc_results = {}

for feature in features_to_evaluate:
    if feature not in df_train.columns:
        print(f"{feature}: NOT FOUND IN DATASET")
        continue
    
    # Use feature values as prediction scores
    scores = df_train[feature].values
    
    # Compute AUC
    auc = roc_auc_score(y_train, scores)
    
    # If AUC < 0.5, invert
    if auc < 0.5:
        auc_inverted = roc_auc_score(y_train, -scores)
        auc_results[feature] = auc_inverted
        print(f"{feature:30s}: {auc:.4f} -> {auc_inverted:.4f} (inverted)")
    else:
        auc_results[feature] = auc
        print(f"{feature:30s}: {auc:.4f}")

# Find best feature
best_feature = max(auc_results, key=auc_results.get)
best_auc = auc_results[best_feature]

print("-"*80)
print(f"\n✓ ANSWER Q1: {best_feature}")
print(f"  (AUC = {best_auc:.4f})")
print("="*80)

# ============================================================================
# QUESTION 2: Training the Model
# ============================================================================
print("\n" + "="*80)
print("QUESTION 2: Training Logistic Regression Model")
print("="*80)

print("\nTask: Train logistic regression and evaluate on VALIDATION set")
print("  - Use DictVectorizer for one-hot encoding")
print("  - LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)")
print("  - Report AUC on validation set (rounded to 3 digits)")

# Prepare features (all except target)
feature_cols = [col for col in df_train.columns if col != 'converted']

print(f"\nFeatures: {len(feature_cols)} columns")

# Convert to dictionaries
train_dicts = df_train[feature_cols].to_dict(orient='records')
val_dicts = df_val[feature_cols].to_dict(orient='records')

# Apply DictVectorizer
print("\nApplying DictVectorizer (one-hot encoding)...")
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

y_train = df_train['converted'].values
y_val = df_val['converted'].values

print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape:   {X_val.shape}")

# Train model
print("\nTraining LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)...")
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on validation
y_val_pred = model.predict_proba(X_val)[:, 1]

# Compute AUC
auc_val = roc_auc_score(y_val, y_val_pred)

print(f"\nValidation AUC: {auc_val:.6f}")
print(f"Rounded to 3 digits: {auc_val:.3f}")

# Match to options
print("\nMultiple choice options:")
options_q2 = [0.32, 0.52, 0.72, 0.92]
closest_q2 = min(options_q2, key=lambda x: abs(x - auc_val))
for opt in options_q2:
    marker = " ← CLOSEST MATCH" if opt == closest_q2 else ""
    print(f"  {opt}{marker}")

print(f"\n✓ ANSWER Q2: {round(auc_val, 3)}")
print("="*80)

# ============================================================================
# QUESTION 3: Precision and Recall Intersection
# ============================================================================
print("\n" + "="*80)
print("QUESTION 3: Precision-Recall Intersection")
print("="*80)

print("\nTask: Find threshold where precision and recall curves intersect")
print("  - Evaluate thresholds from 0.0 to 1.0 with step 0.01")

thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []

for threshold in thresholds:
    y_pred_binary = (y_val_pred >= threshold).astype(int)
    
    # Calculate TP, FP, FN
    tp = np.sum((y_pred_binary == 1) & (y_val == 1))
    fp = np.sum((y_pred_binary == 1) & (y_val == 0))
    fn = np.sum((y_pred_binary == 0) & (y_val == 1))
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    precisions.append(precision)
    recalls.append(recall)

# Find intersection
differences = np.abs(np.array(precisions) - np.array(recalls))
intersection_idx = np.argmin(differences)
intersection_threshold = thresholds[intersection_idx]

print(f"\nIntersection point:")
print(f"  Threshold:  {intersection_threshold:.3f}")
print(f"  Precision:  {precisions[intersection_idx]:.3f}")
print(f"  Recall:     {recalls[intersection_idx]:.3f}")
print(f"  Difference: {differences[intersection_idx]:.6f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', linewidth=2)
plt.plot(thresholds, recalls, label='Recall', linewidth=2)
plt.axvline(x=intersection_threshold, color='red', linestyle='--', 
            label=f'Intersection at {intersection_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('q3_precision_recall.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: q3_precision_recall.png")

# Match to options
print("\nMultiple choice options:")
options_q3 = [0.145, 0.345, 0.545, 0.745]
closest_q3 = min(options_q3, key=lambda x: abs(x - intersection_threshold))
for opt in options_q3:
    marker = " ← CLOSEST MATCH" if opt == closest_q3 else ""
    print(f"  {opt}{marker}")

print(f"\n✓ ANSWER Q3: {intersection_threshold:.3f}")
print("="*80)

# ============================================================================
# QUESTION 4: Maximum F1 Score
# ============================================================================
print("\n" + "="*80)
print("QUESTION 4: Maximum F1 Score")
print("="*80)

print("\nTask: Find threshold that maximizes F1 score")
print("  - F1 = 2*P*R / (P+R)")

f1_scores = []

for i in range(len(thresholds)):
    p = precisions[i]
    r = recalls[i]
    
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    
    f1_scores.append(f1)

# Find maximum F1
max_f1_idx = np.argmax(f1_scores)
max_f1_threshold = thresholds[max_f1_idx]
max_f1_value = f1_scores[max_f1_idx]

print(f"\nMaximum F1 score:")
print(f"  Threshold:  {max_f1_threshold:.3f}")
print(f"  F1 Score:   {max_f1_value:.3f}")
print(f"  Precision:  {precisions[max_f1_idx]:.3f}")
print(f"  Recall:     {recalls[max_f1_idx]:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, linewidth=2, color='green')
plt.axvline(x=max_f1_threshold, color='red', linestyle='--',
            label=f'Max F1 at {max_f1_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('q4_f1_score.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: q4_f1_score.png")

# Match to options
print("\nMultiple choice options:")
options_q4 = [0.14, 0.34, 0.54, 0.74]
closest_q4 = min(options_q4, key=lambda x: abs(x - max_f1_threshold))
for opt in options_q4:
    marker = " ← CLOSEST MATCH" if opt == closest_q4 else ""
    print(f"  {opt}{marker}")

print(f"\n✓ ANSWER Q4: {max_f1_threshold:.2f}")
print("="*80)

# ============================================================================
# QUESTION 5: 5-Fold Cross-Validation
# ============================================================================
print("\n" + "="*80)
print("QUESTION 5: 5-Fold Cross-Validation")
print("="*80)

print("\nTask: Evaluate model with 5-fold CV on df_full_train")
print("  - KFold(n_splits=5, shuffle=True, random_state=42)")
print("  - Report standard deviation of AUC scores")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\nUsing df_full_train (n={len(df_full_train)})")
print("\nTraining 5 models...")

fold_aucs = []

for fold_num, (train_idx, val_idx) in enumerate(kfold.split(df_full_train), 1):
    # Get fold data
    df_fold_train = df_full_train.iloc[train_idx]
    df_fold_val = df_full_train.iloc[val_idx]
    
    # Prepare features
    fold_train_dicts = df_fold_train[feature_cols].to_dict(orient='records')
    fold_val_dicts = df_fold_val[feature_cols].to_dict(orient='records')
    
    # One-hot encode
    dv_fold = DictVectorizer(sparse=False)
    X_fold_train = dv_fold.fit_transform(fold_train_dicts)
    X_fold_val = dv_fold.transform(fold_val_dicts)
    
    y_fold_train = df_fold_train['converted'].values
    y_fold_val = df_fold_val['converted'].values
    
    # Train model
    model_fold = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_fold.fit(X_fold_train, y_fold_train)
    
    # Predict and evaluate
    y_fold_pred = model_fold.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, y_fold_pred)
    fold_aucs.append(fold_auc)
    
    print(f"  Fold {fold_num}: AUC = {fold_auc:.4f}")

# Calculate statistics
mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)

print(f"\nResults:")
print(f"  Mean AUC: {mean_auc:.4f}")
print(f"  Std AUC:  {std_auc:.4f}")

# Match to options
print("\nMultiple choice options:")
options_q5 = [0.0001, 0.006, 0.06, 0.36]
closest_q5 = min(options_q5, key=lambda x: abs(x - std_auc))
for opt in options_q5:
    marker = " ← CLOSEST MATCH" if opt == closest_q5 else ""
    print(f"  {opt}{marker}")

print(f"\n✓ ANSWER Q5: {std_auc:.4f}")
print("="*80)

# ============================================================================
# QUESTION 6: Hyperparameter Tuning
# ============================================================================
print("\n" + "="*80)
print("QUESTION 6: Hyperparameter Tuning")
print("="*80)

print("\nTask: Find best C value using 5-fold CV")
print("  - Test C values: [0.000001, 0.001, 1]")
print("  - Select best based on: highest mean, then lowest std, then smallest C")

C_values = [0.000001, 0.001, 1]
results = []

for C in C_values:
    print(f"\nTesting C = {C}")
    print("-"*40)
    
    fold_aucs_c = []
    
    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(df_full_train), 1):
        # Get fold data
        df_fold_train = df_full_train.iloc[train_idx]
        df_fold_val = df_full_train.iloc[val_idx]
        
        # Prepare features
        fold_train_dicts = df_fold_train[feature_cols].to_dict(orient='records')
        fold_val_dicts = df_fold_val[feature_cols].to_dict(orient='records')
        
        # One-hot encode
        dv_fold = DictVectorizer(sparse=False)
        X_fold_train = dv_fold.fit_transform(fold_train_dicts)
        X_fold_val = dv_fold.transform(fold_val_dicts)
        
        y_fold_train = df_fold_train['converted'].values
        y_fold_val = df_fold_val['converted'].values
        
        # Train model with C
        model_fold = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
        model_fold.fit(X_fold_train, y_fold_train)
        
        # Predict and evaluate
        y_fold_pred = model_fold.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, y_fold_pred)
        fold_aucs_c.append(fold_auc)
        
        print(f"  Fold {fold_num}: {fold_auc:.4f}")
    
    mean_score = np.mean(fold_aucs_c)
    std_score = np.std(fold_aucs_c)
    
    results.append({
        'C': C,
        'mean': round(mean_score, 3),
        'std': round(std_score, 3)
    })
    
    print(f"  Mean: {mean_score:.3f}")
    print(f"  Std:  {std_score:.3f}")

# Display results
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Find best C
# Sort by: mean (descending), std (ascending), C (ascending)
best_idx = results_df.sort_values(by=['mean', 'std', 'C'], 
                                   ascending=[False, True, True]).index[0]
best_C = results_df.loc[best_idx, 'C']

print(f"\n✓ ANSWER Q6: C = {best_C}")
print(f"  Mean: {results_df.loc[best_idx, 'mean']:.3f}")
print(f"  Std:  {results_df.loc[best_idx, 'std']:.3f}")
print("="*80)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL ANSWERS - ALL QUESTIONS")
print("="*80)

print(f"\nQ1: Which numerical variable has the highest AUC?")
print(f"    → {best_feature}")

print(f"\nQ2: AUC on validation dataset (rounded to 3 digits)?")
print(f"    → {round(auc_val, 3)}")

print(f"\nQ3: At which threshold do precision and recall intersect?")
print(f"    → {intersection_threshold:.3f}")

print(f"\nQ4: At which threshold is F1 maximal?")
print(f"    → {max_f1_threshold:.2f}")

print(f"\nQ5: Standard deviation of 5-fold CV scores?")
print(f"    → {std_auc:.4f}")

print(f"\nQ6: Which C leads to the best mean score?")
print(f"    → {best_C}")

print("\n" + "="*80)
print("HOMEWORK COMPLETE!")
print(f"User: {input('ANSON-WAN')}")
print(f"Date: 2025-10-20")
print("="*80)