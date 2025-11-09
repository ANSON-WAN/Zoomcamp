import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Download and load the dataset
print("Downloading dataset...")
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

# Preparation: Fill missing values with zeros
df = df.fillna(0)

# Split the data: 60% train, 20% validation, 20% test
# First split: 60% train, 40% temp (validation + test)
df_train, df_temp = train_test_split(df, test_size=0.4, random_state=1)
# Second split: split temp into 50-50 for validation and test (20% each of total)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=1)

print(f"\nTrain size: {len(df_train)}")
print(f"Validation size: {len(df_val)}")
print(f"Test size: {len(df_test)}")

# Prepare target variable
y_train = df_train['fuel_efficiency_mpg'].values
y_val = df_val['fuel_efficiency_mpg'].values
y_test = df_test['fuel_efficiency_mpg'].values

# Drop target from features
df_train = df_train.drop('fuel_efficiency_mpg', axis=1)
df_val = df_val.drop('fuel_efficiency_mpg', axis=1)
df_test = df_test.drop('fuel_efficiency_mpg', axis=1)

# Convert to dictionaries for DictVectorizer
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

# Use DictVectorizer
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

print(f"\nFeature matrix shape: {X_train.shape}")

# ===== QUESTION 1 =====
print("\n" + "="*60)
print("QUESTION 1: Decision Tree with max_depth=1")
print("="*60)

dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)

# Get the feature used for splitting
feature_names = dv.get_feature_names_out()
tree = dt.tree_
feature_idx = tree.feature[0]  # Root node feature
split_feature = feature_names[feature_idx]

print(f"Feature used for splitting: {split_feature}")
print(f"Answer: {split_feature}")

# ===== QUESTION 2 =====
print("\n" + "="*60)
print("QUESTION 2: Random Forest with n_estimators=10")
print("="*60)

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"RMSE on validation data: {rmse:.3f}")
print(f"Answer: {rmse:.1f}")

# ===== QUESTION 3 =====
print("\n" + "="*60)
print("QUESTION 3: Finding when RMSE stops improving")
print("="*60)

n_estimators_range = range(10, 201, 10)
rmse_scores = []

for n in n_estimators_range:
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    print(f"n_estimators={n:3d}, RMSE={rmse:.3f}")

# Find when RMSE stops improving (to 3 decimal places)
best_rmse = min(rmse_scores)
best_n = list(n_estimators_range)[rmse_scores.index(best_rmse)]

# Find first occurrence where RMSE reaches its minimum (rounded to 3 decimals)
rmse_rounded = [round(r, 3) for r in rmse_scores]
min_rmse_rounded = min(rmse_rounded)
stop_improving_idx = rmse_rounded.index(min_rmse_rounded)
stop_improving_n = list(n_estimators_range)[stop_improving_idx]

print(f"\nBest RMSE: {best_rmse:.3f} at n_estimators={best_n}")
print(f"RMSE stops improving after n_estimators={stop_improving_n}")
print(f"Answer: {stop_improving_n}")

# ===== QUESTION 4 =====
print("\n" + "="*60)
print("QUESTION 4: Best max_depth")
print("="*60)

max_depths = [10, 15, 20, 25]
mean_rmse_per_depth = {}

for max_depth in max_depths:
    print(f"\nTesting max_depth={max_depth}")
    rmse_list = []
    
    for n in range(10, 201, 10):
        rf = RandomForestRegressor(n_estimators=n, max_depth=max_depth, 
                                   random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_list.append(rmse)
    
    mean_rmse = np.mean(rmse_list)
    mean_rmse_per_depth[max_depth] = mean_rmse
    print(f"max_depth={max_depth}, Mean RMSE={mean_rmse:.3f}")

best_max_depth = min(mean_rmse_per_depth, key=mean_rmse_per_depth.get)
print(f"\nBest max_depth: {best_max_depth} with mean RMSE={mean_rmse_per_depth[best_max_depth]:.3f}")
print(f"Answer: {best_max_depth}")

# ===== QUESTION 5 =====
print("\n" + "="*60)
print("QUESTION 5: Most important feature")
print("="*60)

rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = rf.feature_importances_
feature_names = dv.get_feature_names_out()

# Create a dictionary of feature importance
importance_dict = dict(zip(feature_names, feature_importances))

# Look for specific features mentioned in the question
target_features = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
target_importances = {f: importance_dict.get(f, 0) for f in target_features}

print("\nFeature importances for target features:")
for feature, importance in sorted(target_importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")

most_important = max(target_importances, key=target_importances.get)
print(f"\nMost important feature: {most_important}")
print(f"Answer: {most_important}")

# ===== QUESTION 6 =====
print("\n" + "="*60)
print("QUESTION 6: XGBoost with different eta values")
print("="*60)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

watchlist = [(dtrain, 'train'), (dval, 'val')]

# Test eta=0.3
print("\nTraining with eta=0.3...")
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model_03 = xgb.train(xgb_params, dtrain, num_boost_round=100, 
                     evals=watchlist, verbose_eval=20)

y_pred_03 = model_03.predict(dval)
rmse_03 = np.sqrt(mean_squared_error(y_val, y_pred_03))

print(f"\nRMSE with eta=0.3: {rmse_03:.4f}")

# Test eta=0.1
print("\nTraining with eta=0.1...")
xgb_params['eta'] = 0.1

model_01 = xgb.train(xgb_params, dtrain, num_boost_round=100, 
                     evals=watchlist, verbose_eval=20)

y_pred_01 = model_01.predict(dval)
rmse_01 = np.sqrt(mean_squared_error(y_val, y_pred_01))

print(f"\nRMSE with eta=0.1: {rmse_01:.4f}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"eta=0.3: RMSE={rmse_03:.4f}")
print(f"eta=0.1: RMSE={rmse_01:.4f}")

if rmse_03 < rmse_01:
    print("Answer: 0.3")
elif rmse_01 < rmse_03:
    print("Answer: 0.1")
else:
    print("Answer: Both give equal value")

# ===== SUMMARY =====
print("\n" + "="*60)
print("SUMMARY OF ANSWERS")
print("="*60)
print(f"Question 1: {split_feature}")
print(f"Question 2: {rmse:.1f}")
print(f"Question 3: {stop_improving_n}")
print(f"Question 4: {best_max_depth}")
print(f"Question 5: {most_important}")
if rmse_03 < rmse_01:
    print("Question 6: 0.3")
elif rmse_01 < rmse_03:
    print("Question 6: 0.1")
else:
    print("Question 6: Both give equal value")
