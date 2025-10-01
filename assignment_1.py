import pandas as pd
import numpy as np
print("Hello, World!")
df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv")
print(df.head())
print(len(df))
fuel_type_count = df['fuel_type'].nunique()
print("Number of fuel types:", fuel_type_count)
max_efficiency_asia = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
print("Maximum fuel efficiency of cars from Asia:", max_efficiency_asia)

# Median before filling
median_before = df['horsepower'].median()

# Most frequent (mode) value
most_frequent = df['horsepower'].mode()[0]

# Fill missing values
df['horsepower_filled'] = df['horsepower'].fillna(most_frequent)

# Median after filling
median_after = df['horsepower_filled'].median()

# Check if it changed
if median_after > median_before:
    print("Yes, it increased")
elif median_after < median_before:
    print("Yes, it decreased")
else:
    print("No")

# Select cars from Asia
asia_cars = df[df['origin'] == 'Asia'][['vehicle_weight', 'model_year']].head(7)

# Get NumPy array
X = asia_cars.to_numpy()

# Compute XTX and its inverse
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)

# Create y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# Compute w
w = XTX_inv.dot(X.T).dot(y)

# Sum of weights
print("Sum of weights:", round(w.sum(), 3))


# Count columns with missing values
missing_columns = df.isnull().sum()
num_columns_with_missing = (missing_columns > 0).sum()

print("Number of columns with missing values:", num_columns_with_missing)