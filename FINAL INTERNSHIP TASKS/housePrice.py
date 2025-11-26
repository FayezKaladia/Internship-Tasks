# ---------------------------------------
# HOUSE PRICE PREDICTION (CALIFORNIA DATASET)
# FINAL COMPLETE CODE
# ---------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------
# 1. Load California Housing Dataset
# ---------------------------------------

california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name="HouseValue")

print("\nDataset Loaded Successfully!")
print(X.head())

# ---------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------

# Histograms
X.hist(figsize=(12, 8))
plt.suptitle("Feature Histograms")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot (Median Income vs House Price)
plt.scatter(X["MedInc"], y, alpha=0.3)
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Median Income vs House Price")
plt.show()

# ---------------------------------------
# 3. Preprocessing (Normalization)
# ---------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# 4. Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------
# 5. Train Linear Regression Model
# ---------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------
# 6. Make Predictions
# ---------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------
# 7. Evaluation Metrics
# ---------------------------------------

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)

print("\nTask Completed Successfully!")
