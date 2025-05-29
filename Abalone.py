from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from ucimlrepo import fetch_ucirepo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")           # nicer default style


# 1. Load dataset and basic cleaning

abalone = fetch_ucirepo(id=1)        # UCI repository

# Split features / target
X = abalone.data.features.copy()
y = abalone.data.targets.copy() + 1.5    # Rings → Age (yrs)

# Remove rows with invalid Height
mask = X["Height"] > 0
X, y = X[mask], y[mask]

# Define feature groups
categorical_features = ["Sex"]
numerical_features = [
    "Length", "Diameter", "Height", "Whole_weight",
    "Shucked_weight", "Viscera_weight", "Shell_weight"
]

# 2. Exploratory Data Analysis (EDA)
df = X.copy()
df["Age"] = y

print("Dataset shape:", df.shape)
print(df.head(), "\n")
print("Age summary:\n", df["Age"].describe(), "\n")

# 2.1 Age distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Distribution of Abalone Age")
plt.xlabel("Age (years)")
plt.tight_layout()
plt.show()

# 2.2 Correlation heat-map (numeric only)
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(
    numeric_df.corr(), annot=True, cmap="coolwarm",
    fmt=".2f", linewidths=0.5
)
plt.title("Correlation Heat-map – Numerical Features + Age")
plt.tight_layout()
plt.show()

# 2.3 Boxplots of numerical features
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_features])
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2.4 Histograms of numerical features
df[numerical_features].hist(
    bins=20, figsize=(12, 10), layout=(3, 3), color="lightblue"
)
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# 2.5 Average Age by Sex
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Sex", y="Age", palette="Set2")
plt.title("Average Age by Sex")
plt.tight_layout()
plt.show()




# 3. Pre-processing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features),
        ("num", Pipeline([("scaler", StandardScaler())]), numerical_features),
    ]
)


# 4. Model definitions
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(random_state=42),
}


# 5. Cross-validation (5-fold, R²)
print("\nCross-validation results (R²):")
for name, reg in models.items():
    pipe = Pipeline([("prep", preprocessor), ("reg", reg)])
    scores = cross_val_score(
        pipe, X, y.to_numpy().ravel(),
        cv=5, scoring="r2"
    )
    print(f"{name:>16}: mean = {scores.mean():.4f}, std = {scores.std():.4f}")


# 6. Train / test split and final evaluation

X_train, X_test, y_train, y_test = train_test_split(
    X, y.to_numpy().ravel(), test_size=0.2, random_state=42
)

print("\nHold-out test-set results:")
best_name, best_mse, best_r2 = None, float("inf"), float("-inf")

for name, reg in models.items():
    pipe = Pipeline([("prep", preprocessor), ("reg", reg)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"{name:>16}:  MSE = {mse:7.2f}   R² = {r2:6.3f}")

    if mse < best_mse:
        best_name, best_mse, best_r2 = name, mse, r2

print(f"\  Best model: {best_name}")
print(f"    Best MSE : {best_mse:.2f}")
print(f"    Best R²  : {best_r2:.3f}")