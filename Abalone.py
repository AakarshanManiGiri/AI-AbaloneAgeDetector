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

# Load dataset
abalone = fetch_ucirepo(id=1)

# Split features and target
X = abalone.data.features.copy()
y = abalone.data.targets.copy()
df = X.copy()
df["Age"] = y

print("Dataset shape:", df.shape)
print(df.head(), "\n")
print("Age summary:\n", df["Age"].describe(), "\n")


# Convert rings to age
y = y + 1.5

# Remove rows with invalid height
valid_rows = X['Height'] > 0
X = X[valid_rows]
y = y[valid_rows]

# Define columns
categorical_features = ['Sex']
numerical_features = ['Length', 'Diameter', 'Height', 'Whole_weight', 
                      'Shucked_weight', 'Viscera_weight', 'Shell_weight']


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

# Set up preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features),
    ('num', Pipeline([('scaler', StandardScaler())]), numerical_features)
])

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Run cross-validation
print("Cross-validation results (R² scores):")
for name, reg in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    scores = cross_val_score(pipeline, X, y.to_numpy().ravel(), cv=5, scoring='r2')
    print(f"{name}: Mean R² = {scores.mean():.4f}, Std = {scores.std():.4f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().ravel(), test_size=0.2, random_state=42)

# Final evaluation
print("\nTest set results:")
best_model = None
best_mse = float('inf')
best_r2 = float('-inf')

for name, reg in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name}: MSE = {mse:.2f}, R² = {r2 * 100:.2f}%")
    
    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        best_model = name

print(f"\nBest model: {best_model}")
print(f"Best model MSE: {best_mse:.2f}")
print(f"Best model R²: {best_r2 * 100:.2f}%")