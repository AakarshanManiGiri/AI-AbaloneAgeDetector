from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Fetch the Abalone dataset
abalone = fetch_ucirepo(id=1)

# Extract features (X) and target (y)
X = abalone.data.features
y = abalone.data.targets

# Convert target variable (Rings) to age
y = y + 1.5

# Define categorical and continuous features
categorical_features = ['Sex']  # Replace with the actual categorical column name
continuous_features = ['Length', 'Diameter', 'Height', 'Whole_weight', 
                       'Shucked_weight', 'Viscera_weight', 'Shell_weight']

# Define the transformer for categorical features
categorical_transformer = OneHotEncoder()

# Add scaling for continuous features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler())
        ]), continuous_features)
    ]
)

# Define multiple models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().ravel(), test_size=0.2, random_state=42)

best_model = None
best_mse = float('inf')
best_r2 = float('-inf')

# Evaluate each model
for name, regressor in models.items():
    # Create a pipeline with preprocessing and the current model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse:.2f}, Accuracy (R²): {r2 * 100:.2f}%")
    
    # Track the best model
    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        best_model = name

# Print the best model
print(f"\nBest Model: {best_model}")
print(f"Best Model - Mean Squared Error: {best_mse:.2f}, Accuracy (R²): {best_r2 * 100:.2f}%")