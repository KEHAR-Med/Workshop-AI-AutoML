#kEHAR Mohamed Hamza_Homework_AutoML_ AUTO-SKLEARN
print('*'*20 + ' KEHAR Mohamed Hamza_Homework_AutoML_AUTO-SKLEARN' + '*'*20,'\n')

import pandas as pd
import autosklearn.regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("=== AutoML Demo with AUTO-SKLEARN  ===")

# Step 1: Create sample data
print("1. Creating sample dataset...")
df = pd.read_csv('homework.csv')
# Convert to DataFrame
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
print(f"Dataset shape: {df.shape}")

# Step 2: Initialize AUTO-SKLEARN
print("\n2. Initializing AUTO-SKLEARN...")

# Step 3: Define features and target
X = df.drop('price', axis=1)
y = df['price']


# Step 4: Split the data
print("\n4. Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data: Train={X_train.shape[0]}, Test={X_test.shape[0]}")


# Step 5: Run AutoML
print("\n5. Running AutoML (this may take a few minutes)...")

automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120, seed=42)
print("\\nTraining...")
automl.fit(X_train, y_train)

# Step 6: Display results
print("\n6. AutoML Results:")
print("\nLeaderboard (Top Models):")
try:

    lb = automl.leaderboard()
    if lb is not None and not lb.empty:
        print(lb)
    else:
        print("Leaderboard is empty")
except AttributeError:
    try:

        cv_results = automl.cv_results_
        if cv_results is not None:
            print("Available metrics:", list(cv_results.keys()))
    except:
      print("(Leaderboard details not available)")

# Step 7: Get the best model
print(f"\n7. Best Model: ")
models = automl.show_models()
if models:
    print(f"Models in ensemble: {len(models)}")

    # Get model object
    model_info = list(models.values())[0]
    for key in model_info:
        if hasattr(model_info[key], 'predict'):
            print(f" Model type: {type(model_info[key]).__name__}")
            break

# Step 8: Make predictions
print("\n8. Predictions sample (first 5)...")
predictions = automl.predict(X_test)
for i in range(5):
    print(f"  {i + 1}. ${y_test.iloc[i]:.2f} → ${predictions[i]:.2f}")

# Step 9: Model performance
print("\n9. Model Performance:")
score = automl.score(X_test, y_test)

print(f'\n R² SCORE: {score:.4f}')

rmse = np.sqrt(mean_squared_error(y_test, automl.predict(X_test)))
print(f"  RMSE: {rmse:.2f}")

print("\n=== AutoML Complete AUTO-SKLEARN ===")