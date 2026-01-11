#kEHAR Mohamed Hamza_Homework_AutoML_MLJAR      
print('*'*20 + ' KEHAR Mohamed Hamza_Homework_AutoML_MLJAR ' + '*'*20,'\n')

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from supervised import AutoML
from sklearn.model_selection import train_test_split

print("=== AutoML Demo with MLJAR ===")

# Step 1: Load data
print("1. Loading dataset...")
df = pd.read_csv('homework.csv')
print(f"Dataset shape: {df.shape}")

# Step 2: Initialize MLJAR
print("\n2. Initializing MLJAR...")

# Step 3: Define features and target
# Convert to MLJAR format
print("\n3. Setting up AutoML...")
x = df.drop(columns=['price'])
y = df['price']

print(f"Features: {len(x.columns)}")
print(f"Target: price")

# Step 4: Split the data
print("\n4. Splitting data...")


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(f"Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")

# Step 5: Run AutoML
print("\n5. Running AutoML (this may take a few minutes)...")
automl = AutoML(
    total_time_limit=300,
    random_state=42,
    mode="Compete",
    validation_strategy={
        "validation_type": "split",
        "train_ratio": 0.8,
        "shuffle": True,
        "random_seed": 42
    }
)

#automl.fit(x, y)
automl.fit(x_train, y_train)
# Step 6: Display results
print("\n6. AutoML Results:")
print("\nLeaderboard (Top Models):")
try:
    # MLJar stores results in the results_path folder
    results = pd.read_csv("MLJar_Results/leaderboard.csv")
    print("\nLeaderboard (Top Models):")
    print(results[['model_type', 'metric_value', 'train_time']].head())
except:
    print("(Leaderboard will be in MLJar_Results folder)")


# Step 7: Get the best model
print(f"\n7. Best Model: {automl._best_model.get_name() if automl._best_model else 'Not available'}")
best_model = automl._best_model
# Step 8: Make predictions
print("\n8. Predictions sample (first 5)...")
test_predictions = best_model.predict(x_test)

print("Predictions sample:")
predictions = automl.predict(x_test)
for i in range(5):
    print(f"  {i + 1}. ${y_test.iloc[i]:.2f} → ${predictions[i]:.2f}")

# Step 9: Model performance
print("\n9. Model Performance:")

rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
r2 = r2_score(y_test, test_predictions)
print(f"  RMSE: {rmse:.2f}")
print(f"  R²: {r2:.3f}")

print("\n=== AutoML Complete MLJAR ===")