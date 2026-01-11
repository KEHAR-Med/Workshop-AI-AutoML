#kEHAR Mohamed Hamza_Homework_AutoML_TPOT
print('*'*20 + ' KEHAR Mohamed Hamza_Homework_AutoML_TPOT ' + '*'*20,'\n')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

print("=== AutoML Demo with TPOT ===")

# Step 1: Load data
print("1. Loading dataset...")
df = pd.read_csv('homework.csv')
print(f"Dataset shape: {df.shape}")



# Step 2: Initialize TPOT
print("\n2. Initializing TPOT...")

# Convert to TPOT format
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
tpot = TPOTRegressor(
    population_size=1,
    generations=1,
    random_state=42,
    verbose=0,
    n_jobs=1,
    early_stop=1
)
tpot.fit(x_train, y_train)

# Step 6: Display results
print("\n6. AutoML Results:")
print("\nBest pipeline found:")
print(tpot.fitted_pipeline_)

# Step 7: Get the best model
print(f"\n7. Best Model: TPOT optimized pipeline")
best_model = tpot.fitted_pipeline_

# Step 8: Make predictions
print("\n8. Predictions sample (first 5)...")
test_predictions = best_model.predict(x_test)

print("Predictions sample:")
print(test_predictions[:5])

# Step 9: Model performance
print("\n9. Model Performance:")
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
r2 = r2_score(y_test, test_predictions)
mae = np.mean(np.abs(y_test - test_predictions))

print(f"  RMSE: {rmse:.2f}")
print(f"  R²: {r2:.3f}")
print(f"  MAE: {mae:.2f}")
print("\n=== AutoML Complete TPOT ===")
