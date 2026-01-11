#kEHAR Mohamed Hamza_Homework_AutoML_H2O
print('*'*20 + ' KEHAR Mohamed Hamza_Homework_AutoML_H2O ' + '*'*20,'\n')

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("=== AutoML Demo with H2O ===")


# Step 1: Create sample data
print("1. Creating sample dataset...")

df = pd.read_csv('homework.csv')

# Convert to DataFrame

print(f"Dataset shape: {df.shape}")

# Step 2: Initialize H2O
print("\n2. Initializing H2O...")
h2o.init()

# Convert pandas DataFrame to H2O Fram
h2o_df = h2o.H2OFrame(df)

# Step 3: Define features and target
print("\n3. Setting up AutoML...")
x = h2o_df.columns
y = "price"
x.remove(y)

print(f"Features: {len(x)}")
print(f"Target: {y}")

# Step 4: Split the data
train, test = h2o_df.split_frame(ratios=[0.8], seed=42)
print(f"Training set: {train.shape[0]} rows")
print(f"Test set: {test.shape[0]} rows")

# Step 5: Run AutoML
print("\n4. Running AutoML (this may take a few minutes)...")
aml = H2OAutoML(
max_models=10, # Maximum number of models
seed=42, # Reproducibility
max_runtime_secs=300, # 5 minutes max runtime
verbosity="info" # Show progress
)

aml.train(x=x, y=y, training_frame=train)

# Step 6: Display results
print("\n5. AutoML Results:")
print("\nLeaderboard (Top Models):")
lb = aml.leaderboard
print(lb.head())

# Step 7: Get the best model
print(f"\nBest Model: {aml.leader.model_id}")
best_model = aml.leader

# Step 8: Make predictions
print("\n6. Making predictions...")
predictions = best_model.predict(test)
print("Predictions sample:")
print(predictions.head())

# Step 9: Model performance
print("\n7. Model Performance:")
performance = best_model.model_performance(test)
print(performance)

# Step 10: Shutdown H2O
print("\n8. Shutting down H2O...")
h2o.cluster().shutdown()

print("\n=== AutoML Complete H2O===")