# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# Configuration and Data Loading

# %%
MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0,random_state=42),
    "Lasso Regression": Lasso(alpha=0.1,random_state=42),
    "ElasticNet Regression": ElasticNet(alpha=0.1,l1_ratio=0.5,random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5,random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100,random_state=42)
}

# %%
try:
    df = pd.read_csv("GS.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: GS.csv not found. Please entire the file is the same directory")
    exit()

df

# %% [markdown]
# Data Preprocessing

# %%
# Convert the "date" column to datetime objects
df["Date"] = pd.to_datetime(df["Date"])

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# %% [markdown]
# Feature Engineering

# %%
features = ["Open","High","Low","Volume"]
target = "Adj Close"

# Define the feature matrix (X) and the target variable (y)
X = df[features]
y = df[target]

# %% [markdown]
# Data Scaling
# 
# 

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled,columns=features)

# %% [markdown]
# Data Splitting

# %%
# Split the data into training and testing sets (80% train, 20% test)
# random_state ensures reproducibility of the split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# %% [markdown]
# Visualization Before Training

# %%
plt.figure(figsize=(10,6))
sns.scatterplot(x=df["Open"],y=df[target],data=df)
plt.title("Pre-Training Visualization: Open Price vs Adjusted Close Price")
plt.xlabel("Open Price")
plt.ylabel("Adjusted Close Price")
plt.grid(True)
plt.show()

# %% [markdown]
# Model Training and Comparison

# %%
results = {}

print("----- Model Training and Evaluation-----")
for name , model in MODELS.items():
    print(f"Training {name}........")
    # Train the model using the scaled training data
    model.fit(X_train,y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the performance 
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)

    # Store the results
    results[name] = {
        "RMSE":rmse,
        "R2": r2,
        "Model": model
    }

    # Print the results
    print(f"   {name}: R2 = {r2:.4f}, RMSE= ${rmse:.2f}")


# Find the best model based on R2 score (highest R2 typically best for regression)
best_model_name = max(results,key=lambda k :results[k]["R2"])
best_model_data = results[best_model_name]
best_model = best_model_data["Model"]


print(f"----- Best Model Selection -----")
print(f"The best model is '{best_model_name}' with R2 = {best_model_data["R2"]:.4f}")

# %% [markdown]
# Visualization After Training

# %%
# Get predictions from the best model
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10,6))
# scatter plot of Actual values vs Predicted values
# A perfect model would have all points lying exactly on the diagonal line (y=x)
plt.scatter(y_test,y_pred_best,alpha=0.6)
# Plot the ideal prediction line (y=x) for comparison
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"r--",lw=2)
plt.title(f"Post-Training Visualization: Actual vs Predicted Prices ({best_model_name})")
plt.xlabel("Actual Adjusted Close Price")
plt.ylabel("Predicted Ajusted Close Price")
plt.grid(True)
plt.show()

# %% [markdown]
# Interactive Prediction and Input Function

# %%
def make_single_prediction(model,scaler,feature_names):
    print(f"Interactive Prediction using {best_model_name}")
    print("Enter the following values to predict the Adjusted Close Price")

    # Initialize a list to hold the user's input values
    user_input_values = []

    # Loop through the required features and ask the user for input
    for feature in feature_names:
        while True:
            try:
                # Prompt the user for the value of the current feature
                value = float(input(f"Enter {feature}:"))
                user_input_values.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value")
    # Convert the list of inputs into a numpy array (row vector)
    new_data = np.array([user_input_values])

    # The model was trained on scaled data, so the new data must also be scaled
    new_data_scaled = scaler.transform(new_data)

    # Make predictiond using the best trained model
    predicted_price = model.predict(new_data_scaled)[0]

    # Print the final result
    print("-"*50)
    print(f"Input: Open={user_input_values[0]}, High={user_input_values[1]}, Low={user_input_values[2]}, Volume={user_input_values[3]}")
    print(f"The predicted Adjusted Close Price is: ${predicted_price:.2f}")
    print("-"*50)


# Run the interactive prediction function
make_single_prediction(best_model,scaler,features)


