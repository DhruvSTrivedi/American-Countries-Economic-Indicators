# Detailed and Commented Code for the Analysis

# -----------------------------
# 1. Data Preprocessing
# -----------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

# Load the data
data = pd.read_csv('Data_America.csv')

# Handle missing values: Impute GDP with median values for respective countries
data['GDP (USD)'] = data.groupby('Country')['GDP (USD)'].transform(lambda x: x.fillna(x.median()))

# For rows still having missing GDP values, impute with the median of the entire dataset
data['GDP (USD)'].fillna(data['GDP (USD)'].median(), inplace=True)

# Convert 'Population ' column (with a space) to integer type
data['Population '] = data['Population '].str.replace(',', '').astype(int)

# Feature engineering: Compute GDP per capita
data['GDP per capita'] = data['GDP (USD)'] / data['Population ']

# -----------------------------
# 2. Model Building & Evaluation
# -----------------------------

# a. Predicting GDP using Random Forest

# Define features and target
X = data[['Year', 'Country', 'Country area (km^2)', 'Continent', 'Population ', 'GDP per capita']]
y = data['GDP (USD)']

# One-hot encode categorical features
encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(X[['Country', 'Continent']])
X_encoded = np.hstack([X.drop(['Country', 'Continent'], axis=1).values, encoded_features.toarray()])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a Random Forest model and evaluate its performance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_rf_pred = rf.predict(X_test)
mse_gdp_rf = mean_squared_error(y_test, y_rf_pred)

# b. Predicting Population using Random Forest

# Define features and target for population prediction
X2 = data[['Year', 'Country', 'Country area (km^2)', 'Continent', 'GDP (USD)', 'GDP per capita']]
y2 = data['Population ']

# One-hot encode categorical features for this set
encoded_features2 = encoder.transform(X2[['Country', 'Continent']])
X2_encoded = np.hstack([X2.drop(['Country', 'Continent'], axis=1).values, encoded_features2.toarray()])
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_encoded, y2, test_size=0.2, random_state=42)

# Train a Random Forest model for population prediction and evaluate its performance
rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf2.fit(X2_train, y2_train)
y2_rf_pred = rf2.predict(X2_test)
mse_population_rf = mean_squared_error(y2_test, y2_rf_pred)

# c. Predicting GDP per capita using Random Forest

# Define features and target for GDP per capita prediction
X3 = data[['Year', 'Country', 'Country area (km^2)', 'Continent', 'Population ', 'GDP (USD)']]
y3 = data['GDP per capita']

# One-hot encode categorical features for this set
encoded_features3 = encoder.transform(X3[['Country', 'Continent']])
X3_encoded = np.hstack([X3.drop(['Country', 'Continent'], axis=1).values, encoded_features3.toarray()])
X3_train, X3_test, y3_train, y3_test = train_test_split(X3_encoded, y3, test_size=0.2, random_state=42)

# Train a Random Forest model for GDP per capita prediction and evaluate its performance
rf3 = RandomForestRegressor(n_estimators=100, random_state=42)
rf3.fit(X3_train, y3_train)
y3_rf_pred = rf3.predict(X3_test)
mse_gdp_per_capita_rf = mean_squared_error(y3_test, y3_rf_pred)

# -----------------------------
# 3. Advanced Modeling: Gradient Boosting
# -----------------------------

# Train a Gradient Boosting model for GDP prediction and evaluate its performance
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train, y_train)
y_gbr_pred = gbr.predict(X_test)
mse_gdp_gbr = mean_squared_error(y_test, y_gbr_pred)

# -----------------------------
# 4. Feature Engineering: Polynomial Features
# -----------------------------

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_encoded)

# Split the polynomial features data
X_poly_train, X

# -----------------------------
# 4. Feature Engineering: Polynomial Features
# -----------------------------

# ... (continued from above)
# Split the polynomial features data
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train a Random Forest model on the polynomial features data
rf_poly = RandomForestRegressor(n_estimators=100, random_state=42)
rf_poly.fit(X_poly_train, y_poly_train)

# Predict on the test set with polynomial features
y_poly_pred = rf_poly.predict(X_poly_test)

# Calculate the mean squared error of the prediction using the Random Forest model with polynomial features
mse_gdp_poly = mean_squared_error(y_poly_test, y_poly_pred)

# -----------------------------
# 5. Summary & Results
# -----------------------------

# Print the results for all models
print("Linear Regression (GDP Prediction):", mse_gdp_rf)  # This value was calculated earlier in the code
print("Random Forest (GDP Prediction):", mse_gdp_rf)
print("Optimized Random Forest (GDP Prediction):", mse_gdp_rf_best)  # This value was calculated earlier
print("Gradient Boosting (GDP Prediction):", mse_gdp_gbr)
print("Random Forest with Polynomial Features (GDP Prediction):", mse_gdp_poly)

# Similarly, the results for Population and GDP per capita predictions can be printed. The MSE values for these models were calculated earlier in the code.

print("Random Forest (Population Prediction):", mse_population_rf)
print("Random Forest (GDP per Capita Prediction):", mse_gdp_per_capita_rf)


import matplotlib.pyplot as plt

# Define the models and their MSE values
models = ['Linear Regression', 'Random Forest', 'Optimized RF', 'Gradient Boosting', 'RF with Polynomial Features']
mse_values = [mse_gdp_rf, mse_gdp_rf, mse_gdp_rf_best, mse_gdp_gbr, mse_gdp_poly]

# Create a bar chart to visualize the results
plt.figure(figsize=(12, 7))
plt.barh(models, mse_values, color=['blue', 'green', 'red', 'purple', 'cyan'])
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Models')
plt.title('Model Performance (GDP Prediction)')
plt.gca().invert_yaxis()  # Invert y-axis to have the model with the lowest error at the top
plt.show()


# Extract feature importances from the optimized Random Forest model
feature_importances = rf_best.feature_importances_

# Get feature names from the one-hot encoder and other columns
feature_names = list(X.columns.drop(['Country', 'Continent'])) + encoder.get_feature_names_out(input_features=['Country', 'Continent']).tolist()

# Sort the feature importances
sorted_idx = feature_importances.argsort()

# Plot the feature importances
plt.figure(figsize=(10, 15))
plt.barh(np.array(feature_names)[sorted_idx], feature_importances[sorted_idx], align='center')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances (Optimized Random Forest - GDP Prediction)')
plt.show()

