import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
df = pd.read_csv("dataset.csv")

print("Dataset:")
print(df)

# 2. Explore Dataset
print("\nFirst Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()


# 3. Separate Input Features and Target
X = df[["House_Size", "Bedrooms", "Bathrooms", "House_Age"]]

y = df["Price"]


# 4. Split Dataset into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# 5. Create Multiple Linear Regression Model
model = LinearRegression()


# 6. Train the Model
model.fit(X_train, y_train)

print("\nModel Training Completed!")


# 7. Display Coefficients and Intercept
print("\nCoefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)


# 8. Predict Test Data
y_pred = model.predict(X_test)

print("\nActual vs Predicted Prices:")

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(results)


# 9. Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# 10. Predict Price of a New House
new_house = pd.DataFrame({
    "House_Size": [1600],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "House_Age": [4]
})

predicted_price = model.predict(new_house)

print("\nNew House Details:")
print(new_house)

print("\nPredicted House Price:", predicted_price[0], "Lakhs")