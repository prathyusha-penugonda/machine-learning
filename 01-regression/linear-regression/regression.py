
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("dataset.csv")

print("\nDataset:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

X = df[["House_Size"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

model = LinearRegression()
model.fit(X_train, y_train)

print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "House_Size": X_test["House_Size"].values,
    "Actual_Price": y_test.values,
    "Predicted_Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(comparison)

new_house = pd.DataFrame({"House_Size": [1600]})
predicted_price = model.predict(new_house)

print("\nNew House Prediction:")
print("House Size: 1600 sq. ft.")
print("Predicted Price:", round(predicted_price[0], 2), "Lakhs")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

print("\nLinear Regression Project Completed Successfully!")