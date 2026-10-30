# HOUSE PRICE PREDICTION USING LINEAR REGRESSION

## 1. Project Overview

This is my first Machine Learning project, developed as part of my AI Engineering learning journey.

The main objective of this project is to understand how a Machine Learning model learns from data and predicts house prices based on house sizes.

I implemented a Linear Regression algorithm using Python and Scikit-learn.

This project covers the complete workflow, starting from environment setup and data preparation to model training, prediction, and evaluation.

---

## 2. Technologies Used

- Python 3.12
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- VS Code
- Ubuntu Linux
- Git and GitHub

---

## 3. Project Structure

```text
machine-learning/
│
├── venv/
│
├── 01-regression/
│   └── linear-regression/
│       │
│       ├── dataset.csv
│       ├── regression.py
│       └── README.md
│
├── .gitignore
└── requirements.txt
```

**Note:** The virtual environment (`venv`) is excluded from GitHub using `.gitignore`.

---

## 4. Environment Setup and Installation

### Step 1: Install Python

Check whether Python is installed:

```bash
python3 --version
```

Check pip:

```bash
pip3 --version
```

### Step 2: Create the Project Folder

```bash
mkdir -p ~/AI-Engineering/machine-learning
cd ~/AI-Engineering/machine-learning
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 4: Activate the Virtual Environment

```bash
source venv/bin/activate
```

After activation, the terminal displays:

```text
(venv)
```

### Step 5: Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Step 6: Verify Installed Libraries

```bash
pip list
```

---

## 5. Dataset Preparation

I created a CSV dataset named `dataset.csv`.

The dataset contains two columns:

- **House_Size:** Size of the house in square feet.
- **Price:** House price in lakhs.

### Dataset

| House Size (sq. ft.) | Price (Lakhs) |
|---:|---:|
| 1000 | 40 |
| 1200 | 48 |
| 1500 | 60 |
| 1800 | 72 |
| 2000 | 80 |
| 2200 | 88 |
| 2500 | 100 |
| 2800 | 112 |
| 3000 | 120 |

The dataset contains 9 records.

---

## 6. Understanding the Machine Learning Workflow

The following steps were implemented:

1. Import required libraries.
2. Load the dataset using Pandas.
3. Explore and understand the dataset.
4. Separate independent and dependent variables.
5. Split the dataset into training and testing data.
6. Create a Linear Regression model.
7. Train the model using training data.
8. Predict house prices using testing data.
9. Predict the price of a new house.
10. Evaluate the model's performance.

---

## 7. Data Loading and Exploration

I used Pandas to load and explore the dataset.

```python
import pandas as pd

df = pd.read_csv("dataset.csv")

print(df.head())
print(df.shape)
print(df.info())
```

### Concepts Learned

- `read_csv()` loads CSV data into a DataFrame.
- `head()` displays the first five records.
- `shape` returns the number of rows and columns.
- `info()` provides information about columns and data types.

---

## 8. Separating Input and Output Variables

In Machine Learning:

- **X:** Independent variable (input).
- **Y:** Dependent variable (output).

For this project:

- X = House_Size
- Y = Price

```python
X = df[["House_Size"]]

Y = df["Price"]

print(X)
print(Y)

print(X.shape)
print(Y.shape)
```

The input shape is `(9, 1)` and the output shape is `(9,)`.

---

## 9. Splitting Training and Testing Data

I used Scikit-learn's `train_test_split()` function.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
```

### Concepts Learned

- Training data is used to teach the model.
- Testing data is used to check how well the model predicts unseen records.
- `test_size=0.2` reserves 20% of the dataset for testing.
- `random_state=42` makes the split reproducible.

For this dataset:

- Training records: 7
- Testing records: 2

---

## 10. Building and Training the Linear Regression Model

I used the `LinearRegression` algorithm from Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, Y_train)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
```

### Concepts Learned

- `LinearRegression()` creates the model.
- `fit()` trains the model using training data.
- `coef_` represents the slope.
- `intercept_` represents the intercept.

The model learns the relationship between house size and house price.

---

## 11. Making Predictions

After training, I used the model to predict prices for the testing dataset.

```python
Y_pred = model.predict(X_test)

print("Predicted Prices:")
print(Y_pred)

print("Actual Prices:")
print(Y_test)
```

### Comparing Actual and Predicted Values

```python
comparison = pd.DataFrame({
    "Actual Price": Y_test.values,
    "Predicted Price": Y_pred
})

print(comparison)
```

This comparison helps us understand the difference between actual and predicted prices.

---

## 12. Predicting the Price of a New House

I also tested the model with a new house size of 1600 square feet.

```python
new_house = pd.DataFrame({
    "House_Size": [1600]
})

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0], "Lakhs")
```

### Expected Result

```text
Predicted House Price: 64.0 Lakhs
```

The model predicts approximately ₹64 lakhs for a 1600 sq. ft. house.

---

## 13. Model Evaluation

I used evaluation metrics to understand the performance of the model.

```python
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(Y_test, Y_pred)

mse = mean_squared_error(Y_test, Y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(Y_test, Y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)
```

### Evaluation Metrics Explained

**1. Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted values.

**2. Mean Squared Error (MSE)**

Calculates the average squared prediction error. Larger errors receive a greater penalty.

**3. Root Mean Squared Error (RMSE)**

The square root of MSE. It expresses the error in the same unit as the house price (lakhs).

**4. R² Score**

Measures how well the model explains variations in the target values.

- 1.0 = Perfect predictions.
- 0.0 = No improvement over predicting the test-set mean.
- Negative = Worse than predicting the test-set mean.

---

## 14. How to Run the Project

Follow these steps whenever you want to execute the project.

### Step 1: Open the Project Directory

```bash
cd ~/AI-Engineering/machine-learning
```

### Step 2: Activate the Virtual Environment

```bash
source venv/bin/activate
```

### Step 3: Navigate to the Linear Regression Folder

```bash
cd 01-regression/linear-regression
```

### Step 4: Execute the Python File

```bash
python regression.py
```

The program displays:

- Model slope and intercept.
- Actual and predicted prices.
- Prediction for a new house.
- Model evaluation metrics.

---

## 15. What I Learned From This Project

Through this project, I gained practical experience in:

- Setting up Python and virtual environments in Linux.
- Installing and managing Python libraries.
- Reading and exploring datasets using Pandas.
- Understanding rows, columns, DataFrames, and Series.
- Separating independent and dependent variables.
- Understanding training and testing datasets.
- Implementing Linear Regression using Scikit-learn.
- Training a Machine Learning model using `fit()`.
- Making predictions using `predict()`.
- Evaluating model performance using MAE, MSE, RMSE, and R².
- Understanding the complete basic Machine Learning workflow.

---

## 16. Important Note

This project uses a small, manually created dataset containing only 9 records.

The dataset follows a simple linear relationship, so the model can learn it easily.

This project is intended for learning and understanding the fundamentals of Linear Regression. Its predictions should not be treated as real-world property valuations.

---

## 17. Author

**Prathyusha Penugonda**

GitHub: https://github.com/prathyusha-penugonda

**Project:** Machine Learning – Linear Regression

**Learning Goal:** Building practical foundations for AI Engineering.