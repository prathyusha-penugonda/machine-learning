# LINEAR REGRESSION – HOUSE PRICE PREDICTION

## 1. Project Overview
This project implements a Machine Learning Linear Regression model to predict house prices based on house size.

The model learns the relationship between house size and price using training data and predicts prices for unseen houses.

## 2. Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn

## 3. Dataset
The dataset contains two columns:

- House_Size: Size of the house in square feet.
- Price: Price of the house in lakhs.

## 4. Machine Learning Concepts Implemented
- Dataset loading using Pandas
- Feature and target selection
- Training and testing data splitting
- Linear Regression model creation
- Model training using `fit()`
- Prediction using `predict()`
- Model evaluation using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score

## 5. Project Structure

```text
linear-regression/
├── dataset.csv
├── regression.py
└── README.md
```

## 6. How to Run

Clone the repository and navigate to the project folder.

Install the required libraries:

```bash
pip install numpy pandas scikit-learn
```

Run the Python program:

```bash
python regression.py
```

## 7. Learning Outcome
Understood the basic Machine Learning workflow, trained a Linear Regression model, generated predictions, and evaluated model performance.

## 8. Note
This project uses a small, artificially created dataset for learning purposes. Its results should not be treated as real-world house price predictions.
