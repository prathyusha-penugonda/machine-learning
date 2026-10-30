import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score
#load the dataset using with the pandas
df=pd.read_csv("dataset.csv")
X=df[["House_Size"]]
Y=df["Price"]
#model training and testing starts here
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#print("Training Features:")
#print(X_train)

#print("Training Prices:")
#print(Y_train)

#print("Testing Features:")
#print(X_test)

#print("Actual Testing Prices:")
#print(Y_test)
model=LinearRegression()
model.fit(X_train,Y_train)
#print(model.coef_)
#print(model.intercept_)
y_pred=model.predict(X_test)
print("model is predicted value")
comparision=pd.DataFrame({
    "Actual_values":Y_test,
    "predicted_value":y_pred
})
new_house=pd.DataFrame({
    "House_Size":[1600]
})
y_pred2=model.predict(new_house)
# print(y_pred2)
#print(y_pred)
#print(Y_test)
print(comparision)
mae=mean_absolute_error(Y_test,y_pred)
mse=mean_squared_error(Y_test,y_pred)
r2s=r2_score(Y_test,y_pred)
print(mae)
print(mse)
print(r2s)
