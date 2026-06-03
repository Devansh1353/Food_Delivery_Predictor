from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pandas as pd
df=pd.read_csv("food_delivery_dataset.csv")
df.head()
df['Traffic_Level'] = df['Traffic_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
df.head()
X=df[['Distance_km','Traffic_Level']]
y=df['Delivery_Time_Min']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print("R2 Score:",r2_score(y_test, prediction))
print("MAE:",mean_absolute_error(y_test,prediction))
print("Slope:", model.coef_[0])
print("Intercept:",model.intercept_)
print("X values:",X_test)
print("Prediction:",prediction)
