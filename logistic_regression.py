import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("food_delivery_dataset.csv")

data = pd.get_dummies(
    data,
    columns=["Traffic_Level", "Weather", "Vehicle_Type"],
    drop_first=True
)

x = data.drop(["On_Time", "Delivery_Time_Min"], axis=1)

y = data["On_Time"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

probs = model.predict_proba(x_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nReport:")
print(classification_report(y_test, predictions))

print("\nX_test:")
print(x_test)

print("\nY_test (Actual):")
print(y_test.values)

print("\nPredictions:")
print(predictions)

print("\nProbabilities:")
print(probs)