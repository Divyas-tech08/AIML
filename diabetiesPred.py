import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Glucose": [80, 120, 130, 95, 150, 160, 85, 140],
    "Diabetes": [0,0,1,0,1,1,0,1]
}

df = pd.DataFrame(data)

X = df[["Glucose"]]
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# New prediction
print("Prediction for glucose 145:", model.predict([[145]]))