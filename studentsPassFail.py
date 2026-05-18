import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "Hours":[1,2,3,4,5],
    "res" : [0,0,1,1,1]
}

df = pd.DataFrame(data)

x = df[["Hours"]]
y = df["res"]

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
    )

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# New prediction
print("Prediction for 5 hours:", model.predict([[5]]))

