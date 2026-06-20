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


## 📘 Workflow

1. Create student study hours dataset  
2. Split data into training and testing sets  
3. Train Logistic Regression model  
4. Predict student pass or fail result  
5. Evaluate model accuracy  

---

## 🎯 Why Logistic Regression?

Logistic Regression was selected because the project predicts only two outcomes:

- Pass (1)  
- Fail (0)  

It is simple, fast, and effective for binary classification problems.  
The model works well for small datasets and helps beginners understand machine learning classification concepts easily.

---

## 📈 Evaluation

The model performance was evaluated using:

- Accuracy Score  
- Prediction Results  

---

## 🎓 Target Classes

- Pass  
- Fail  

Where:

- `1 = Pass`
- `0 = Fail`

---

## 📌 Learning Outcomes

Through this project, I learned:

- Data preprocessing  
- Train-test split  
- Classification algorithms  
- Logistic Regression model  
- Model evaluation using accuracy score  
- Basic machine learning workflow

