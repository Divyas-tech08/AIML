from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
pred = model.predict(new_flower)

print("Predicted Flower Class:", pred)
print("Flower Name:", iris.target_names[pred][0])


# 🌸 Iris Flower Classification using KNN

This project is a Machine Learning classification model built using the famous Iris dataset. The model predicts the species of an iris flower based on features such as sepal length, sepal width, petal length, and petal width.

## 🤖 Machine Learning Model
- K-Nearest Neighbors (KNN)

## 📊 Workflow
1. Load Iris dataset
2. Split data into training and testing sets
3. Train KNN model
4. Predict flower species
5. Evaluate model accuracy

## 🎯 Why KNN?
KNN was selected because the Iris dataset is small, clean, and well-structured. Similar flower measurements tend to belong to the same species, making KNN effective for this classification task.

## 📈 Evaluation
The model performance was evaluated using:
- Accuracy Score
- Confusion Matrix

## 🌸 Target Classes
- Setosa
- Versicolor
- Virginica

## 📌 Learning Outcomes
Through this project, I learned:
- Data preprocessing
- Train-test split
- Classification algorithms
- Model evaluation metrics
- Basic machine learning workflow
