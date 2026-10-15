"""Train a small classifier and save it to model.joblib.

Swap this for your own model: just make sure it ends by saving a file that
app.py can load.
"""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

joblib.dump({"model": model, "classes": list(iris.target_names)}, "model.joblib")
print("Saved model.joblib")
