import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into training and testing sets (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save the model and target names to a single file
# Saving target_names (e.g., 'setosa') helps your API return names instead of numbers (0, 1, 2)
model_data = {
    "model": model,
    "target_names": iris.target_names
}

joblib.dump(model_data, "iris_model.joblib")
print("Model and metadata saved successfully to iris_model.joblib")
