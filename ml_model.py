# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate sample dataset
data = pd.DataFrame({
    'Feature1': np.random.randint(1, 100, 100),
    'Feature2': np.random.randint(1, 100, 100),
    'Label': np.random.choice([0, 1], 100)
})

# Split into features (X) and target (y)
X = data[['Feature1', 'Feature2']]
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
import joblib
joblib.dump(model, "model.pkl")
