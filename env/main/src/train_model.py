import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load dataset
df = pd.read_csv('data/isl_landmarks.csv')

# Step 2: Split into features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 4: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate and print accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Step 6: Save the model to models/ folder
with open('models/gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved to models/gesture_model.pkl")