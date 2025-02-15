import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to split data into training and testing sets
def split_data(data):
    X = data.drop(["Fault"], axis=1)
    y = data["Fault"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train the Random Forest classifier
def train_random_forest(X_train, y_train):
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf_rf.fit(X_train, y_train)
    return clf_rf

# Function to take user input and make predictions
def predict_fault_status(user_input, trained_model, feature_columns):
    # Ensure user_input has the same length as the expected features
    if len(user_input) != len(feature_columns):
        raise ValueError("Invalid number of input values. Expected {} values.".format(len(feature_columns)))

    # Convert the user input into a DataFrame for prediction
    user_data = pd.DataFrame([user_input], columns=feature_columns)

    # Predict using the trained model
    user_prediction = trained_model.predict(user_data)

    return user_prediction[0]

# Assuming your original dataset is loaded into 'data'
# If you've already loaded the dataset in a previous cell, you can skip this step.
data = pd.read_csv(r"C:\Users\athar\Downloads\Processed_final_data.csv")

# Select the specified feature columns
selected_features = ['S1_F3', 'S1_F4', 'S1_F9', 'S1_F10', 'S1_F11', 'S1_F13', 'S2_F7', 'S2_F8', 'S2_F9', 'S2_F10']

# Convert 'Fault' into binary categorical variable
threshold = 0  # Set your threshold based on the actual data distribution
data['Fault'] = np.where(data['Fault'] > threshold, 1, 0)  # Use 1 for 'Broken' and 0 for 'Healthy'

# Extract selected features from the dataset
selected_data = data[['Fault'] + selected_features]

# Split the data
X_train, X_test, y_train, y_test = split_data(selected_data)

# Train Random Forest classifier
clf_rf = train_random_forest(X_train, y_train)

# Evaluate the accuracy on the testing set
y_pred = clf_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy on Testing Set: {:.2f}%".format(accuracy * 100))

# Take user input for vibrations
user_input_values = []
for feature in selected_features:
    user_input = float(input(f"Enter value for {feature}: "))
    user_input_values.append(user_input)

# Make prediction using user input
prediction = predict_fault_status(user_input_values, clf_rf, selected_features)

# Print the prediction
print("Predicted Fault Status: {}".format(prediction))
