import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("students.csv")

# Encode categorical columns
label_encoders = {}
for column in ['Gender', 'School_Type', 'Parent_Education', 'Urban_or_Rural', 'Passed']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
X = df.drop("Passed", axis=1)
y = df["Passed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy, 2))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# --- Predict from user input ---
print("\n--- Predict Pass/Fail ---")
gender = input("Enter Gender (Male/Female): ")
school = input("Enter School Type (Government/Private): ")
hours = float(input("Enter Study Hours per day: "))
attendance = float(input("Enter Attendance Percentage: "))
parent_edu = input("Enter Parent Education (NoSchool/Primary/HighSchool/Graduate): ")
area = input("Enter Area (Urban/Rural): ")

# Raw user input for logging
user_input_raw = {
    "Gender": gender,
    "School_Type": school,
    "Study_Hours": hours,
    "Attendance_Percentage": attendance,
    "Parent_Education": parent_edu,
    "Urban_or_Rural": area
}

# Encoded user input for model
input_data = pd.DataFrame([{
    "Gender": label_encoders["Gender"].transform([gender])[0],
    "School_Type": label_encoders["School_Type"].transform([school])[0],
    "Study_Hours": hours,
    "Attendance_Percentage": attendance,
    "Parent_Education": label_encoders["Parent_Education"].transform([parent_edu])[0],
    "Urban_or_Rural": label_encoders["Urban_or_Rural"].transform([area])[0]
}])

# Make prediction
result_encoded = model.predict(input_data)[0]
result_label = label_encoders["Passed"].inverse_transform([result_encoded])[0]
print("Prediction:", result_label)

# Log the prediction
with open("prediction_log.txt", "a") as f:
    f.write(f"Input: {user_input_raw}, Prediction: {result_label}\n")
