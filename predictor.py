import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict from user input
print("\n--- Predict Pass/Fail ---")
gender = input("Enter Gender (Male/Female): ")
school = input("Enter School Type (Government/Private): ")
hours = float(input("Enter Study Hours per day: "))
attendance = float(input("Enter Attendance Percentage: "))
parent_edu = input("Enter Parent Education (NoSchool/Primary/HighSchool/Graduate): ")
area = input("Enter Area (Urban/Rural): ")

# Encode user inputs
input_data = pd.DataFrame([{
    "Gender": label_encoders["Gender"].transform([gender])[0],
    "School_Type": label_encoders["School_Type"].transform([school])[0],
    "Study_Hours": hours,
    "Attendance_Percentage": attendance,
    "Parent_Education": label_encoders["Parent_Education"].transform([parent_edu])[0],
    "Urban_or_Rural": label_encoders["Urban_or_Rural"].transform([area])[0]
}])

result = model.predict(input_data)[0]
prediction = label_encoders["Passed"].inverse_transform([result])[0]
print("Prediction:", prediction)
