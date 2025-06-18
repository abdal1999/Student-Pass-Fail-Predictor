# Student Performance Predictor

A simple machine learning project that predicts whether a student is likely to pass based on study hours, attendance, school type, and other features.

## Dataset

A synthetic dataset of 50 Indian students with the following columns:
- `Gender`: Male or Female
- `School_Type`: Government or Private
- `Study_Hours`: Daily average study hours
- `Attendance_Percentage`: Class attendance (in %)
- `Parent_Education`: Education level of parents (NoSchool, Primary, HighSchool, Graduate)
- `Urban_or_Rural`: Where the student lives
- `Passed`: Target column (Yes or No)

## Model Used

- **Decision Tree Classifier** from `scikit-learn`
- Encodes categorical values using `LabelEncoder`
- Trained on user-provided or existing CSV data

## Features

- Accepts user input (or reads from dataset)
- Predicts if a student will pass
- Shows model accuracy
- Clean and simple codebase for beginners

## ▶️ How to Run

1. Clone the repo or download the files.
2. Install dependencies:
