# Student Performance Predictor

A beginner-friendly machine learning project that predicts whether a student is likely to pass based on their study habits, school type, and other personal factors. Designed using real-world Indian education context.

---

## Dataset

A synthetic dataset of 50 Indian students with the following columns:

- `Gender`: Male or Female  
- `School_Type`: Government or Private  
- `Study_Hours`: Daily average study hours  
- `Attendance_Percentage`: Class attendance in %  
- `Parent_Education`: NoSchool, Primary, HighSchool, Graduate  
- `Urban_or_Rural`: Living area  
- `Passed`: Target column — Yes or No

---

## Model Used

- **Logistic Regression** from `scikit-learn`
- Label encoding for categorical columns using `LabelEncoder`
- Trained using 80/20 train-test split

---

## Features

- Accepts real-time user input via terminal
- Predicts whether a student will **pass** or **fail**
- Calculates and prints model **accuracy**
- Saves and displays a **confusion matrix**
- Logs all user inputs and predictions to a `.txt` file

---

## How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor
````

### Step 2: Set up virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# OR
source venv/bin/activate   # On Mac/Linux
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the predictor

```bash
python predictor.py
```

---

## Notes

* This is an educational project with synthetic data, intended to demonstrate how classification models work in Python.
* You can modify or expand the dataset to train with more student records.

```