import joblib
import streamlit as st
import pandas as pd

# Load trained model
model = joblib.load("linear_regression_model.pkl")

st.title("Exam Score Prediction")

# Options (match your dataset categories)
course_options = ['bca', 'diploma', 'b.com', 'ba', 'b.sc', 'b.tech', 'bba']
study_methods = ['self-study', 'online videos', 'group study', 'coaching', 'mixed']
sleep_quality_options = ['poor', 'average', 'good']
facility_options = ['low', 'medium', 'high']
exam_difficulty_options = ['easy', 'moderate', 'hard']

# User inputs
student_id = st.number_input("Student ID", min_value=0, value=1000, step=1)
course_selected = st.selectbox("Select Course", course_options)
study_method_selected = st.selectbox("Select study method", study_methods)

study_hours = st.number_input("Enter number of hours studied", min_value=0, value=5)
class_attendance = st.number_input("Enter class attendance percentage", min_value=0, max_value=100, value=75)
internet_access = st.selectbox("Do you have internet access?", ['yes', 'no'])

sleep_hours = st.number_input("Enter average sleep hours per night", min_value=0, max_value=24, value=7)
sleep_quality_selected = st.selectbox("Select sleep quality", sleep_quality_options)
facility_rating_selected = st.selectbox("Select facility rating", facility_options)
exam_difficulty_selected = st.selectbox("Select exam difficulty", exam_difficulty_options)

if st.button("Predict Exam Score"):
    # Build raw input row using TRAINING column names
    df_input = pd.DataFrame([{
        "student_id": int(student_id),
        "course": course_selected,
        "study_method": study_method_selected,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality_selected,
        "facility_rating": facility_rating_selected,
        "exam_difficulty": exam_difficulty_selected
    }])

    # ---- SAME CLEANING AS NOTEBOOK ----
    # standardize case
    for col in ["internet_access", "sleep_quality", "facility_rating", "exam_difficulty"]:
        df_input[col] = df_input[col].astype(str).str.lower()

    # map to numeric (same mappings)
    df_input["internet_access"] = df_input["internet_access"].map({"yes": 1, "no": 0})

    sleep_quality_map = {"poor": 1, "average": 2, "good": 3}
    facility_map = {"low": 1, "medium": 2, "high": 3}
    exam_diff_map = {"easy": 1, "moderate": 2, "hard": 3}

    df_input["sleep_quality"] = df_input["sleep_quality"].map(sleep_quality_map)
    df_input["facility_rating"] = df_input["facility_rating"].map(facility_map)
    df_input["exam_difficulty"] = df_input["exam_difficulty"].map(exam_diff_map)

    # one-hot encode like training
    df_input = pd.get_dummies(df_input, drop_first=True)

    # align to exactly what the model was trained on
    try:
        expected_cols = model.feature_names_in_
        df_input = df_input.reindex(columns=expected_cols, fill_value=0)
    except Exception:
        st.error("Model does not contain feature_names_in_. Re-train saving columns list.")
        st.stop()

    # predict
    try:
        pred = model.predict(df_input)[0]
        st.success(f"Predicted Exam Score: {pred:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
