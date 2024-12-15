import streamlit as st
import joblib
from load_model_predict import load_and_predict  # Importing load_and_predict from load_model_predict.py

# Load the training columns to ensure user input keys match them
try:
    training_columns = joblib.load('training_columns.pkl')
    st.write("Training columns loaded successfully.")
except Exception as e:
    st.error("Failed to load training columns.")
    st.stop()

# Streamlit Interface
st.title("Student Grade Prediction (G3)")

# Collect user inputs
st.header("Enter Student Information")

# Numeric Inputs
age = st.slider("Age", 15, 22, 18, help="Student's age, ranging from 15 to 22")
G1 = st.slider("First Period Grade (G1)", 0, 20, 10, step=1, key="G1_slider")
G2 = st.slider("Second Period Grade (G2)", 0, 20, 10, step=1, key="G2_slider")

# Travel Time (SelectBox)
traveltime = st.selectbox("Travel Time to School", options=[1, 2, 3, 4],
                          format_func=lambda x: {1: "<15 min", 2: "15-30 min", 3: "30 min to 1 hour", 4: ">1 hour"}[x], key="traveltime_select")

# Study Time (SelectBox)
studytime = st.selectbox("Weekly Study Time", options=[1, 2, 3, 4],
                         format_func=lambda x: {1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"}[x], key="studytime_select")

# Number of Past Failures (Segmented options)
failures = st.radio("Number of Past Failures", options=[0, 1, 2, 3],
                    format_func=lambda x: "{} failure(s)".format(x), key="failures_radio")

# Absences (Slider)
absences = st.slider("Number of Absences", 0, 93, 6, step=1, key="absences_slider")

# Categorical Inputs
famrel = st.slider("Family Relationship Quality", 1, 5, 4, key="famrel_slider")
freetime = st.slider("Free Time After School", 1, 5, 3, key="freetime_slider")
goout = st.slider("Going Out with Friends", 1, 5, 2, key="goout_slider")
Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1, key="dalc_slider")
Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1, key="walc_slider")
health = st.slider("Current Health Status", 1, 5, 5, key="health_slider")

# Collect categorical inputs (binary encoding)
sex_m = 1 if st.selectbox("Sex", ['M', 'F'], key="sex_select") == 'M' else 0
address_u = 1 if st.selectbox("Address", ['Urban', 'Rural'], key="address_select") == 'Urban' else 0
famsize_le3 = 1 if st.selectbox("Family Size", ['LE3', 'GT3'], key="famsize_select") == 'LE3' else 0
pstatus_t = 1 if st.selectbox("Parent Cohabitation Status", ['Together', 'Apart'], key="pstatus_select") == 'Together' else 0
internet = 1 if st.selectbox("Internet Access at Home", ['Yes', 'No'], key="internet_select") == 'Yes' else 0
romantic = 1 if st.selectbox("In a Romantic Relationship", ['Yes', 'No'], key="romantic_select") == 'Yes' else 0
higher = 1 if st.selectbox("Plans for Higher Education", ['Yes', 'No'], key="higher_select") == 'Yes' else 0
nursery = 1 if st.selectbox("Attended Nursery School", ['Yes', 'No'], key="nursery_select") == 'Yes' else 0

# Education Levels for Mother and Father
medu = st.selectbox("Mother's Education", options=[0, 1, 2, 3, 4],
                     format_func=lambda x: {0: "None", 1: "Primary", 2: "5th to 9th", 3: "Secondary", 4: "Higher"}[x], key="medu_select")
fedu = st.selectbox("Father's Education", options=[0, 1, 2, 3, 4],
                     format_func=lambda x: {0: "None", 1: "Primary", 2: "5th to 9th", 3: "Secondary", 4: "Higher"}[x], key="fedu_select")

# User Input Dictionary
user_input = {
    'age': age,
    'G1': G1,
    'G2': G2,
    'traveltime': traveltime,
    'studytime': studytime,
    'failures': failures,
    'absences': absences,
    'famrel': famrel,
    'freetime': freetime,
    'goout': goout,
    'Dalc': Dalc,
    'Walc': Walc,
    'health': health,
    'sex_M': sex_m,
    'address_U': address_u,
    'famsize_LE3': famsize_le3,
    'Pstatus_T': pstatus_t,
    'internet': internet,
    'romantic': romantic,
    'higher': higher,
    'nursery': nursery,
    'Medu': medu,
    'Fedu': fedu
}

# Check if training_columns contains the necessary keys and fill missing ones with 0
for column in training_columns:
    if column not in user_input:
        user_input[column] = 0  # Ensure all expected columns are included

# Debug output for user input
st.write("User input for prediction:", user_input)

# Prediction button
if st.button("Predict Final Grade (G3)"):
    predicted_g3 = load_and_predict(user_input)  # Call load_and_predict with user input
    if predicted_g3 is not None:
        st.success(f"Predicted Final Grade (G3): {predicted_g3:.2f}")
    else:
        st.error("Prediction failed. Please check the input values.")
