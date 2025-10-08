
import streamlit as st
import pandas as pd
import requests


# Load columns from test data
columns = pd.read_csv("data/processed/X_test.csv").columns.tolist()

# Feature type mapping (from features_list.md)
categorical_features = [
    "Marital status", "Application mode", "Course", "Daytime/evening attendance", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"
]
numerical_features = [
    "Application order", "Age at enrollment",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP", "Total_credits", "GPA_ratio"
]

# Placeholder encoding maps (update with your actual mappings)
encoding_maps = {
    "Gender": {"Male": 0, "Female": 1},
    "Marital status": {"Single": 0, "Married": 1, "Widowed": 2, "Divorced": 3},
    "Application mode": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "10": 10, "15": 15, "16": 16, "17": 17, "18": 18, "26": 26, "27": 27, "39": 39, "42": 42, "43": 43, "44": 44, "51": 51, "53": 53, "57": 57, "58": 58, "59": 59, "62": 62, "65": 65, "72": 72},
    "Course": {"Biofuel Production Technologies": 33, "Animation and Multimedia Design": 171, "Social Service (evening attendance)": 8014, "Agronomy": 9003, "Communication Design": 9070, "Veterinary Nursing": 9085, "Informatics Engineering": 9119, "Equinculture": 9130, "Management": 9147, "Social Service": 9238, "Tourism": 9254},
    "Daytime/evening attendance": {"Daytime": 1, "Evening": 0},
    "Previous qualification": {"Secondary education": 1, "Higher education - bachelor's degree": 2, "Higher education - degree": 3, "Higher education - master's": 4, "12th year of schooling - not completed": 5, "11th year of schooling - not completed": 6, "Other": 9},
    "Nacionality": {"Portuguese": 1, "German": 2, "Spanish": 6, "Italian": 11, "Dutch": 13, "Cape Verdean": 14, "Angolan": 17, "Guinean": 21, "Mozambican": 22, "Santomean": 24, "Brazilian": 25, "Romanian": 26, "Moldova (Republic of)": 32, "Ukrainian": 41, "Lithuanian": 62, "American": 100, "Other": 105},
    "Mother's qualification": {"Secondary education": 1, "Higher education - bachelor's degree": 2, "Higher education - degree": 3, "Higher education - master's": 4, "12th year of schooling - not completed": 5, "11th year of schooling - not completed": 6, "Other": 9},
    "Father's qualification": {"Secondary education": 1, "Higher education - bachelor's degree": 2, "Higher education - degree": 3, "Higher education - master's": 4, "12th year of schooling - not completed": 5, "11th year of schooling - not completed": 6, "Other": 9},
    "Mother's occupation": {"Student": 0, "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 1, "Specialists in Intellectual and Scientific Activities": 2, "Intermediate Level Technicians and Professions": 3, "Administrative staff": 4, "Personal Services, Security and Safety Workers and Sellers": 5, "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry": 6, "Skilled Workers in Industry, Construction and Craftsmen": 7, "Installation and Machine Operators and Assembly Workers": 8, "Unskilled Workers": 9, "Armed Forces Professions": 10, "Other": 99},
    "Father's occupation": {"Student": 0, "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 1, "Specialists in Intellectual and Scientific Activities": 2, "Intermediate Level Technicians and Professions": 3, "Administrative staff": 4, "Personal Services, Security and Safety Workers and Sellers": 5, "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry": 6, "Skilled Workers in Industry, Construction and Craftsmen": 7, "Installation and Machine Operators and Assembly Workers": 8, "Unskilled Workers": 9, "Armed Forces Professions": 10, "Other": 99},
    "Displaced": {"No": 0, "Yes": 1},
    "Educational special needs": {"No": 0, "Yes": 1},
    "Debtor": {"No": 0, "Yes": 1},
    "Tuition fees up to date": {"No": 0, "Yes": 1},
    "Scholarship holder": {"No": 0, "Yes": 1},
    "International": {"No": 0, "Yes": 1}
}


st.title("Student Dropout Risk Predictor")
st.write("Enter student features below:")


user_input = []
for col in columns:
    if col in categorical_features:
        options = list(encoding_maps.get(col, {}).keys())
        if options:
            val = st.selectbox(f"{col}", options)
            user_input.append(encoding_maps[col][val])
        else:
            val = st.text_input(f"{col} (categorical, please enter code)")
            user_input.append(int(val) if val else 0)
    else:
        val = st.number_input(f"{col}", value=0.0)
        user_input.append(val)


if st.button("Predict"):
    try:
        features = [float(x) for x in user_input]
        response = requests.post(
            "http://localhost:8000/predict",
            json={"features": features}
        )
        if response.status_code == 200:
            pred = response.json()["prediction"]
            st.success(f"Predicted class: {pred}")
            if pred == 1:
                st.warning("High risk of dropout")
            else:
                st.info("Low risk of dropout")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")
