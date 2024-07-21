import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'C:\\Users\\~ideapadGAMING~\\Desktop\\Parkinson\\ML_MODEL\\lblabla.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('Parkinsons Diagnosis Prediction')

    # Add a description
    st.write('Enter patient information to predict diagnosis.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Patient Information')

        # Add input fields for features
        patient_name = st.text_input('Patient Name')
        age = st.slider("Age", 1, 100, 50)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        ethnicity = st.selectbox("Ethnicity", ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other'])
        education_level = st.selectbox("Education Level", ['High School', 'Some College', 'College Graduate', 'Post Graduate'])
        bmi = st.slider("BMI", 15, 40, 25)
        smoking = st.selectbox("Smoking", ['Never', 'Former', 'Current'])
        alcohol_consumption = st.slider("Alcohol Consumption (per week)", 0, 21, 7)
        physical_activity = st.slider("Physical Activity (minutes per week)", 0, 300, 150)
        diet_quality = st.selectbox("Diet Quality", ['Poor', 'Fair', 'Good', 'Excellent'])
        sleep_quality = st.selectbox("Sleep Quality", ['Poor', 'Fair', 'Good', 'Excellent'])
        family_history_parkinsons = st.selectbox("Family History of Parkinsons", ['No', 'Yes'])
        traumatic_brain_injury = st.selectbox("Traumatic Brain Injury", ['No', 'Yes'])
        hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
        diabetes = st.selectbox("Diabetes", ['No', 'Yes'])
        depression = st.selectbox("Depression", ['No', 'Yes'])
        stroke = st.selectbox("Stroke", ['No', 'Yes'])
        systolic_bp = st.slider("Systolic Blood Pressure", 90, 200, 120)
        diastolic_bp = st.slider("Diastolic Blood Pressure", 60, 120, 80)
        cholesterol_total = st.slider("Total Cholesterol", 100, 300, 200)
        cholesterol_ldl = st.slider("LDL Cholesterol", 50, 200, 100)
        cholesterol_hdl = st.slider("HDL Cholesterol", 30, 100, 60)
        cholesterol_triglycerides = st.slider("Triglycerides", 50, 500, 150)
        updrs = st.slider("UPDRS Score", 0, 100, 50)
        moca = st.slider("MoCA Score", 0, 30, 15)
        functional_assessment = st.slider("Functional Assessment Score", 0, 100, 50)
        tremor = st.selectbox("Tremor", ['None', 'Mild', 'Moderate', 'Severe'])
        rigidity = st.selectbox("Rigidity", ['None', 'Mild', 'Moderate', 'Severe'])
        bradykinesia = st.selectbox("Bradykinesia", ['None', 'Mild', 'Moderate', 'Severe'])
        postural_instability = st.selectbox("Postural Instability", ['None', 'Mild', 'Moderate', 'Severe'])
        speech_problems = st.selectbox("Speech Problems", ['None', 'Mild', 'Moderate', 'Severe'])
        sleep_disorders = st.selectbox("Sleep Disorders", ['None', 'Mild', 'Moderate', 'Severe'])
        constipation = st.selectbox("Constipation", ['None', 'Mild', 'Moderate', 'Severe'])

    # Convert categorical inputs to numerical
    gender = 1 if gender == 'Male' else 0
    ethnicity = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Hispanic': 3, 'Other': 4}.get(ethnicity, 0)
    education_level = {'High School': 0, 'Some College': 1, 'College Graduate': 2, 'Post Graduate': 3}.get(education_level, 0)
    smoking = {'Never': 0, 'Former': 1, 'Current': 2}.get(smoking, 0)
    diet_quality = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}.get(diet_quality, 0)
    sleep_quality = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}.get(sleep_quality, 0)
    family_history_parkinsons = 1 if family_history_parkinsons == 'Yes' else 0
    traumatic_brain_injury = 1 if traumatic_brain_injury == 'Yes' else 0
    hypertension = 1 if hypertension == 'Yes' else 0
    diabetes = 1 if diabetes == 'Yes' else 0
    depression = 1 if depression == 'Yes' else 0
    stroke = 1 if stroke == 'Yes' else 0
    tremor = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(tremor, 0)
    rigidity = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(rigidity, 0)
    bradykinesia = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(bradykinesia, 0)
    postural_instability = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(postural_instability, 0)
    speech_problems = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(speech_problems, 0)
    sleep_disorders = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(sleep_disorders, 0)
    constipation = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(constipation, 0)

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Ethnicity': [ethnicity],
        'EducationLevel': [education_level],
        'BMI': [bmi],
        'Smoking': [smoking],
        'AlcoholConsumption': [alcohol_consumption],
        'PhysicalActivity': [physical_activity],
        'DietQuality': [diet_quality],
        'SleepQuality': [sleep_quality],
        'FamilyHistoryParkinsons': [family_history_parkinsons],
        'TraumaticBrainInjury': [traumatic_brain_injury],
        'Hypertension': [hypertension],
        'Diabetes': [diabetes],
        'Depression': [depression],
        'Stroke': [stroke],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'CholesterolTotal': [cholesterol_total],
        'CholesterolLDL': [cholesterol_ldl],
        'CholesterolHDL': [cholesterol_hdl],
        'CholesterolTriglycerides': [cholesterol_triglycerides],
        'UPDRS': [updrs],
        'MoCA': [moca],
        'FunctionalAssessment': [functional_assessment],
        'Tremor': [tremor],
        'Rigidity': [rigidity],
        'Bradykinesia': [bradykinesia],
        'PosturalInstability': [postural_instability],
        'SpeechProblems': [speech_problems],
        'SleepDisorders': [sleep_disorders],
        'Constipation': [constipation]
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            st.write(f'Prediction for {patient_name}: {"Positive" if prediction[0] == 1 else "Negative"}')
            st.write(f'Probability of Positive Diagnosis: {probability:.2f}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot Positive/Negative probability
            sns.barplot(x=['Negative', 'Positive'], y=[1 - probability, probability], ax=axes[0], palette=['red', 'green'])
            axes[0].set_title('Positive/Negative Probability')
            axes[0].set_ylabel('Probability')

            # Plot Positive/Negative pie chart
            axes[2].pie([1 - probability, probability], labels=['Negative', 'Positive'], autopct='%1.1f%%', colors=['red', 'green'])
            axes[2].set_title('Positive/Negative Pie Chart')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.error(f"{patient_name} is likely to have Parkinsons. Consider consulting a medical professional.")
            else:
                st.success(f"{patient_name} is likely not to have Parkinsons. However, it's always a good idea to maintain a healthy lifestyle.")

if __name__ == '__main__':
    main()
