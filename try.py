import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'D:\\Parkinson\\ML_MODEL\\logistic_regression_model_blabla.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title("Parkinson's Disease Prediction")

    # Add a description
    st.write('Enter Information')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Patient Information')

        # Add input fields for features
        age = st.slider('Age', 18, 100, 65)
        gender = st.selectbox('Gender', ['Female', 'Male'])
        ethnicity = st.selectbox('Ethnicity', ['Asian', 'Black', 'Hispanic', 'White'])
        education_level = st.selectbox('Education Level', ['High School', 'Some College', 'College Graduate', 'Post Graduate'])
        bmi = st.slider('BMI', 15.0, 40.0, 25.0)
        smoking = st.selectbox('Smoking', ['Never', 'Former', 'Current'])
        alcohol_consumption = st.slider('Alcohol Consumption (0-10)', 0, 10, 5)
        physical_activity = st.slider('Physical Activity (0-10)', 0, 10, 5)
        diet_quality = st.slider('Diet Quality (0-10)', 0, 10, 5)
        sleep_quality = st.slider('Sleep Quality (0-10)', 0, 10, 5)
        family_history_parkinsons = st.selectbox('Family History of Parkinson\'s', ['No', 'Yes'])
        traumatic_brain_injury = st.selectbox('Traumatic Brain Injury', ['No', 'Yes'])
        hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
        diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
        depression = st.selectbox('Depression', ['No', 'Yes'])
        stroke = st.selectbox('Stroke', ['No', 'Yes'])
        systolic_bp = st.slider('Systolic Blood Pressure', 90, 200, 120)
        diastolic_bp = st.slider('Diastolic Blood Pressure', 60, 120, 80)
        cholesterol_total = st.slider('Total Cholesterol', 100, 300, 200)
        cholesterol_ldl = st.slider('LDL Cholesterol', 50, 200, 100)
        cholesterol_hdl = st.slider('HDL Cholesterol', 20, 100, 50)
        cholesterol_triglycerides = st.slider('Triglycerides', 50, 500, 150)
        updrs = st.slider('UPDRS', 0, 100, 50)
        moca = st.slider('MoCA Score', 0, 30, 24)
        functional_assessment = st.slider('Functional Assessment (0-10)', 0, 10, 5)
        tremor = st.selectbox('Tremor', ['No', 'Yes'])
        rigidity = st.selectbox('Rigidity', ['No', 'Yes'])
        bradykinesia = st.selectbox('Bradykinesia', ['No', 'Yes'])
        postural_instability = st.selectbox('Postural Instability', ['No', 'Yes'])
        speech_problems = st.selectbox('Speech Problems', ['No', 'Yes'])
        sleep_disorders = st.selectbox('Sleep Disorders', ['No', 'Yes'])
        constipation = st.selectbox('Constipation', ['No', 'Yes'])

    # Convert categorical inputs to numerical
    gender = 1 if gender == 'Female' else 0
    ethnicity = {'Asian': 0, 'Black': 1, 'Hispanic': 2, 'White': 3}.get(ethnicity, 0)
    education_level = {'High School': 0, 'Some College': 1, 'College Graduate': 2, 'Post Graduate': 3}.get(education_level, 0)
    smoking = {'Never': 0, 'Former': 1, 'Current': 2}.get(smoking, 0)
    family_history_parkinsons = 1 if family_history_parkinsons == 'Yes' else 0
    traumatic_brain_injury = 1 if traumatic_brain_injury == 'Yes' else 0
    hypertension = 1 if hypertension == 'Yes' else 0
    diabetes = 1 if diabetes == 'Yes' else 0
    depression = 1 if depression == 'Yes' else 0
    stroke = 1 if stroke == 'Yes' else 0
    tremor = 1 if tremor == 'Yes' else 0
    rigidity = 1 if rigidity == 'Yes' else 0
    bradykinesia = 1 if bradykinesia == 'Yes' else 0
    postural_instability = 1 if postural_instability == 'Yes' else 0
    speech_problems = 1 if speech_problems == 'Yes' else 0
    sleep_disorders = 1 if sleep_disorders == 'Yes' else 0
    constipation = 1 if constipation == 'Yes' else 0

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
    missing_columns = set(expected_columns) - set(input_data.columns)
    if missing_columns:
        st.error(f"The following expected columns are missing in the input data: {missing_columns}")
        st.stop()

    input_data = input_data[expected_columns]

       # Ensure columns are in the same order as during model training
    try:
        input_data = input_data[expected_columns]
    except KeyError as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # Log input data for debugging
    st.write("Input Data:")
    st.write(input_data)

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            try:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.stop()
            
            st.write(f'Prediction for Parkinsons Disease: {"True" if prediction[0] == 1 else "False"}')
            st.write(f'Probability of Having disease: {probability:.2f}')
            #st.write(f'Overall Score: {overall_score}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot Hit/Fail probability
            sns.barplot(x=['Flop', 'Hit'], y=[1 - probability, probability], ax=axes[0], palette=['red', 'green'])
            axes[0].set_title('Flop/Hit Probability')
            axes[0].set_ylabel('Probability')

            # Plot GPA distribution
            #sns.histplot(input_data['overall_score'], kde=True, ax=axes[1])
            #axes[1].set_title('Overall Distribution')

            # Plot Pass/Fail pie chart
            #axes[2].pie([1 - probability, probability], labels=['Not Diagnosed', 'Diagnosed'], autopct='%1.1f%%', colors=['red', 'green'])
            #axes[2].set_title(' Pie Chart')

            # Display the plots
            #st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.success(f"You are likely to be have Pakinson disease.")
            else:
                st.error(f"You are not likely to be have Pakinson disease.")

if __name__ == '__main__':
    main()


           
