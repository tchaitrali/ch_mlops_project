import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
# Assuming the model is saved as 'best_model.joblib' in the same directory as app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")
model = joblib.load(MODEL_PATH)

st.title("Wellness Tourism Package Purchase Predictor")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Create input widgets for each feature
with st.form("prediction_form"):
    st.header("Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        typeofcontact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Business', 'Other', 'Large Business'])
        gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male'])
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5, 0]) # 0 for no preference or not applicable

    with col2:
        maritalstatus = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced', 'Unmarried'])
        numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
        passport = st.checkbox("Has Passport", value=False)
        owncar = st.checkbox("Owns Car", value=False)
        numberofchildrenvisiting = st.number_input("Number of Children Visiting (Age < 5)", min_value=0, max_value=5, value=0)
        designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director', 'Junior Executive', 'President', 'Chief Executive'])
        monthlyincome = st.number_input("Monthly Income", min_value=0, value=30000)

    st.header("Interaction Data")
    col3, col4 = st.columns(2)
    with col3:
        pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
        productpitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Premium'])
    with col4:
        numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Create a DataFrame from inputs
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [typeofcontact],
            'CityTier': [citytier],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [numberofpersonvisiting],
            'PreferredPropertyStar': [preferredpropertystar],
            'MaritalStatus': [maritalstatus],
            'NumberOfTrips': [numberoftrips],
            'Passport': [1 if passport else 0],
            'OwnCar': [1 if owncar else 0],
            'NumberOfChildrenVisiting': [numberofchildrenvisiting],
            'Designation': [designation],
            'MonthlyIncome': [monthlyincome],
            'PitchSatisfactionScore': [pitchsatisfactionscore],
            'ProductPitched': [productpitched],
            'NumberOfFollowups': [numberoffollowups],
            'DurationOfPitch': [durationofpitch]
        })

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"The customer is likely to purchase the Wellness Tourism Package! (Probability: {prediction_proba[0]:.2f})")
        else:
            st.warning(f"The customer is not likely to purchase the Wellness Tourism Package. (Probability: {prediction_proba[0]:.2f})")
