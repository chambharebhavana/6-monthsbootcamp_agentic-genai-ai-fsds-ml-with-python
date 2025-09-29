import streamlit as st 
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r'C:\Users\chamb\python classwork 2025\Machine Learning\Simple Linear Regression\linear_regression_model.pkl', 'rb'))

# set the title of the streamlit app
st.title("Salary Prediction App")

# Add a brief description
st.write("This app predicts the salary based on years of experience using a simple linear regression model.")

# Add input widget for user to enter years of expperience
years_experience = st.number_input("Enter Years of Experience: ", min_value = 0.0,max_value =50.0,  value = 1.0, step= 0.5)

#when the button is clicked, make predictions
if st.button("Predict Salary"):
    # Make a prediction using the trained model
    experience_input = np.array([[years_experience]]) #convert the input to 2D array for prediction
    prediction = model.predict(experience_input)
    
    # Display the result
    st.success(f"The Predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}")
    
# Display information about the model
st.write("This model was trained using a dataset of salaries and years of experience.")
    