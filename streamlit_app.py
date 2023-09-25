import pandas as pd
import streamlit as st
import pickle
from PIL import Image

# Define a custom CSS style to set the background color to red
custom_css = """
<style>
body {
    background-color: #ff0000; /* Red color */
}
</style>
"""

# Use st.markdown to inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


st.write("""
    # OLA Drivers Churn Prediction Model
    """)

st.image(Image.open('./artefacts/ola.jpg'))

col1, col2, col3 = st.columns(3)

city = col1.selectbox("Select City",
        ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 
        'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29'
        ])

education = col1.selectbox("Select Education Level", [0, 1, 2])

income = col1.number_input("Enter Income", value=0, format='%d')

business_val = col1.number_input("Enter Total Business Value", value=0, format='%d')

gender = col2.selectbox("Select Gender", [0, 1])

join_dsgn = col2.selectbox("Select Joining Designation", [1, 2, 3, 4, 5])

duration = col2.number_input("Enter Duration", value=1, format='%d')

inc_flag = col2.selectbox("Select Income Flag", [0, 1])

age = col3.number_input("Enter Age", value=0, format='%d')

grade = col3.selectbox("Select Joining Grade", [1, 2, 3, 4, 5])

qtr_rat = col3.selectbox("Select Last Quarterly Rating", [1, 2, 3,4])

qr_flag = col3.selectbox("Select Quarterly Rating Flag", [0, 1])


if(st.button("Predict Churn")):

    input_features = [[city, education, income, age, gender, join_dsgn,
                    grade, business_val, qtr_rat, inc_flag, qr_flag, duration]]

    df_columns = ['City', 'Education_Level', 'Income', 'Age', 'Gender', 'Joining Designation', 
                'Grade', 'Total Business Value','Quarterly Rating', 'Inc_Flag', 'QR_Flag', 'Duration']

    input_features = pd.DataFrame(data = input_features, columns= df_columns)

    model = pickle.load(open('./artefacts/classifier.pkl', 'rb'))

    churn = model.predict(input_features)

    if churn == 1:
         st.write("""
        ### The driver is going to Churn.
        """)
    else:
         st.write("""
        ### The driver is not going to Churn.
        """)