# creating a simple BMI calculator using streamlit python
import streamlit as st

# 1. Show a title
st.title("BMI Calculator")
st.write("Enter your weight and height to calculate your Body Mass Index (BMI).")

# 2. Ask for weight
weight = st.number_input(
    "Enter your weight (kg)",
    min_value=1.0,
    max_value=300.0,
    value=60.0
)

# 3. Ask for height
height = st.number_input(
    "Enter your height (meters)",
    min_value=0.5,
    max_value=7.5,
    value=1.70
)

# 4. Calculate button
if st.button("Calculate BMI"):

    # Calculate BMI
    bmi = weight / (height ** 2)

    # Show BMI result
    st.success(f"Your BMI is: {bmi:.2f}")

    # 5. Check BMI category
    if bmi < 18.5:
        st.warning("Your BMI is below average (Underweight).")

    elif bmi >= 18.5 and bmi < 25:
        st.success("Your BMI is in the safe/healthy range.")

    elif bmi >= 25 and bmi < 30:
        st.warning("Your BMI is above average (Overweight).")

    else:
        st.error("Your BMI is significantly above average (Obesity range).")