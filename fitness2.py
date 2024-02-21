import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def calculate_bmi(weight, height):
    bmi = weight / ((height / 100) ** 2)
    return bmi

def classify_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi >= 18.5 and bmi < 25:
        return "Normal weight"
    elif bmi >= 25 and bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def classify_weight_goal(weight_goal, gender, age):
    # Here you can implement classification logic based on gender and age
    # For the sake of this example, let's just provide generic tips
    if weight_goal == 'Weight Loss':
        return "For weight loss, focus on a balanced diet and regular exercise."
    elif weight_goal == 'Weight Gain':
        return "To gain weight, increase your calorie intake and focus on strength training exercises."

# Title of the application
st.title('AI Fitness Trainer')

# Sidebar section for user input
st.sidebar.header('User Profile')
name = st.sidebar.text_input('Name')
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=25)
weight = st.sidebar.number_input('Weight (kg)', min_value=20.0, max_value=500.0, value=70.0)
height = st.sidebar.number_input('Height (cm)', min_value=100.0, max_value=300.0, value=170.0)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
fitness_goal = st.sidebar.selectbox('Fitness Goal', ['Weight Loss', 'Weight Gain'])

# Time period for the goal
time_period = st.sidebar.number_input('Time Period (days)', min_value=1, max_value=365, value=30)

# Submit button
if st.sidebar.button('Get Recommendations'):
    # Calculate BMI
    bmi = calculate_bmi(weight, height)
    bmi_category = classify_bmi(bmi)
    st.write(f'**BMI Value:** {bmi:.2f}')
    st.write(f'**BMI Category:** {bmi_category}')

    # Classify weight goal and provide tips
    weight_goal_tips = classify_weight_goal(fitness_goal, gender, age)
    st.write(f'**Weight Goal Tips:** {weight_goal_tips}')

    # Use GPT-2 to generate workout plan
    input_text = f"Generate workout plan for a {fitness_goal.lower()} goal for {time_period} days"
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    output_ids = gpt2_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    workout_plan = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    st.subheader('Workout Plan')
    st.write(workout_plan)

    # Use GPT-2 to generate diet plan
    input_text = f"Generate diet plan for a {fitness_goal.lower()} goal for {time_period} days"
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    output_ids = gpt2_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    diet_plan = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    st.subheader('Diet Plan')
    st.write(diet_plan)

    # Calculate daily workout time
    if fitness_goal == 'Weight Loss':
        workout_time_per_day = 60  # 30 minutes of cardio + 30 minutes of strength training
    elif fitness_goal == 'Weight Gain':
        workout_time_per_day = 60  # 45 minutes of strength training + 15 minutes of HIIT
    workout_time_per_day = workout_time_per_day / time_period
    st.write(f'**Daily Workout Time:** {workout_time_per_day:.2f} minutes')
