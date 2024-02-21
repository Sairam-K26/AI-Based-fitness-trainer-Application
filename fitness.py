import streamlit as st
from langchain.llms import CTransformers
from langchain import prompts, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load the locally stored llama-2-13b-chat.ggmlv3.q4_0.bin model
llama_model_path = 'llama-2-13b-chat.ggmlv3.q4_0.bin'

# Initialize LangChain model
llm = CTransformers(model=llama_model_path, model_type='llama', callbacks=[StreamingStdOutCallbackHandler()])
template = """
[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. Your answers are always brief. and add a role as a chatbot yourself.
<</SYS>>
{text}[/INST]
"""
prompt = prompts.PromptTemplate(template=template, input_variables=["text"])
llm_chain = chains.LLMChain(prompt=prompt, llm=llm)

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

# Calculate BMI
bmi = weight / ((height / 100) ** 2)

# Submit button
if st.sidebar.button('Get Recommendations'):
    bmi_category = ""
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal weight"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
        
    st.write(f'**BMI Value:** {bmi:.2f}')
    st.write(f'**BMI Category:** {bmi_category}')

    # Use the model to generate workout plan
    input_text = f"Provide a comprehensive and detailed workout plan tailored for a {fitness_goal.lower()} goal over the course of {time_period} days. Please include specific exercises, sets, reps, rest intervals, and any additional recommendations for optimal results."
    workout_plan = llm_chain.run(input_text)
    st.subheader('Workout Plan')
    st.write(workout_plan)

    # Use the model to generate diet plan
    input_text = f"Provide a comprehensive and detailed diet plan tailored for a {fitness_goal.lower()} goal over the course of {time_period} days. Please include specific meal options, portion sizes, macronutrient breakdowns, meal timings, and any additional recommendations for optimal nutrition and progress towards the desired fitness goal."
    diet_plan = llm_chain.run(input_text)
    st.subheader('Diet Plan')
    st.write(diet_plan)

    # Calculate daily workout time for the entire period
    if fitness_goal == 'Weight Loss':
        workout_time_per_day = 60  # 30 minutes of cardio + 30 minutes of strength training
    elif fitness_goal == 'Weight Gain':
        workout_time_per_day = 60  # 45 minutes of strength training + 15 minutes of HIIT
    total_workout_time = workout_time_per_day * time_period
    st.write(f'**Total Workout Time for {time_period} days:** {total_workout_time:.2f} minutes')
