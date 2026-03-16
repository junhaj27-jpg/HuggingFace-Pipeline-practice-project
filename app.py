import streamlit as st
from transformers import pipeline

st.title("🩺 AI Medical Symptom Checker")

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_model()

symptom = st.text_input("Enter your symptom")

if st.button("Analyze"):

    if symptom == "":
        st.warning("Please enter a symptom")

    else:
        prompt = f"""
        Patient symptom: {symptom}
        List 5 possible illnesses.
        """

        result = generator(prompt, max_new_tokens=80)

        st.write(result[0]['generated_text'])

        st.info("This is not a medical diagnosis.")