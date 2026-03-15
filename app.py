import streamlit as st
from transformers import pipeline

st.title("🩺 AI Medical Symptom Checker")

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_model()

symptom = st.text_input("Enter your symptom (example: headache, fever)")

if st.button("Analyze Symptom"):

    if symptom == "":
        st.warning("Please enter a symptom")

    else:
        prompt = f"""
        You are a medical assistant.

        Patient symptoms: {symptom}

        List 5 possible illnesses related to these symptoms.
        Answer as a numbered list.
        """

        result = generator(
            prompt,
            max_new_tokens=80,
            temperature=0.7
        )

        st.subheader("Possible illnesses")
        st.write(result[0]["generated_text"].strip())

        st.info("This AI is for educational purposes only.")