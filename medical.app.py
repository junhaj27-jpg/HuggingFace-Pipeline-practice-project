import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="AI Medical Summarizer", page_icon="AI", layout="wide")


@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="Falconsai/medical_summarization", device=device)


with st.spinner("Loading the medical summarization model..."):
    medical_ai = load_model()

st.title("AI Medical Document Summarizer")
st.markdown(
    """
This app summarizes longer English medical notes for study and demonstration.
It is not a diagnostic tool.
---
"""
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    sample_text = (
        "The patient is a 72-year-old female with a history of hypertension and "
        "chronic kidney disease. She presents with increased swelling in her lower "
        "extremities and shortness of breath over the past week. Laboratory results "
        "indicate elevated creatinine levels."
    )
    user_input = st.text_area("Enter medical text to summarize:", value=sample_text, height=300)
    btn_analyze = st.button("Analyze")

with col2:
    st.subheader("Output")
    if btn_analyze:
        if user_input.strip() == "":
            st.error("Please enter text.")
        else:
            with st.spinner("Summarizing..."):
                result = medical_ai(user_input, max_length=60, min_length=20, do_sample=False)
                summary_text = result[0]["summary_text"]
                st.success("Summary complete.")
                st.info(f"**Summary**\n\n{summary_text}")
                st.warning(
                    "This AI output is for educational purposes only and is not a medical diagnosis."
                )
    else:
        st.write("Run analysis from the left panel.")

st.sidebar.title("About Project")
st.sidebar.info("Streamlit app using HuggingFace medical summarization models.")
