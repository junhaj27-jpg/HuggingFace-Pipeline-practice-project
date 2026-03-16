import streamlit as st
from transformers import pipeline
import torch

# 1. 페이지 설정
st.set_page_config(page_title="AI Medical Summarizer", page_icon="🩺", layout="wide")

# 2. 모델 로드 (캐싱을 사용하여 앱 속도 향상)
@st.cache_resource
def load_model():
    # 의료 문서 요약에 특화된 T5 모델 파이프라인
    # device=0은 GPU 사용, -1은 CPU 사용입니다.
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="Falconsai/medical_summarization", device=device)
    return summarizer

# 모델 불러오기
with st.spinner("의료 AI 모델을 로드 중입니다... 잠시만 기다려주세요."):
    medical_ai = load_model()

# 3. UI 구성
st.title("🩺 AI 의료 문서 요약 시스템")
st.markdown("""
이 시스템은 복잡한 **의료 진단서나 환자 상담 기록**을 핵심 내용으로 요약해줍니다.
현재 영문 의료 텍스트에 최적화되어 있습니다.
---
""")

# 레이아웃 나누기
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("입력창 (Input)")
    sample_text = "The patient is a 72-year-old female with a history of hypertension and chronic kidney disease. She presents with increased swelling in her lower extremities and shortness of breath over the past week. Laboratory results indicate elevated creatinine levels."
    user_input = st.text_area("분석할 의료 텍스트를 입력하세요:", value=sample_text, height=300)
    
    btn_analyze = st.button("AI 분석 실행 🚀")

with col2:
    st.subheader("분석 결과 (Output)")
    if btn_analyze:
        if user_input.strip() == "":
            st.error("텍스트를 입력해주세요!")
        else:
            with st.spinner("데이터를 분석하고 요약하는 중..."):
                # 요약 실행
                # max_length와 min_length는 모델 특성에 맞춰 조절 가능합니다.
                result = medical_ai(user_input, max_length=60, min_length=20, do_sample=False)
                summary_text = result[0]['summary_text']
                
                # 결과 출력
                st.success("요약이 완료되었습니다!")
                st.info(f"**[요약 내용]**\n\n{summary_text}")
                
                # 가이드라인 안내
                st.warning("⚠️ **주의:** 본 결과는 인공지능에 의해 생성되었으며, 실제 진단 시에는 반드시 전문 의료진과 상의하십시오.")
    else:
        st.write("왼쪽에서 분석 실행 버튼을 눌러주세요.")

# 하단 정보
st.sidebar.title("About Project")
st.sidebar.info("HuggingFace의 `medical_summarization` 모델을 활용한 Streamlit 앱입니다.")