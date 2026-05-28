# HuggingFace-Pipeline-practice-project

![HuggingFace pipeline banner](assets/readme/hf-pipeline-banner.svg)

> Hugging Face `pipeline` 기반으로 의료 텍스트 예제를 실습하는 Streamlit 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 Hugging Face `transformers`의 `pipeline` API를 활용해, 비교적 짧은 코드로 텍스트 생성과 문서 요약 기능을 빠르게 실험해 보는 데 목적이 있습니다.

현재 저장소에는 두 가지 데모 앱이 포함되어 있습니다.

- `app.py`: 증상 입력을 바탕으로 가능한 질환 후보를 생성하는 간단한 예제
- `medical.app.py`: 긴 의료 문서를 요약하는 예제

두 앱 모두 학습용·시연용 프로젝트이며, 실제 의료 진단 도구로 사용하면 안 됩니다.

## 포함된 데모

### 1. AI Medical Symptom Checker

`app.py`는 `google/flan-t5-base` 모델을 이용해 사용자가 입력한 증상을 바탕으로 가능한 질환 목록을 생성합니다.

- 입력: 사용자가 직접 입력한 증상 문장
- 출력: 가능한 질환 5개 제안
- 특징: 구성은 단순하지만, 프롬프트 기반 생성 실습에 적합

### 2. AI Medical Document Summarizer

`medical.app.py`는 `Falconsai/medical_summarization` 모델을 활용해 비교적 긴 영어 의료 문서를 요약합니다.

- 입력: 의료 메모 또는 진료 기록 형태의 텍스트
- 출력: 핵심 내용을 요약한 짧은 문장
- 특징: GPU 사용 가능 시 자동으로 활용하고, Streamlit UI로 바로 확인 가능

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 증상 분석 앱 실행

```bash
streamlit run app.py
```

### 3. 의료 문서 요약 앱 실행

```bash
streamlit run medical.app.py
```

## 사용 기술

- Python
- Streamlit
- Hugging Face Transformers
- Torch

## 주의 사항

- 이 프로젝트는 학습 및 데모 목적의 예제입니다.
- 출력 결과는 참고용이며, 의료적 판단이나 진단을 대신하지 않습니다.
- 모델 최초 실행 시 다운로드로 인해 시간이 걸릴 수 있습니다.

## 한 줄 요약

> Hugging Face pipeline을 이용해 의료 텍스트 생성과 요약 흐름을 가볍게 실습할 수 있도록 구성한 Streamlit 예제 프로젝트입니다.
