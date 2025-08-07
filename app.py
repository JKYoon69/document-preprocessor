# app.py

import streamlit as st
import document_processor

st.set_page_config(page_title="태국 법률 문서 분석기", page_icon="🇹🇭", layout="wide")
st.title("🇹🇭 태국 법률 문서 분석기 v1.0")
st.write("텍스트 파일을 업로드하면 Gemini 2.5 Flash-lite 모델이 계층 구조로 분석하고 요약합니다.")

uploaded_file = st.file_uploader("분석할 .txt 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("분석 시작하기"):
        with st.status("분석을 시작합니다...", expanded=True) as status:
            try:
                status.write("파일을 읽고 텍스트를 준비합니다...")
                document_text = uploaded_file.getvalue().decode("utf-8")
                
                if not document_text.strip():
                    status.update(label="오류", state="error", expanded=True)
                    st.error("업로드된 파일에 내용이 없습니다.")
                else:
                    api_key = st.secrets["GEMINI_API_KEY"]
                    final_json = document_processor.run_pipeline(
                        document_text=document_text, 
                        api_key=api_key,
                        status_container=status
                    )
                    status.update(label="✅ 분석 완료!", state="complete", expanded=False)
                    st.subheader("✅ 최종 분석 결과")
                    st.json(final_json)

            except Exception as e:
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")
                st.error("Secrets에 API 키가 올바르게 설정되었는지 확인해주세요.")