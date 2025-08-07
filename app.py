# app.py

import streamlit as st
import document_processor # 우리가 만든 로직 파일을 불러옵니다.

# --- 화면 UI 구성 ---
st.title("🇹🇭 태국 법률 문서 분석기 v0.2") # 버전 업!
st.write("텍스트 파일을 업로드하면 Gemini AI가 계층 구조로 분석하고 요약합니다.")

uploaded_file = st.file_uploader("분석할 .txt 파일을 선택하세요.", type=['txt'])

# --- 로직 실행 부분 ---
if uploaded_file is not None:
    if st.button("분석 시작하기"):
        with st.spinner('문서를 분석 중입니다. Gemini AI를 호출하고 있습니다...'):
            try:
                document_text = uploaded_file.getvalue().decode("utf-8")
                
                # --- 여기가 핵심! 실제 파이프라인 함수를 호출합니다. ---
                # Secrets에서 API 키를 안전하게 가져와서 전달합니다.
                api_key = st.secrets["GEMINI_API_KEY"]
                final_json = document_processor.run_pipeline(document_text, api_key=api_key)
                # ----------------------------------------------------

                st.subheader("✅ 분석 결과")
                st.json(final_json)

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.error("API 키가 정확한지, Streamlit Cloud의 Secrets에 잘 설정되었는지 확인해보세요.")