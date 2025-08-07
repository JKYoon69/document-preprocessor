# app.py

import streamlit as st
import document_processor

st.set_page_config(page_title="태국 법률 문서 분석기", page_icon="🇹🇭", layout="wide")
st.title("🇹🇭 태국 법률 문서 분석기 v1.1")
st.write("텍스트 파일을 업로드하면 Gemini 2.5 Flash-lite 모델이 문서를 분석합니다.")

uploaded_file = st.file_uploader("분석할 .txt 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("분석 시작하기"):
        with st.status("분석을 시작합니다...", expanded=True) as status:
            try:
                document_text = uploaded_file.getvalue().decode("utf-8")
                api_key = st.secrets["GEMINI_API_KEY"]
                
                final_json = document_processor.run_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status
                )
                
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)
                
                # ✅ 최종 결과와 디버그 정보 분리해서 표시
                st.subheader("📊 최종 분석 결과")
                
                # 디버그 정보가 있다면 먼저 보여줌
                if "debug_info" in final_json:
                    with st.expander("🔍 디버깅 정보 보기"):
                        st.json(final_json["debug_info"])
                
                # 실제 결과 표시 (디버그 정보 제외)
                display_json = {k: v for k, v in final_json.items() if k != 'debug_info'}
                st.json(display_json)

            except Exception as e:
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")