# app.py

import streamlit as st
import document_processor
import json
import traceback
import pandas as pd

st.set_page_config(page_title="계층적 분석기", page_icon="📚", layout="wide")
st.title("📚 태국 법률 문서 계층적 분석기 (1단계: Chapter 추출)")
st.write("문서를 의미 기반으로 분할하고, 슬라이딩 윈도우 방식으로 Chapter 정보를 추출합니다.")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("Chapter 분석 실행"):
        document_text = uploaded_file.getvalue().decode('utf-8')
        st.session_state.analysis_result = None
        
        with st.status("분석을 시작합니다...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                final_json_result = document_processor.run_hierarchical_pipeline_step1(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status
                )
                st.session_state.analysis_result = {
                    "data": final_json_result,
                    "file_name": uploaded_file.name
                }
                status.update(label="✅ Chapter 분석 완료!", state="complete", expanded=False)
                st.success("🎉 Chapter 정보 추출이 성공적으로 완료되었습니다!")

            except Exception as e:
                st.session_state.analysis_result = None
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")
                st.code(traceback.format_exc())

# 결과가 있으면 화면에 표시
if st.session_state.analysis_result:
    result = st.session_state.analysis_result["data"]
    file_name = st.session_state.analysis_result["file_name"]
    
    st.subheader("📊 분석 결과 요약")
    st.write("**전체 문서 요약:**")
    st.info(result.get("global_summary", "요약 정보 없음"))

    st.write("**추출된 Chapter 목록:**")
    
    # Chapter 정보를 표로 변환하여 표시
    chapters = result.get("chapters", [])
    if chapters:
        display_data = []
        for i, chap in enumerate(chapters):
            display_data.append({
                "No.": i,
                "Title": chap.get("title"),
                "Summary": chap.get("summary"),
                "Global Start": chap.get("global_start"),
                "Global End": chap.get("global_end")
            })
        st.table(pd.DataFrame(display_data).set_index("No."))
    else:
        st.warning("추출된 Chapter가 없습니다.")

    st.subheader("📋 결과 다운로드")
    st.download_button(
       label="✔️ Chapter 분석 결과(JSON) 다운로드",
       data=json.dumps(result, indent=2, ensure_ascii=False),
       file_name=f"{file_name.split('.')[0]}_chapters.json",
       mime="application/json",
    )