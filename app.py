# app.py

import streamlit as st
import document_processor
import json
import traceback

st.set_page_config(page_title="RAG 전처리기", page_icon="⚖️", layout="wide")
st.title("⚖️ 태국 법률 문서 RAG 전처리기 (최종 버전)")
st.write("문서를 의미/계층적으로 분석하고, 재귀적으로 요약하여 최종 JSON 결과물을 생성합니다.")

if 'final_result' not in st.session_state:
    st.session_state.final_result = None

uploaded_file = st.file_uploader("분석할 .txt 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("전체 파이프라인 실행"):
        st.session_state.final_result = None
        document_text = uploaded_file.getvalue().decode('utf-8')
        
        with st.status("전체 분석을 시작합니다...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                # ✅✅✅ 'uploaded_file' 대신 'file_name'으로 파일 이름을 전달 ✅✅✅
                final_json_result = document_processor.run_final_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status,
                    file_name=uploaded_file.name 
                )
                st.session_state.final_result = {
                    "data": final_json_result,
                    "file_name": uploaded_file.name
                }
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)
                st.success("🎉 모든 작업이 성공적으로 완료되었습니다!")

            except Exception as e:
                st.session_state.final_result = None
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")
                st.code(traceback.format_exc())

if st.session_state.final_result:
    result = st.session_state.final_result["data"]
    file_name = st.session_state.final_result["file_name"]
    
    st.subheader("📊 최종 분석 결과 미리보기")
    st.json(result, expanded=False)

    st.download_button(
       label="⬇️ 최종 결과(JSON) 다운로드",
       data=json.dumps(result, indent=2, ensure_ascii=False),
       file_name=f"{file_name.split('.')[0]}_final_processed.json",
       mime="application/json",
    )