# app.py

import streamlit as st
import document_processor
import json
import traceback

st.set_page_config(page_title="구조 분석 안정화", page_icon="🛡️", layout="wide")
st.title("🛡️ 구조 분석 안정화 테스트 (v2.2)")
st.write("LLM 기반 구조 추출의 안정성을 검증하고, LLM의 원본 응답을 상세히 추적합니다.")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("구조 추출 실행"):
        document_text = uploaded_file.getvalue().decode('utf-8')
        st.session_state.analysis_result = None
        
        with st.status("분석을 시작합니다...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                # ✅ document_processor.py에 정의된 함수 이름과 정확히 일치합니다.
                final_result, debug_info = document_processor.run_extraction_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status
                )
                st.session_state.analysis_result = {
                    "final": final_result,
                    "debug": debug_info,
                    "file_name": uploaded_file.name
                }
                status.update(label="✅ 구조 추출 완료!", state="complete", expanded=False)
                st.success("🎉 평탄화된 구조 추출이 성공적으로 완료되었습니다!")
            except Exception as e:
                st.session_state.analysis_result = None
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")
                st.code(traceback.format_exc())

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    final_result_data = result["final"]
    debug_info = result["debug"]
    file_name = result["file_name"]
    
    st.subheader("📋 결과 확인 및 다운로드")
    
    if "error" in final_result_data:
        st.error(f"**분석 실패:** {final_result_data['error']}")
    
    with st.expander("추출된 구조 미리보기 (JSON)", expanded=False):
        st.json(final_result_data)
    
    with st.expander("디버그 로그 미리보기 (JSON)", expanded=True):
        st.json({"llm_responses": debug_info})

    col1_dl, col2_dl = st.columns(2)
    with col1_dl:
        st.download_button(
           label="✔️ 추출된 구조(JSON) 다운로드",
           data=json.dumps(final_result_data, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_flat_structure.json",
           mime="application/json",
        )
    with col2_dl:
        st.download_button(
           label="🐞 디버그 로그(JSON) 다운로드",
           data=json.dumps(debug_info, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_debug.json",
           mime="application/json",
        )