# app.py
import streamlit as st
import document_processor as dp
import json
import traceback
import pandas as pd

# --- 페이지 설정 ---
st.set_page_config(page_title="법률 문서 계층 분석기", page_icon="🏛️", layout="wide")

# --- 세션 상태 초기화 ---
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- UI 레이아웃 ---
st.title("🏛️ 태국 법률 문서 계층 분석기 (3-Step Pipeline Ver.)")
st.markdown(f"**LLM Model:** `{dp.MODEL_NAME}` (수정은 `document_processor.py`에서 가능)")
st.markdown("LLM에 대한 3단계 순차적 임무 부여를 통해 법률 문서의 계층 구조를 정밀하게 분석합니다.")

# --- 프롬프트 수정 UI ---
with st.expander("⚙️ 각 단계별 프롬프트 수정하기"):
    tab1, tab2, tab3 = st.tabs(["1단계: Architect", "2단계: Surveyor", "3단계: Detailer"])

    with tab1:
        st.info("문서 전체에서 최상위 구조(Book, Part, Chapter)를 찾는 임무를 정의합니다.")
        prompt_architect = st.text_area(
            "Architect Prompt", value=dp.PROMPT_ARCHITECT, height=300, key="prompt1"
        )
    with tab2:
        st.info("각 Chapter 내부에서 중간 구조(Section)를 찾는 임무를 정의합니다.")
        prompt_surveyor = st.text_area(
            "Surveyor Prompt", value=dp.PROMPT_SURVEYOR, height=300, key="prompt2"
        )
    with tab3:
        st.info("가장 작은 단위(Section 또는 Chapter) 내부에서 최하위 구조(Article)를 찾는 임무를 정의합니다.")
        prompt_detailer = st.text_area(
            "Detailer Prompt", value=dp.PROMPT_DETAILER, height=300, key="prompt3"
        )

uploaded_file = st.file_uploader("분석할 태국 법률 텍스트 파일(.txt)을 업로드하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("계층 구조 분석 실행", type="primary"):
        document_text = uploaded_file.getvalue().decode('utf-8')
        st.session_state.analysis_result = None
        
        with st.status("3단계 분석 파이프라인을 시작합니다...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                final_result, debug_info = dp.run_pipeline(
                    document_text=document_text,
                    api_key=api_key,
                    status_container=status,
                    prompt_architect=prompt_architect,
                    prompt_surveyor=prompt_surveyor,
                    prompt_detailer=prompt_detailer
                )
                st.session_state.analysis_result = {
                    "final": final_result,
                    "debug": debug_info,
                    "file_name": uploaded_file.name
                }
                status.update(label="✅ 계층 분석 완료!", state="complete", expanded=False)
                st.success("🎉 성공적으로 계층 트리 구조를 생성했습니다!")

            except Exception as e:
                st.session_state.analysis_result = None
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")
                st.code(traceback.format_exc())

# --- 결과 표시 ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    final_result_data = result["final"]
    debug_info = result["debug"]
    file_name = result["file_name"]

    st.header("📄 분석 결과 확인 및 다운로드")

    if "error" in final_result_data or not final_result_data.get("tree"):
        st.error("분석 실패: 문서에서 유효한 구조를 추출하지 못했습니다. 아래 디버그 로그를 확인해주세요.")
    
    tab1, tab2 = st.tabs(["✔️ 최종 결과 (계층 트리)", "🐞 상세 디버그 로그"])

    with tab1:
        st.json(final_result_data, expanded=True)
        st.download_button(
           label="결과 트리 (JSON) 다운로드",
           data=json.dumps(final_result_data, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_pipeline_tree.json",
           mime="application/json",
        )

    with tab2:
        st.write("파이프라인 각 단계에서 LLM이 반환한 원본 데이터 등 상세한 로그를 포함합니다.")
        st.json({"pipeline_logs": debug_info}, expanded=False)
        st.download_button(
           label="디버그 로그 (JSON) 다운로드",
           data=json.dumps(debug_info, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_pipeline_debug.json",
           mime="application/json",
        )