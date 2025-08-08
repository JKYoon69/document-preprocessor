# app.py
import streamlit as st
import document_processor
import json
import traceback
import pandas as pd

# --- 페이지 설정 ---
st.set_page_config(page_title="법률 문서 계층 분석기", page_icon="🌳", layout="wide")

# --- 세션 상태 초기화 ---
# 스크립트가 재실행되어도 'analysis_result' 값을 유지하기 위해 세션 상태를 사용합니다.
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- UI 레이아웃 ---
st.title("🌳 태국 법률 문서 계층 분석기 (Hybrid Ver.)")
st.markdown("Top-down 분석과 Python 후처리를 결합하여 태국 법률 문서의 계층 구조를 분석합니다.")

uploaded_file = st.file_uploader("분석할 태국 법률 텍스트 파일(.txt)을 업로드하세요.", type=['txt'])

if uploaded_file is not None:
    # '계층 구조 분석 실행' 버튼
    if st.button("계층 구조 분석 실행", type="primary"):
        # 분석 시작 전, 이전 결과를 초기화합니다.
        st.session_state.analysis_result = None
        document_text = uploaded_file.getvalue().decode('utf-8')
        
        # 'with st.status'를 사용하여 처리 과정을 시각적으로 보여줍니다.
        with st.status("하이브리드 분석 파이프라인을 시작합니다...", expanded=True) as status:
            try:
                # secrets.toml 또는 Streamlit 클라우드의 환경 변수에서 API 키를 가져옵니다.
                api_key = st.secrets["GEMINI_API_KEY"]
                
                # 핵심 분석 함수를 호출합니다.
                final_result, debug_info = document_processor.run_hybrid_pipeline(
                    document_text=document_text,
                    api_key=api_key,
                    status_container=status
                )
                
                # [핵심] 분석 결과를 세션 상태에 저장합니다.
                st.session_state.analysis_result = {
                    "final": final_result,
                    "debug": debug_info,
                    "file_name": uploaded_file.name
                }
                
                status.update(label="✅ 계층 분석 완료!", state="complete", expanded=False)
                st.success("🎉 성공적으로 계층 트리 구조를 생성했습니다! 아래에서 결과를 확인하세요.")

            except Exception as e:
                # 오류 발생 시 세션 상태를 다시 초기화합니다.
                st.session_state.analysis_result = None
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")
                st.code(traceback.format_exc())

# --- 결과 표시 ---
# [핵심] 세션 상태에 저장된 'analysis_result'가 있을 경우에만 이 블록을 실행합니다.
if st.session_state.analysis_result:
    # 세션 상태에서 결과 데이터를 가져옵니다.
    result = st.session_state.analysis_result
    final_result_data = result["final"]
    debug_info = result["debug"]
    file_name = result["file_name"]

    st.header("📄 분석 결과 확인 및 다운로드")

    # 결과 데이터에 오류가 있는지 확인합니다.
    if "error" in final_result_data or not final_result_data.get("tree"):
        st.error("분석 실패: 문서에서 유효한 구조를 추출하지 못했습니다. 아래 디버그 로그를 확인해주세요.")
    
    # 청킹 정보를 표시합니다.
    chunk_info_list = [item for item in debug_info if "chunking_details" in item]
    if chunk_info_list:
        with st.expander("문서 분할(Chunking) 정보 보기"):
            details = chunk_info_list[0]['chunking_details']
            st.write(f"입력된 문서가 총 **{len(details)}**개의 조각으로 분할되었습니다.")
            st.dataframe(pd.DataFrame(details).set_index('chunk'))

    # 탭으로 최종 결과와 디버그 로그를 분리하여 표시합니다.
    tab1, tab2 = st.tabs(["✔️ 최종 결과 (계층 트리)", "🐞 상세 디버그 로그"])

    with tab1:
        st.json(final_result_data, expanded=True)
        st.download_button(
           label="결과 트리 (JSON) 다운로드",
           data=json.dumps(final_result_data, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_hybrid_tree.json",
           mime="application/json",
        )

    with tab2:
        st.write("파이프라인 각 단계에서 LLM이 반환한 원본 데이터 등 상세한 로그를 포함합니다.")
        st.json({"pipeline_logs": debug_info}, expanded=False)
        st.download_button(
           label="디버그 로그 (JSON) 다운로드",
           data=json.dumps(debug_info, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_hybrid_debug.json",
           mime="application/json",
        )