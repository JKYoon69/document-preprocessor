# app.py
import streamlit as st
import document_processor
import json
import traceback

# --- 페이지 설정 ---
st.set_page_config(page_title="법률 문서 계층 분석기", page_icon="🌳", layout="wide")

# --- 세션 상태 초기화 ---
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- UI 레이아웃 ---
st.title("🌳 태국 법률 문서 계층 분석기 (Hybrid Ver.)")
st.markdown("""
이 도구는 **Top-down 분석**과 **Python 후처리**를 결합한 Hybrid 방식으로 태국 법률 문서를 분석합니다.
1.  **(1단계) Chapter 분석:** LLM이 문서 전체에서 최상위 구조(`Chapter`, `Part` 등)를 식별합니다.
2.  **(2단계) 후처리:** Python 코드가 각 Chapter의 경계를 보정하여 내용 누락을 방지합니다.
3.  **(3단계) 하위 구조 분석:** 보정된 각 Chapter 내부에서 `Section`, `Article`을 다시 LLM으로 식별하고 후처리합니다.
4.  **(결과) 계층형 JSON 생성:** 최종적으로 부모-자식 관계가 명확한 트리 구조의 JSON을 생성합니다.
""")

uploaded_file = st.file_uploader("분석할 태국 법률 텍스트 파일(.txt)을 업로드하세요.", type=['txt'])

if uploaded_file is not None:
    # 분석 실행 버튼
    if st.button("계층 구조 분석 실행", type="primary"):
        document_text = uploaded_file.getvalue().decode('utf-8')
        st.session_state.analysis_result = None # 이전 결과 초기화
        
        with st.status("하이브리드 분석 파이프라인을 시작합니다...", expanded=True) as status:
            try:
                # secrets.toml 또는 환경 변수에서 API 키 가져오기
                api_key = st.secrets["GEMINI_API_KEY"]
                
                final_result, debug_info = document_processor.run_hybrid_pipeline(
                    document_text=document_text,
                    api_key=api_key,
                    status_container=status
                )
                
                # 결과를 세션 상태에 저장
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
    
    # 탭으로 결과와 디버그 로그 분리
    tab1, tab2 = st.tabs(["✔️ 최종 결과 (계층 트리)", "🐞 디버그 로그"])

    with tab1:
        st.json(final_result_data, expanded=True)
        st.download_button(
           label="결과 트리 (JSON) 다운로드",
           data=json.dumps(final_result_data, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_hybrid_tree.json",
           mime="application/json",
        )

    with tab2:
        st.json({"llm_responses_and_errors": debug_info}, expanded=False)
        st.download_button(
           label="디버그 로그 (JSON) 다운로드",
           data=json.dumps(debug_info, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_hybrid_debug.json",
           mime="application/json",
        )