# app.py
import streamlit as st
import document_processor as dp
import json
import traceback
import pandas as pd
import time

# --- Helper Functions ---
def count_short_nodes(node, threshold, count=0):
    """재귀적으로 트리를 순회하며 텍스트가 짧은 노드의 개수를 셉니다."""
    if len(node.get('text', '')) < threshold:
        count += 1
    for child in node.get('children', []):
        count = count_short_nodes(child, threshold, count)
    return count

# --- 페이지 설정 ---
st.set_page_config(page_title="법률 문서 계층 분석기", page_icon="🏛️", layout="wide")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- UI 레이아웃 ---
st.title("🏛️ 태국 법률 문서 계층 분석기 (v3.2)")
st.markdown(f"**LLM Model:** `{dp.MODEL_NAME}` (수정은 `document_processor.py`에서 가능)")
st.markdown("3단계 파이프라인과 자동 재시도, 성능 측정을 통해 법률 문서 구조를 분석합니다.")

with st.expander("⚙️ 각 단계별 프롬프트 수정하기"):
    tab1, tab2, tab3 = st.tabs(["1단계: Architect", "2단계: Surveyor", "3단계: Detailer"])

    with tab1:
        st.info("문서 전체에서 최상위 구조(Book, Part, Chapter)를 찾는 임무를 정의합니다.")
        st.session_state.prompt1 = st.text_area(
            "Architect Prompt", value=dp.PROMPT_ARCHITECT, height=250
        )
    with tab2:
        st.info("각 Chapter 내부에서 중간 구조(Section)를 찾는 임무를 정의합니다.")
        st.session_state.prompt2 = st.text_area(
            "Surveyor Prompt", value=dp.PROMPT_SURVEYOR, height=250
        )
    with tab3:
        st.info("가장 작은 단위(Section 또는 Chapter) 내부에서 최하위 구조(Article)를 찾는 임무를 정의합니다.")
        st.session_state.prompt3 = st.text_area(
            "Detailer Prompt", value=dp.PROMPT_DETAILER, height=250
        )

uploaded_file = st.file_uploader("분석할 태국 법률 텍스트 파일(.txt)을 업로드하세요.", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    char_count = len(document_text)
    st.info(f"📁 업로드된 파일: **{uploaded_file.name}** | 총 글자 수: **{char_count:,}** 자")

    if st.button("계층 구조 분석 실행", type="primary"):
        st.session_state.analysis_result = None
        
        @st.cache_data
        def get_debug_info_collector():
            return []
        
        debug_info_collector = get_debug_info_collector()
        debug_info_collector.clear()

        def display_intermediate_result(result, container, debug_info):
            llm_duration = next((item.get('llm_duration_seconds', 0) for item in debug_info if "step1_architect_response" in item), 0)
            container.write(f"✅ 1단계 완료! (LLM 응답 시간: {llm_duration:.2f}초)")
            container.write("찾아낸 최상위 구조:")
            display_data = [{"type": n.get('type'), "title": n.get('title')} for n in result]
            container.dataframe(display_data)

        with st.status("3단계 분석 파이프라인을 시작합니다...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                final_result, debug_info = dp.run_pipeline(
                    document_text=document_text,
                    api_key=api_key,
                    status_container=status,
                    prompt_architect=st.session_state.prompt1,
                    prompt_surveyor=st.session_state.prompt2,
                    prompt_detailer=st.session_state.prompt3,
                    intermediate_callback=display_intermediate_result
                )
                
                st.session_state.analysis_result = {
                    "final": final_result,
                    "debug": debug_info,
                    "file_name": uploaded_file.name
                }
                status.update(label="✅ 계층 분석 완료!", state="complete", expanded=False)
                st.success("🎉 성공적으로 계층 트리 구조를 생성했습니다!")
                time.sleep(0.5)
                st.rerun()

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

    st.header("📄 분석 결과 확인 및 다운로드")

    if "error" in final_result_data or not final_result_data.get("tree"):
        st.error("분석 실패: 문서에서 유효한 구조를 추출하지 못했습니다. 아래 디버그 로그를 확인해주세요.")
    else:
        short_node_threshold = 15
        total_short_nodes = sum(count_short_nodes(node, short_node_threshold) for node in final_result_data.get('tree', []))
        
        timings_list = [item for item in debug_info if "performance_timings" in item]
        if timings_list:
            timings = timings_list[0]['performance_timings']
            st.subheader("📊 분석 요약")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("1단계 (Architect)", f"{timings.get('step1_architect_duration', 0):.2f} 초")
            col2.metric("2단계 (Surveyor)", f"{timings.get('step2_surveyor_duration', 0):.2f} 초")
            col3.metric("3단계 (Detailer)", f"{timings.get('step3_detailer_duration', 0):.2f} 초")
            col4.metric(f"짧은 노드 (<{short_node_threshold}자)", f"{total_short_nodes} 개",
                        help=f"텍스트 내용이 {short_node_threshold}자 미만인 노드의 수입니다.",
                        delta=f"-{total_short_nodes}" if total_short_nodes > 0 else None,
                        delta_color="inverse" if total_short_nodes > 0 else "off")

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