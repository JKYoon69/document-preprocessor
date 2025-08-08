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
st.title("🏛️ 태국 법률 문서 계층 분석기 (v3.1)")
st.markdown(f"**LLM Model:** `{dp.MODEL_NAME}` (수정은 `document_processor.py`에서 가능)")
st.markdown("3단계 파이프라인과 자동 재시도, 성능 측정을 통해 법률 문서 구조를 분석합니다.")

with st.expander("⚙️ 각 단계별 프롬프트 수정하기"):
    # ... 이전과 동일 ...

uploaded_file = st.file_uploader("분석할 태국 법률 텍스트 파일(.txt)을 업로드하세요.", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    char_count = len(document_text)
    st.info(f"📁 업로드된 파일: **{uploaded_file.name}** | 총 글자 수: **{char_count:,}** 자")

    if st.button("계층 구조 분석 실행", type="primary"):
        st.session_state.analysis_result = None
        
        def display_intermediate_result(result, container):
            """1단계 완료 후 중간 결과를 표시하기 위한 콜백 함수"""
            llm_duration = next((item.get('llm_duration', 0) for item in debug_info if "step1_architect_response" in item), 0)
            container.write(f"✅ 1단계 완료! (LLM 응답 시간: {llm_duration:.2f}초)")
            container.write("찾아낸 최상위 구조:")
            container.json([{"type": n.get('type'), "title": n.get('title')} for n in result])

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
                # ... 이전과 동일 ...

# --- 결과 표시 ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    final_result_data = result["final"]
    debug_info = result["debug"]
    file_name = result["file_name"]

    st.header("📄 분석 결과 확인 및 다운로드")

    if "error" in final_result_data or not final_result_data.get("tree"):
        # ... 이전과 동일 ...
    else:
        # --- [신규] 품질 및 성능 요약 ---
        short_node_threshold = 15
        total_short_nodes = sum(count_short_nodes(node, short_node_threshold) for node in final_result_data['tree'])
        
        timings_list = [item for item in debug_info if "performance_timings" in item]
        if timings_list:
            timings = timings_list[0]['performance_timings']
            
            st.subheader("📊 분석 요약")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("1단계 (Architect)", f"{timings.get('step1_architect_duration', 0):.2f} 초")
            col2.metric("2단계 (Surveyor)", f"{timings.get('step2_surveyor_duration', 0):.2f} 초")
            col3.metric("3단계 (Detailer)", f"{timings.get('step3_detailer_duration', 0):.2f} 초")
            col4.metric(f"짧은 노드 (<{short_node_threshold}자)", f"{total_short_nodes} 개", 
                        help="텍스트 내용이 15자 미만인 노드의 수입니다. 이 수치가 높으면 일부 구조의 경계가 잘못되었을 수 있습니다.",
                        delta=f"-{total_short_nodes}", delta_color="inverse")

    # 탭으로 결과와 디버그 로그 분리
    tab1, tab2 = st.tabs(["✔️ 최종 결과 (계층 트리)", "🐞 상세 디버그 로그"])
    # ... 탭 내용 이전과 동일 ...