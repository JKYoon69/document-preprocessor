# app.py

import streamlit as st
import document_processor
import json
import pandas as pd
import traceback

st.set_page_config(page_title="법률 문서 분석기", page_icon="⚖️", layout="wide")
st.title("⚖️ 태국 법률 문서 RAG 전처리기 (상세 분석 모드)")
st.write("문서의 모든 처리 과정을 상세히 추적하고, 최종 결과와 디버그 로그를 다운로드합니다.")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("전체 파이프라인 실행"):
        document_text = uploaded_file.getvalue().decode('utf-8')
        st.session_state.analysis_result = None
        
        with st.status("분석을 시작합니다...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                # ✅✅✅ 함수 이름을 'run_final_pipeline'으로 수정 ✅✅✅
                final_result, stats, debug_info = document_processor.run_final_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status
                )
                st.session_state.analysis_result = {
                    "final": final_result,
                    "stats": stats,
                    "debug": debug_info,
                    "file_name": uploaded_file.name
                }
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)
                st.success("🎉 모든 작업이 성공적으로 완료되었습니다!")
            except Exception as e:
                st.session_state.analysis_result = None
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")
                st.code(traceback.format_exc())

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    stats_data = result["stats"]
    final_result_data = result["final"]
    debug_info = result["debug"]
    file_name = result["file_name"]
    
    st.subheader("📊 분석 통계 요약")
    
    st.markdown("---")
    st.write("**1. 청킹(Chunking) 정보**")
    if stats_data and stats_data.get("chunk_stats"):
        st.write(f"총 **{len(stats_data['chunk_stats'])}** 개의 청크로 분할되었습니다.")
    else:
        num_chunks = len([d for d in debug_info if "chunk_" in next(iter(d))])
        st.write(f"총 **{num_chunks}** 개의 청크로 분할되었습니다.")


    st.markdown("---")
    st.write("**2. 구조 파악 정보**")
    with st.expander("청크별 상세 분석 결과 보기"):
        if stats_data and stats_data.get("chunk_stats"):
            st.table(pd.DataFrame(stats_data["chunk_stats"]).set_index("Chunk Number"))
        else:
            st.write("청크 분석 결과가 없습니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**중복 발견 항목**"); dup_counts = stats_data.get("duplicate_counts", {})
        st.metric("Books", dup_counts.get("book", 0)); st.metric("Parts", dup_counts.get("part", 0)); st.metric("Chapters", dup_counts.get("chapter", 0)); st.metric("Sections", dup_counts.get("section", 0)); st.metric("Articles", dup_counts.get("article", 0))
    with col2:
        st.write("**최종 항목 (고유)**"); final_counts = stats_data.get("final_counts", {})
        st.metric("Books", final_counts.get("book", 0)); st.metric("Parts", final_counts.get("part", 0)); st.metric("Chapters", final_counts.get("chapter", 0)); st.metric("Sections", final_counts.get("section", 0)); st.metric("Articles", final_counts.get("article", 0))

    st.markdown("---")
    st.subheader("📋 결과 확인 및 다운로드")
    
    with st.expander("최종 결과 미리보기 (JSON)", expanded=False):
        st.json(final_result_data)
    
    with st.expander("디버그 로그 미리보기 (JSON)", expanded=True):
        st.json({"llm_responses": debug_info})

    col1_dl, col2_dl = st.columns(2)
    with col1_dl:
        st.download_button(
           label="✔️ 최종 결과(JSON) 다운로드",
           data=json.dumps(final_result_data, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_structure.json",
           mime="application/json",
        )
    with col2_dl:
        st.download_button(
           label="🐞 디버그 로그(JSON) 다운로드",
           data=json.dumps(debug_info, indent=2, ensure_ascii=False),
           file_name=f"{file_name.split('.')[0]}_debug.json",
           mime="application/json",
        )