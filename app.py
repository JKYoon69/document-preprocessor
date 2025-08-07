# app.py (구조 분석 결과 표시 및 다운로드)

import streamlit as st
import document_processor
import json
import pandas as pd

st.set_page_config(page_title="구조 분석 및 통합", page_icon="🧩")
st.title("🧩 2단계: 구조 분석 및 통합")
st.write("의미 기반으로 분할된 각 청크의 구조를 분석하고, 결과를 통합하여 통계를 표시합니다.")

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    st.write(f"파일 정보: 총 **{len(document_text):,}** 글자")

    if st.button("구조 분석 실행"):
        with st.status("분석을 시작합니다...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                result = document_processor.run_structure_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status
                )
                
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)
                
                # --- 결과 표시 ---
                st.subheader("📊 분석 통계 요약")

                # 1. 청크별 분석 결과
                with st.expander("1. 청크별 상세 분석 결과 보기"):
                    if result.get("chunk_stats"):
                        df_chunk = pd.DataFrame(result["chunk_stats"])
                        st.table(df_chunk.set_index("청크 번호"))
                    else:
                        st.write("청크 분석 결과가 없습니다.")
                
                # 2. 중복 및 최종 결과 요약
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**중복 발견 항목**")
                    dup_counts = result.get("duplicate_counts", {})
                    st.metric(label="Chapters (หมวด)", value=dup_counts.get("chapter", 0))
                    st.metric(label="Sections (ส่วน)", value=dup_counts.get("section", 0))
                    st.metric(label="Articles (มาตรา)", value=dup_counts.get("article", 0))
                
                with col2:
                    st.write("**최종 항목 (고유)**")
                    final_counts = result.get("final_counts", {})
                    st.metric(label="Chapters (หมวด)", value=final_counts.get("chapter", 0))
                    st.metric(label="Sections (ส่วน)", value=final_counts.get("section", 0))
                    st.metric(label="Articles (มาตรา)", value=final_counts.get("article", 0))

                # 3. 최종 결과 다운로드 버튼
                st.download_button(
                   label="📋 최종 결과(JSON) 다운로드",
                   data=json.dumps(result.get("final_result", {}), indent=2, ensure_ascii=False),
                   file_name=f"{uploaded_file.name.split('.')[0]}_structure.json",
                   mime="application/json",
                )

            except Exception as e:
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")