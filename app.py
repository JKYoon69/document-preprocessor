# app.py (최종 완성 버전)

import streamlit as st
import document_processor
import json
import pandas as pd

st.set_page_config(page_title="태국 법률 문서 분석기", page_icon="⚖️", layout="wide")
st.title("⚖️ 태국 법률 문서 분석기 v3.0 Final")
st.write("문서를 의미 기반으로 분할하고, 구조를 분석하여 통계 및 JSON 결과물을 생성합니다.")

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    st.write(f"파일 정보: 총 **{len(document_text):,}** 글자")

    if st.button("전체 파이프라인 실행"):
        with st.status("분석을 시작합니다...", expanded=True) as status:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                # 파이프라인 실행 후 두 종류의 결과를 받음
                final_result_data, stats_data = document_processor.run_full_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status
                )
                
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)
                
                # --- 결과 표시 ---
                st.subheader("📊 분석 통계 요약")

                # 1. 청크별 분석 결과
                with st.expander("1. 청크별 상세 분석 결과 보기"):
                    if stats_data.get("chunk_stats"):
                        df_chunk = pd.DataFrame(stats_data["chunk_stats"])
                        st.table(df_chunk.set_index("청크 번호"))
                    else:
                        st.write("청크 분석 결과가 없습니다.")
                
                # 2. 중복 및 최종 결과 요약
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**중복 발견 항목**")
                    dup_counts = stats_data.get("duplicate_counts", {})
                    st.metric(label="Chapters (หมวด)", value=dup_counts.get("chapter", 0))
                    st.metric(label="Sections (ส่วน)", value=dup_counts.get("section", 0))
                    st.metric(label="Articles (มาตรา)", value=dup_counts.get("article", 0))
                
                with col2:
                    st.write("**최종 항목 (고유)**")
                    final_counts = stats_data.get("final_counts", {})
                    st.metric(label="Chapters (หมวด)", value=final_counts.get("chapter", 0))
                    st.metric(label="Sections (ส่วน)", value=final_counts.get("section", 0))
                    st.metric(label="Articles (มาตรา)", value=final_counts.get("article", 0))

                # 3. 최종 결과 다운로드 버튼
                st.download_button(
                   label="📋 최종 구조 분석 결과(JSON) 다운로드",
                   data=json.dumps(final_result_data, indent=2, ensure_ascii=False),
                   file_name=f"{uploaded_file.name.split('.')[0]}_structure.json",
                   mime="application/json",
                )
                st.success("🎉 모든 작업이 성공적으로 완료되었습니다!")

            except Exception as e:
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")