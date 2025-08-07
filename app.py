# app.py (의미 기반 청킹 테스트용)

import streamlit as st
import document_processor
import pandas as pd

st.set_page_config(page_title="의미 기반 청킹 테스트", page_icon="🧠")
st.title("🧠 의미 기반 청킹(Semantic Chunking) 검증")
st.write("두 가지 청킹 방식의 결과를 비교합니다.")

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    st.write(f"파일 정보: 총 **{len(document_text):,}** 글자")

    if st.button("두 방식 모두 실행하여 비교하기"):
        
        # --- 1. 기존 방식: 글자 수 기준 ---
        st.subheader("1. 글자 수 기준 (단순 분할)")
        char_chunks = document_processor.chunk_text_by_char(document_text)
        
        char_data = []
        for i, indices in enumerate(char_chunks):
            start, end = indices["start_char"], indices["end_char"]
            chunk_content = document_text[start:end]
            char_data.append({
                "청크 번호": i + 1,
                "시작 위치": start,
                "끝 위치": end,
                "글자 수": len(chunk_content)
            })
        st.table(pd.DataFrame(char_data))

        # --- 2. 새로운 방식: 의미 기반 ---
        st.subheader("2. 의미 기반 (Semantic Chunking)")
        semantic_chunks = document_processor.chunk_text_semantic(document_text)

        semantic_data = []
        for i, indices in enumerate(semantic_chunks):
            start, end = indices["start_char"], indices["end_char"]
            chunk_content = document_text[start:end]
            semantic_data.append({
                "청크 번호": i + 1,
                "시작 위치": start,
                "끝 위치": end,
                "글자 수": len(chunk_content),
                "마지막 10글자": "..." + chunk_content[-10:].replace("\n", "\\n")
            })
        st.table(pd.DataFrame(semantic_data))