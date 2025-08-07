# app.py (전역 요약 생성 테스트용)

import streamlit as st
import document_processor
import json
import pandas as pd

st.set_page_config(page_title="전역 요약 생성 테스트", page_icon="📜")
st.title("📜 전역 요약(Global Summary) 생성 검증")
st.write("의미 기반으로 문서를 분할하고, 첫 번째 청크를 이용해 문서 전체의 요약을 생성합니다.")

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    st.write(f"파일 정보: 총 **{len(document_text):,}** 글자")

    if st.button("요약 생성 실행"):
        # --- 1. 의미 기반 청킹 실행 ---
        st.subheader("1. 의미 기반 청킹 결과")
        chunks = document_processor.chunk_text_semantic(document_text)
        
        if not chunks:
            st.error("오류: 문서를 청크로 분할할 수 없습니다.")
        else:
            st.success(f"✅ 분할 성공: 총 **{len(chunks)}** 개의 청크가 생성되었습니다.")
            
            # --- 2. 첫 번째 청크로 전역 요약 생성 ---
            st.subheader("2. 전역 요약 생성 결과")
            with st.spinner("Gemini 모델을 호출하여 요약을 생성 중입니다..."):
                first_chunk_text = chunks[0]["text"]
                api_key = st.secrets["GEMINI_API_KEY"]
                
                summary_result = document_processor.get_global_summary(first_chunk_text, api_key)
                
                st.info("LLM 원본 응답:")
                st.text(summary_result)

                # LLM 응답이 JSON 형식이면 파싱하여 예쁘게 보여줌
                try:
                    # 응답 텍스트에서 JSON 부분만 추출
                    json_part = summary_result[summary_result.find('{'):summary_result.rfind('}')+1]
                    parsed_json = json.loads(json_part)
                    st.success("✅ 요약 생성 및 JSON 파싱 성공!")
                    st.json(parsed_json)
                except (json.JSONDecodeError, IndexError):
                    st.error("오류: LLM의 응답에서 유효한 JSON을 찾을 수 없습니다.")