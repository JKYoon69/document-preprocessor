# app.py (진행 과정 상세 표시 버전)

import streamlit as st
import document_processor
import json
import pandas as pd

st.set_page_config(page_title="상세 진행 과정 표시", page_icon="💬")
st.title("💬 상세 진행 과정 표시기")
st.write("의미 기반으로 문서를 분할하고, 첫 번째 청크를 이용해 문서 전체의 요약을 생성합니다.")

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode('utf-8')
    st.write(f"파일 정보: 총 **{len(document_text):,}** 글자")

    if st.button("요약 생성 실행"):
        # st.status를 사용하여 전체 작업의 진행 상태를 관리
        with st.status("분석을 시작합니다...", expanded=True) as status:
            try:
                # --- 1. 의미 기반 청킹 실행 ---
                status.update(label="1/2: 의미 기반으로 청크 분할 중...")
                chunks = document_processor.chunk_text_semantic(document_text)
                
                if not chunks:
                    st.error("오류: 문서를 청크로 분할할 수 없습니다.")
                    status.update(label="오류 발생", state="error")
                else:
                    # 청크 분할 결과를 표로 명확하게 표시
                    chunk_data = [{
                        "청크 번호": i + 1,
                        "시작 위치": c["start_char"],
                        "끝 위치": c["end_char"],
                        "글자 수": len(c["text"])
                    } for i, c in enumerate(chunks)]
                    
                    st.write("✅ 청킹 완료!")
                    st.table(pd.DataFrame(chunk_data).set_index("청크 번호"))
                    
                    # --- 2. 첫 번째 청크로 전역 요약 생성 ---
                    status.update(label="2/2: 전역 요약 생성 중...")
                    st.write("첫 번째 청크를 사용하여 Gemini 모델을 호출합니다...")
                    
                    first_chunk_text = chunks[0]["text"]
                    api_key = st.secrets["GEMINI_API_KEY"]
                    
                    summary_result_text = document_processor.get_global_summary(first_chunk_text, api_key)
                    
                    st.write("✅ LLM으로부터 응답을 받았습니다.")
                    st.info("LLM 원본 응답:")
                    st.text(summary_result_text)

                    # LLM 응답이 JSON 형식이면 파싱하여 예쁘게 보여줌
                    st.write("응답에서 JSON 데이터를 파싱합니다...")
                    try:
                        json_part = summary_result_text[summary_result_text.find('{'):summary_result_text.rfind('}')+1]
                        parsed_json = json.loads(json_part)
                        
                        st.write("✅ JSON 파싱 성공!")
                        st.json(parsed_json)
                        
                        # --- 3. 최종 완료 ---
                        status.update(label="분석 완료!", state="complete", expanded=False)
                        st.success("🎉 모든 작업이 성공적으로 완료되었습니다!")

                    except (json.JSONDecodeError, IndexError):
                        st.error("오류: LLM의 응답에서 유효한 JSON을 찾을 수 없습니다.")
                        status.update(label="오류 발생", state="error")

            except Exception as e:
                status.update(label="치명적 오류 발생", state="error", expanded=True)
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {e}")