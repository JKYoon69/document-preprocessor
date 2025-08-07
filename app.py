# app.py (글자 수 기준 최종 테스트용)

import streamlit as st
import document_processor
import pandas as pd

st.set_page_config(page_title="글자 수 기준 청크 테스트", page_icon="📄")
st.title("📄 글자 수 기준 청크 분할 최종 검증")
st.write("파일을 업로드하면, '글자 수' 기준으로 청크를 나눈 결과를 확인합니다.")

uploaded_file = st.file_uploader("법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    document_bytes = uploaded_file.getvalue()
    document_text = document_bytes.decode('utf-8')
    
    st.write(f"파일 정보: 총 **{len(document_bytes):,}** 바이트, 총 **{len(document_text):,}** 글자")

    if st.button("청크 분할 실행"):
        st.write("---")
        st.subheader("`chunk_text_by_char` 함수 실행 결과")

        # 글자 수 100,000 / 중첩 20,000 으로 실행
        chunk_indices = document_processor.chunk_text_by_char(
            document_text, 
            chunk_size_chars=100000, 
            overlap_chars=20000
        )
        
        if chunk_indices:
            st.success(f"✅ 분할 성공: 총 **{len(chunk_indices)}** 개의 청크가 생성되었습니다.")
            
            display_data = []
            for i, indices in enumerate(chunk_indices):
                start, end = indices["start_char"], indices["end_char"]
                # 실제 텍스트 슬라이싱
                chunk_content = document_text[start:end]
                
                display_data.append({
                    "청크 번호": i + 1,
                    "시작 글자 위치": start,
                    "끝 글자 위치": end,
                    "청크 글자 수": len(chunk_content)
                })
            
            st.table(pd.DataFrame(display_data))

        else:
            st.error("❌ 분할 실패: 청크가 생성되지 않았습니다.")