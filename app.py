# app.py (최종 청크 분할 테스트용)

import streamlit as st
import document_processor
import pandas as pd

st.set_page_config(page_title="청크 분할 최종 테스트", page_icon="🔬")
st.title("🔬 청크 분할 최종 검증")
st.write("파일을 업로드하면, 새로 작성된 `chunk_text` 함수의 결과를 상세히 확인할 수 있습니다.")

uploaded_file = st.file_uploader("385KB 법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("청크 분할 실행"):
        document_text = uploaded_file.getvalue().decode("utf-8")
        
        st.write("---")
        st.subheader("`chunk_text` 함수 실행 결과")

        chunk_indices = document_processor.chunk_text(document_text)
        
        if chunk_indices:
            st.success(f"✅ 분할 성공: 총 **{len(chunk_indices)}** 개의 청크가 생성되었습니다.")
            
            # 표로 보여줄 데이터 준비
            display_data = []
            for i, indices in enumerate(chunk_indices):
                display_data.append({
                    "청크 번호": i + 1,
                    "시작 인덱스": indices["start"],
                    "끝 인덱스": indices["end"],
                    "청크 크기 (Bytes)": indices["end"] - indices["start"]
                })
            
            st.table(pd.DataFrame(display_data))

            # 요청하신 대로 처음 3개 청크의 인덱스 값을 별도로 출력
            st.subheader("처음 3개 청크 상세 인덱스")
            for i in range(min(3, len(chunk_indices))):
                st.write(f"**청크 {i+1}**: 시작 인덱스 = `{chunk_indices[i]['start']}`, 끝 인덱스 = `{chunk_indices[i]['end']}`")

        else:
            st.error("❌ 분할 실패: 청크가 생성되지 않았습니다.")