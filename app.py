# app.py (청크 분할 테스트용)

import streamlit as st
import document_processor
import pandas as pd # 결과를 표로 보여주기 위해 추가

st.set_page_config(page_title="청크 분할 테스트", page_icon="🧪")
st.title("🧪 청크 분할 기능 검증")
st.write("파일을 업로드하면 `chunk_text` 함수의 결과를 직접 확인할 수 있습니다.")
st.info("이 테스트 앱은 LLM을 호출하지 않으므로 API 키가 필요 없습니다.")

uploaded_file = st.file_uploader("385KB 법률 파일을 선택하세요.", type=['txt'])

if uploaded_file is not None:
    if st.button("청크 분할 실행"):
        document_text = uploaded_file.getvalue().decode("utf-8")
        
        st.write("---")
        st.subheader("`chunk_text` 함수 실행 결과")

        chunk_list = document_processor.chunk_text(document_text)
        
        if chunk_list:
            st.success(f"✅ 분할 성공: 총 **{len(chunk_list)}** 개의 청크가 생성되었습니다.")

            # 결과를 표로 변환
            display_data = []
            for i, chunk in enumerate(chunk_list):
                display_data.append({
                    "청크 번호": i + 1,
                    "전역 시작 위치": chunk["global_start"],
                    "청크 크기 (Bytes)": chunk["size"]
                })
            
            # 표로 출력
            st.table(pd.DataFrame(display_data))

        else:
            st.error("❌ 분할 실패: 청크가 생성되지 않았습니다.")