# app.py

import streamlit as st
import document_processor

# --- 화면 UI 구성 ---
st.set_page_config(page_title="태국 법률 문서 분석기", page_icon="🇹🇭")
st.title("🇹🇭 태국 법률 문서 분석기 v0.3")
st.write("텍스트 파일을 업로드하면 Gemini 2.5 Flash-lite 모델이 계층 구조로 분석하고 요약합니다.")

uploaded_file = st.file_uploader("분석할 .txt 파일을 선택하세요.", type=['txt'])

# --- 로직 실행 부분 ---
if uploaded_file is not None:
    if st.button("분석 시작하기"):
        
        # 👇 st.spinner 대신 st.status를 사용합니다.
        # "with" 구문이 끝나면 상태가 "Completed!"로 자동 변경됩니다.
        with st.status("분석을 준비하고 있습니다...", expanded=True) as status:
            try:
                # 1. 파일 읽기
                status.write("파일을 읽고 텍스트를 준비합니다...")
                document_text = uploaded_file.getvalue().decode("utf-8")
                
                # 2. API 키 가져오기
                status.write("API 키를 확인합니다...")
                api_key = st.secrets["GEMINI_API_KEY"]
                
                # 3. ⭐️ 실제 파이프라인 함수 호출 (status 객체를 전달!)
                # 이제 파이프라인 내부에서 진행 상황을 업데이트할 수 있습니다.
                final_json = document_processor.run_pipeline(
                    document_text=document_text, 
                    api_key=api_key,
                    status_container=status  # status 객체를 넘겨줍니다.
                )
                
                # 4. 모든 작업이 끝나면 성공 메시지 표시
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)

                # 최종 결과를 화면에 예쁘게 출력
                st.subheader("✅ 최종 분석 결과")
                st.json(final_json)

            except Exception as e:
                # 오류 발생 시 상태를 업데이트하고 에러 메시지 표시
                status.update(label="오류 발생", state="error")
                st.error(f"오류가 발생했습니다: {e}")
                st.error("API 키가 정확한지, Streamlit Cloud의 Secrets에 잘 설정되었는지 확인해보세요.")