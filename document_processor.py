# document_processor.py

import google.generativeai as genai

def run_pipeline(document_text, api_key):
    """
    문서 전체 텍스트를 입력받아 최종 JSON을 반환하는 메인 함수
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # 🎯 1. Global Summary 생성 (실제 LLM 호출)
    # 문서의 첫 3000자를 서문으로 간주하고 요약을 요청합니다.
    preamble = document_text[:3000]
    prompt_global_summary = f"""
        Analyze the preamble of the Thai legal document provided below. 
        Summarize the document's purpose, background, and core principles in 2-3 sentences in Korean.

        [Preamble text]
        {preamble}
    """
    response_summary = model.generate_content(prompt_global_summary)
    generated_global_summary = response_summary.text.strip()
    
    # 🎯 2. Generated Title 생성 (이전과 동일)
    prompt_title = f"다음 텍스트의 제목을 한 줄로 만들어줘: {document_text[:200]}"
    response_title = model.generate_content(prompt_title)
    generated_title = response_title.text.strip()
    
    # 최종 JSON 조립
    final_json = {
      "generated_title": generated_title,
      "global_summary": generated_global_summary, # <-- 실제 요약으로 교체!
      "document_title": "분석된 문서",
      "chapters": [ # <-- 이 부분은 아직 가짜 데이터입니다.
        {
          "type": "chapter",
          "title": "หมวด 1 บททั่วไป",
          "summary": "이 챕터는 태국의 주권과 통치 형태를 설명합니다."
        }
      ]
    }
    
    return final_json