# document_processor.py

import google.generativeai as genai
import json # LLM이 반환한 텍스트를 JSON으로 파싱하기 위해 import합니다.

# 헬퍼 함수: LLM의 응답에서 JSON만 깔끔하게 추출
def extract_json_from_response(text):
    try:
        # LLM 응답에 설명이 붙는 경우, ```json ... ``` 부분만 찾아서 파싱
        start = text.find('```json') + len('```json')
        end = text.rfind('```')
        if start != -1 and end != -1:
            json_text = text[start:end].strip()
            return json.loads(json_text)
        else: # ```json``` 형식이 없는 경우
            return json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"원본 텍스트: {text}")
        return None # 오류 발생 시 None 반환

# --- 메인 파이프라인 함수 ---
def run_pipeline(document_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # 1. Global Summary 생성 (이전과 동일, 정상 작동)
    preamble = document_text[:3000]
    prompt_global_summary = f"""
        Analyze the preamble of the Thai legal document provided below.
        Summarize the document's purpose, background, and core principles in 2-3 sentences in Korean.

        [Preamble text]
        {preamble}
    """
    response_summary = model.generate_content(prompt_global_summary)
    generated_global_summary = response_summary.text.strip()

    # 2. 🎯 문서 전체 구조 분석 (가장 중요한 새 기능!)
    # 이 프롬프트는 문서 전체를 주고, 모든 구조적 헤더를 찾아달라고 요청합니다.
    prompt_structure = f"""
        The following text is a Thai legal document. Identify all hierarchical headers such as 'หมวด', 'ส่วน', and 'มาตรา'.
        For each identified element, extract its type, title, and the full text content under it until the next header begins.
        Return the result as a JSON array.

        Example format for a single element:
        {{
          "type": "chapter" or "section" or "article",
          "title": "หมวด 1 บททั่วไป" or "มาตรา 1",
          "text": "The full text of this element..."
        }}
        
        [Full Document Text]
        {document_text}
    """
    
    # 구조 분석 LLM 호출
    response_structure = model.generate_content(prompt_structure)
    
    # LLM 응답(텍스트)에서 JSON 데이터만 추출
    # 이 단계에서 LLM이 항상 완벽한 JSON을 주지 않을 수 있어 오류 처리가 중요합니다.
    structured_list = extract_json_from_response(response_structure.text)
    
    # ❗️ 여기서 한 단계 더 필요:
    # 현재 'structured_list'는 [หมวด, ส่วน, มาตรา, มาตรา, ...] 와 같이 평평한 리스트입니다.
    # 이것을 실제로는 [หมวด: { sections: [ส่วน: { articles: [มาตรา, ...] }] }] 와 같이
    # 중첩된 트리 구조로 만들어야 합니다. 이 로직은 복잡하므로 다음 단계에서 구현합니다.
    # 지금은 LLM이 구조를 잘 뽑아주는지만 확인합니다.

    # 최종 JSON 조립
    final_json = {
      "global_summary": generated_global_summary,
      "document_title": "분석된 문서 (구조 분석 완료)",
      # 'chapters'에 평평한 리스트를 임시로 넣어서 확인
      "chapters": structured_list if structured_list else "구조 분석에 실패했습니다."
    }
    
    return final_json