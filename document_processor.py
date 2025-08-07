# document_processor.py

import google.generativeai as genai
import json

# 헬퍼 함수 (이전과 동일)
def extract_json_from_response(text):
    try:
        start = text.find('```json') + len('```json')
        end = text.rfind('```')
        if start != -1 and end != -1:
            json_text = text[start:end].strip()
            return json.loads(json_text)
        else:
            return json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"원본 텍스트: {text}")
        return None

# --- 메인 파이프라인 함수 ---
# 👇 status_container 인자가 추가되었습니다.
def run_pipeline(document_text, api_key, status_container):
    
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # 1. Global Summary 생성
    # 👇 사용자에게 현재 작업 내용을 알려줍니다.
    status_container.write(f"1/2: **{model_name}** 모델을 호출하여 문서 전체의 **'전역 요약'**을 생성합니다...")
    
    preamble = document_text[:3000]
    prompt_global_summary = f"""
        Analyze the preamble of the Thai legal document provided below.
        Summarize the document's purpose, background, and core principles in 2-3 sentences in Korean.
        [Preamble text]
        {preamble}
    """
    response_summary = model.generate_content(prompt_global_summary)
    generated_global_summary = response_summary.text.strip()

    # 2. 문서 전체 구조 분석
    # 👇 두 번째 작업 내용을 알려줍니다.
    status_container.write(f"2/2: **{model_name}** 모델을 호출하여 문서의 **'전체 구조(목차)'**를 분석합니다. 이 작업은 시간이 오래 걸릴 수 있습니다...")
    
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
    
    response_structure = model.generate_content(prompt_structure)
    structured_list = extract_json_from_response(response_structure.text)
    
    # 최종 JSON 조립
    final_json = {
      "global_summary": generated_global_summary,
      "document_title": f"분석된 문서 (모델: {model_name})",
      "chapters": structured_list if structured_list else "구조 분석에 실패했습니다."
    }
    
    return final_json