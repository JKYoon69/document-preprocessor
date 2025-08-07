# document_processor.py

import google.generativeai as genai
import json
import time

# 헬퍼 함수: LLM 응답에서 JSON 추출 (이전과 동일)
def extract_json_from_response(text):
    try:
        start = text.find('```json') + len('```json')
        end = text.rfind('```')
        if start > -1 and end > -1:
            json_text = text[start:end].strip()
            return json.loads(json_text)
        else:
            return json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"JSON 파싱 오류: {e}\n원본 텍스트: {text}")
        return None

# ⭐️ 헬퍼 함수: 텍스트를 청크로 분할 (100kb / 20% 중첩으로 수정)
def chunk_text(text, chunk_size=100000, overlap=20000): # 100kb, 20% overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append({"text": text[start:end], "global_start": start})
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks

# --- 메인 파이프라인 함수 ---
def run_pipeline(document_text, api_key, status_container):
    
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    
    system_instruction = "You are an expert in analyzing legal documents and your primary task is to return data in a clean, valid JSON format."
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_instruction
    )

    # 1. Global Summary 생성
    status_container.write(f"1/3: **{model_name}** 모델로 **'전역 요약'**을 생성합니다...")
    preamble = document_text[:3000]
    prompt_global_summary = f"""Analyze the preamble of the Thai legal document provided below. Summarize the document's purpose, background, and core principles in 2-3 sentences in Korean.\n\n[Preamble text]\n{preamble}"""
    response_summary = model.generate_content(prompt_global_summary)
    generated_global_summary = response_summary.text.strip()

    # 2. 청크 분할
    status_container.write(f"2/3: 문서 분할...")
    document_chunks = chunk_text(document_text) # 수정된 chunk_text 함수 사용
    status_container.write(f"총 {len(document_chunks)}개의 청크로 분할되었습니다. 각 청크별로 구조 분석을 시작합니다.")
    time.sleep(1)

    # 3. 각 청크별 구조 분석 및 결과 병합
    all_headers = []
    prompt_structure_template = """The following text is part of a Thai legal document. Identify all hierarchical headers such as 'หมวด', 'ส่วน', and 'มาตรา'.
For each identified element, extract its type, title, and its character start/end positions within the provided text.
Return the result as a JSON array. Ensure every object in the array contains 'type', 'title', 'start_index', and 'end_index' keys.

Example format:
[{"type": "chapter", "title": "หมวด 1", "start_index": 10, "end_index": 500}]

[Text Chunk]
{text_chunk}"""

    for i, chunk in enumerate(document_chunks):
        status_container.write(f"3/{len(document_chunks)+2}: 청크 {i+1}/{len(document_chunks)}를 분석 중입니다...")
        
        try:
            # ⭐️⭐️⭐️ 여기가 핵심 수정 부분: .format()을 사용하여 프롬프트에 실제 청크 텍스트를 삽입
            final_prompt = prompt_structure_template.format(text_chunk=chunk["text"])
            
            response = model.generate_content(final_prompt)
            
            # 디버깅을 위해 원본 응답을 계속 표시
            status_container.info(f"청크 {i+1}에 대한 LLM 원본 응답:\n```\n{response.text}\n```")
            
            headers_in_chunk = extract_json_from_response(response.text)
            
            if headers_in_chunk and isinstance(headers_in_chunk, list):
                for header in headers_in_chunk:
                    if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                        header["global_start"] = header["start_index"] + chunk["global_start"]
                        header["global_end"] = header["end_index"] + chunk["global_start"]
                        all_headers.append(header)
                    else:
                        status_container.warning(f"청크 {i+1}에서 예상과 다른 형식의 응답을 받았습니다. 건너뜁니다: {header}")
            else:
                 status_container.warning(f"청크 {i+1}에서 유효한 리스트 형식의 JSON을 받지 못했습니다.")

        except Exception as e:
            status_container.error(f"청크 {i+1} 처리 중 예상치 못한 오류 발생: {e}")
            continue

    status_container.write("모든 청크 분석 완료. 결과 병합 및 중복을 제거합니다...")
    if not all_headers:
        return {
            "global_summary": generated_global_summary,
            "document_title": f"분석된 문서 (모델: {model_name})",
            "error": "문서 구조를 추출하지 못했습니다. LLM 응답을 확인해주세요."
        }

    unique_headers = list({(h['global_start'], h['title']): h for h in all_headers}.values())
    sorted_headers = sorted(unique_headers, key=lambda x: x['global_start'])

    for header in sorted_headers:
        header["text"] = document_text[header["global_start"]:header["global_end"]]

    final_json = {
      "global_summary": generated_global_summary,
      "document_title": f"분석된 문서 (모델: {model_name})",
      "chapters": sorted_headers
    }
    
    return final_json