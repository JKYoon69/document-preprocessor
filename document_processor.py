# document_processor.py

import google.generativeai as genai
import json
import time # 각 단계별 시간 측정을 위해 추가

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

# ⭐️ 새로운 헬퍼 함수: 텍스트를 청크로 분할
def chunk_text(text, chunk_size=100000, overlap=10000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # 청크의 끝이 문서의 끝을 넘지 않도록 조정
        if end > len(text):
            end = len(text)
        
        chunks.append({
            "text": text[start:end],
            "global_start": start
        })
        
        # 다음 청크의 시작 위치를 overlap을 고려하여 설정
        start += chunk_size - overlap
        if start >= len(text):
            break
            
    return chunks

# --- 메인 파이프라인 함수 (대대적으로 수정됨) ---
def run_pipeline(document_text, api_key, status_container):
    
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # 1. Global Summary 생성 (이전과 동일, 속도 빠름)
    status_container.write(f"1/3: **{model_name}** 모델로 **'전역 요약'**을 생성합니다...")
    preamble = document_text[:3000]
    prompt_global_summary = f"""Analyze the preamble of the Thai legal document provided below. Summarize the document's purpose, background, and core principles in 2-3 sentences in Korean. [Preamble text]\n{preamble}"""
    response_summary = model.generate_content(prompt_global_summary)
    generated_global_summary = response_summary.text.strip()

    # 2. ⭐️ 청크 분할
    status_container.write(f"2/3: 문서가 크므로 작은 조각(청크)으로 분할합니다...")
    document_chunks = chunk_text(document_text)
    status_container.write(f"총 {len(document_chunks)}개의 청크로 분할되었습니다. 각 청크별로 구조 분석을 시작합니다.")
    time.sleep(1) # 사용자가 메시지를 읽을 시간을 줍니다.

    # 3. ⭐️ 각 청크별 구조 분석 및 결과 병합
    all_headers = []
    
    # 👇 새로운 구조 분석 프롬프트 (인덱스만 요청)
    prompt_structure_index_only = """
        The following text is part of a Thai legal document. Identify all hierarchical headers such as 'หมวด', 'ส่วน', and 'มาตรา'.
        For each identified element, extract its type, title, and its character start/end positions within the provided text.
        Return the result as a JSON array.
        
        Example format:
        [
          { "type": "chapter", "title": "หมวด 1", "start_index": 10, "end_index": 500 },
          { "type": "article", "title": "มาตรา 1", "start_index": 30, "end_index": 150 }
        ]
        
        [Text Chunk]
        {text_chunk}
    """

    for i, chunk in enumerate(document_chunks):
        status_container.write(f"3/{len(document_chunks)+2}: 청크 {i+1}/{len(document_chunks)}를 분석 중입니다...")
        
        # LLM 호출
        response = model.generate_content(prompt_structure_index_only.format(text_chunk=chunk["text"]))
        headers_in_chunk = extract_json_from_response(response.text)
        
        if headers_in_chunk:
            for header in headers_in_chunk:
                # 로컬 인덱스를 전역 인덱스로 변환
                header["global_start"] = header["start_index"] + chunk["global_start"]
                header["global_end"] = header["end_index"] + chunk["global_start"]
                all_headers.append(header)

    # 4. ⭐️ 중복 제거 및 정렬 (나중에 트리구조 만들 때 중요)
    status_container.write("모든 청크 분석 완료. 결과 병합 및 중복을 제거합니다...")
    # global_start와 title을 기준으로 중복 제거
    unique_headers = list({ (h['global_start'], h['title']): h for h in all_headers }.values())
    # 시작 위치 기준으로 정렬
    sorted_headers = sorted(unique_headers, key=lambda x: x['global_start'])

    # 5. ⭐️ 최종 결과 조립 (이제 텍스트는 최종 단계에서 추가)
    # 각 헤더에 실제 텍스트 내용을 추가합니다. (LLM 호출 없이 Python으로 처리)
    for header in sorted_headers:
        header["text"] = document_text[header["global_start"]:header["global_end"]]

    final_json = {
      "global_summary": generated_global_summary,
      "document_title": f"분석된 문서 (모델: {model_name})",
      # 아직 평평한 리스트지만, 이제는 훨씬 빠르고 안정적으로 생성됩니다.
      "chapters": sorted_headers
    }
    
    return final_json