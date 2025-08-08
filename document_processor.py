# document_processor.py

import google.generativeai as genai
import json
import traceback
from collections import Counter
import time

# 헬퍼 함수: LLM 응답에서 JSON 추출
def extract_json_from_response(text):
    if '```json' in text:
        try:
            return json.loads(text.split('```json', 1)[1].split('```', 1)[0].strip())
        except (json.JSONDecodeError, IndexError):
            return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# 헬퍼 함수: 의미 기반 청킹
def chunk_text_semantic(text, chunk_size_chars=100000, overlap_chars=20000):
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text), "text": text, "global_start": 0}]

    chunks = []
    start_char = 0
    while start_char < len(text):
        ideal_end = start_char + chunk_size_chars
        actual_end = min(ideal_end, len(text))
        
        if ideal_end >= len(text):
            chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end], "global_start": start_char})
            break

        separators = ["\n\n", ". ", " ", ""]
        best_sep_pos = -1
        for sep in separators:
            search_start = max(start_char, ideal_end - 5000)
            best_sep_pos = text.rfind(sep, search_start, ideal_end)
            if best_sep_pos != -1:
                actual_end = best_sep_pos + len(sep)
                break
        if best_sep_pos == -1:
            actual_end = ideal_end

        chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end], "global_start": start_char})
        start_char = actual_end - overlap_chars
    return chunks

# 구조 추출에 집중하는 파이프라인
def run_extraction_pipeline(document_text, api_key, status_container):
    # ✅✅✅ 모델 이름을 여기서 변경 ✅✅✅
    model_name = 'gemini-2.5-flash' 
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    debug_info = []
    all_headers = []
    chunk_stats = []
    global_summary = ""

    # 1. 전역 요약 생성
    status_container.write(f"1/3: **{model_name}** 모델로 전역 요약 생성 중...")
    try:
        preamble = document_text[:4000]
        prompt_global = f"Summarize the preamble of this Thai legal document in Korean.\n\n[Preamble]\n{preamble}"
        response_summary = model.generate_content(prompt_global)
        global_summary = response_summary.text.strip()
    except Exception as e:
        global_summary = f"전역 요약 생성 중 오류: {e}"
        debug_info.append({"global_summary_error": traceback.format_exc()})

    # 2. 청킹 및 구조 분석
    status_container.write("2/3: 문서를 청킹하고 구조 분석 실행 중...")
    chunks = chunk_text_semantic(document_text)
    
    prompt_structure = """You are a highly accurate legal document parsing tool. Your task is to analyze the following chunk of a Thai legal document and identify all structural elements.

Follow these rules with extreme precision:
1.  Identify the introductory text before the first formal article as `preamble`.
2.  Identify all headers such as 'ภาค', 'ลักษณะ', 'หมวด', 'ส่วน', and 'มาตรา'.
3.  For each element, create a JSON object.
4.  The `title` field MUST contain ONLY the short header text (e.g., "มาตรา ๑").
5.  The `end_index` of an element MUST extend to the character right before the `start_index` of the NEXT element. If it is the last element in the chunk, its `end_index` is the end of the chunk. This is crucial to avoid missing any text.
6.  Map the Thai header to the `type` field using these exact rules:
    - Text before 'มาตรา ๑' -> 'preamble'
    - 'ภาค' -> 'book'
    - 'ลักษณะ' -> 'part'
    - 'หมวด' -> 'chapter'
    - 'ส่วน' -> 'section'
    - 'มาตรา' -> 'article'
7.  Return a single JSON array of these objects. If no headers are found, return an empty array [].

Example of expected output for a chunk:
[
  {{
    "type": "preamble",
    "title": "Preamble",
    "start_index": 0,
    "end_index": 533
  }},
  {{
    "type": "article",
    "title": "มาตรา ๑",
    "start_index": 534,
    "end_index": 611
  }},
  {{
    "type": "article",
    "title": "มาตรา ๒",
    "start_index": 612,
    "end_index": 688
  }}
]

[Text Chunk]
{text_chunk}"""
    
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        status_container.write(f"청크 {chunk_num}/{len(chunks)} 분석 중...")
        try:
            prompt = prompt_structure.format(text_chunk=chunk["text"])
            response = model.generate_content(prompt)
            debug_info.append({f"chunk_{chunk_num}_response": response.text})
            headers_in_chunk = extract_json_from_response(response.text)
            
            if isinstance(headers_in_chunk, list) and headers_in_chunk:
                counts = Counter(h.get('type', 'unknown') for h in headers_in_chunk)
                chunk_stats.append({"Chunk Number": chunk_num, "book": counts.get('book',0), "part": counts.get('part',0), "chapter": counts.get('chapter', 0), "section": counts.get('section', 0), "article": counts.get('article', 0)})
                for header in headers_in_chunk:
                    if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                        header["global_start"] = header["start_index"] + chunk["global_start"]
                        header["global_end"] = header["end_index"] + chunk["global_start"]
                        all_headers.append(header)
            else:
                debug_info.append({f"chunk_{chunk_num}_parsing_error": "응답이 유효한 JSON 리스트가 아닙니다."})
        except Exception as e:
            debug_info.append({f"chunk_{chunk_num}_critical_error": traceback.format_exc()})
            continue

    # 3. 결과 통합
    status_container.write("3/3: 결과 통합 및 후처리 중...")
    
    if not all_headers:
        return {"global_summary": global_summary, "structure": []}, debug_info

    unique_headers = sorted(list({h['global_start']: h for h in all_headers}.values()), key=lambda x: x['global_start'])
    
    if unique_headers and unique_headers[0]["global_start"] > 0:
        unique_headers.insert(0, {"type": "preamble", "title": "Preamble", "global_start": 0, "global_end": unique_headers[0]["global_start"]})
    for i in range(len(unique_headers) - 1):
        unique_headers[i]["global_end"] = unique_headers[i+1]["global_start"]
    if unique_headers:
        unique_headers[-1]["global_end"] = len(document_text)

    for header in unique_headers:
        header["text"] = document_text[header["global_start"]:header["global_end"]]

    return {"global_summary": global_summary, "structure": unique_headers}, debug_info