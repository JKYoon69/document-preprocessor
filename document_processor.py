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

# 전체 파이프라인
def run_full_pipeline(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    debug_info = []
    all_headers = []
    chunk_stats = []

    # 1. 전역 요약 생성
    status_container.write("1/4: 문서 전역 요약 생성 중...")
    try:
        preamble_text = document_text[:4000]
        prompt_global = f"Summarize the following Thai legal document preamble in Korean. Respond with the summary text only.\n\n[Preamble]\n{preamble_text}"
        
        start_time = time.time()
        response_summary = model.generate_content(prompt_global)
        end_time = time.time()
        
        global_summary = response_summary.text.strip()
        debug_info.append({"global_summary_response_time": f"{end_time - start_time:.2f} 초"})
        
    except Exception as e:
        global_summary = f"전역 요약 생성 중 오류 발생: {e}"
        debug_info.append({"global_summary_error": traceback.format_exc()})

    # 2. 청크 분할
    status_container.write("2/4: 문서를 의미 기반 청크로 분할 중...")
    chunks = chunk_text_semantic(document_text)
    status_container.write(f"분할 완료: 총 {len(chunks)}개 청크 생성됨.")
    
    # 3. 각 청크별 구조 분석
    # ✅✅✅ 프롬프트 전면 수정 ✅✅✅
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
7.  Return a single JSON array of these objects.

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
        status_container.write(f"3/4: 청크 {chunk_num}/{len(chunks)} 구조 분석 중...")
        try:
            prompt = prompt_structure.format(text_chunk=chunk["text"])
            
            start_time = time.time()
            response = model.generate_content(prompt)
            end_time = time.time()
            
            debug_info.append({
                f"chunk_{chunk_num}_response_time": f"{end_time - start_time:.2f} 초",
                f"chunk_{chunk_num}_response": response.text
            })

            headers_in_chunk = extract_json_from_response(response.text)
            
            if isinstance(headers_in_chunk, list):
                counts = Counter(h.get('type', 'unknown') for h in headers_in_chunk)
                chunk_stats.append({"청크 번호": chunk_num, "preamble": counts.get('preamble',0), "book": counts.get('book',0), "part": counts.get('part',0), "chapter": counts.get('chapter', 0), "section": counts.get('section', 0), "article": counts.get('article', 0)})
                
                for header in headers_in_chunk:
                    if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                        header["global_start"] = header["start_index"] + chunk["global_start"]
                        header["global_end"] = header["end_index"] + chunk["global_start"]
                        all_headers.append(header)
            else:
                debug_info.append({f"chunk_{chunk_num}_parsing_error": "응답이 유효한 JSON 리스트가 아닙니다."})
        except Exception as e:
            status_container.error(f"청크 {chunk_num} 처리 중 오류: {traceback.format_exc()}")
            debug_info.append({f"chunk_{chunk_num}_critical_error": traceback.format_exc()})
            continue

    # 4. 결과 통합 및 중복 제거
    status_container.write("4/4: 결과 통합 및 중복 제거 중...")
    
    unique_headers, duplicate_counts, final_counts = [], {}, {}
    try:
        original_counts = Counter(h.get('type', 'unknown') for h in all_headers)
        # global_start를 기준으로 딕셔너리를 만들어 중복 제거 (같은 위치에 여러 태그가 붙는 경우 첫번째 것만 선택됨)
        unique_headers_map = {h['global_start']: h for h in sorted(all_headers, key=lambda x: x['global_start'])}
        unique_headers = list(unique_headers_map.values())
        
        final_counts = Counter(h.get('type', 'unknown') for h in unique_headers)
        duplicate_counts = {
            "preamble": original_counts.get('preamble', 0) - final_counts.get('preamble', 0),
            "book": original_counts.get('book', 0) - final_counts.get('book', 0),
            "part": original_counts.get('part', 0) - final_counts.get('part', 0),
            "chapter": original_counts.get('chapter', 0) - final_counts.get('chapter', 0),
            "section": original_counts.get('section', 0) - final_counts.get('section', 0),
            "article": original_counts.get('article', 0) - final_counts.get('article', 0)
        }
    except KeyError as e:
        debug_info.append({"deduplication_error": f"중복 제거 중 키 오류 발생: {e}"})

    final_result_data = {
        "global_summary": global_summary,
        "structure": unique_headers
    }
    stats_data = {
        "chunk_stats": chunk_stats,
        "duplicate_counts": duplicate_counts,
        "final_counts": final_counts
    }

    return final_result_data, stats_data, debug_info