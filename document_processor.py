# document_processor.py (최종 완성 버전)

import google.generativeai as genai
import json
from collections import Counter

# 헬퍼 함수: LLM 응답에서 JSON 추출
def extract_json_from_response(text):
    if '```json' in text:
        try: return json.loads(text.split('```json', 1)[1].split('```', 1)[0].strip())
        except (json.JSONDecodeError, IndexError): return None
    try: return json.loads(text)
    except json.JSONDecodeError: return None

# 검증 완료된 의미 기반 청킹 함수
def chunk_text_semantic(text, chunk_size_chars=100000, overlap_chars=20000):
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text), "text": text}]
    chunks, start_char = [], 0
    while start_char < len(text):
        ideal_end = start_char + chunk_size_chars
        actual_end = min(ideal_end, len(text))
        if ideal_end >= len(text):
            chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end]})
            break
        separators = ["\n\n", ". ", " ", ""]
        best_sep_pos = -1
        for sep in separators:
            search_start = max(start_char, ideal_end - 5000)
            best_sep_pos = text.rfind(sep, search_start, ideal_end)
            if best_sep_pos != -1:
                actual_end = best_sep_pos + len(sep)
                break
        if best_sep_pos == -1: actual_end = ideal_end
        chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end]})
        start_char = actual_end - overlap_chars
    return chunks

# ✅✅✅ 전체 파이프라인을 실행하는 최종 메인 함수 ✅✅✅
def run_full_pipeline(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # --- 1단계: 전역 요약 생성 ---
    status_container.write("1/4: 문서 전역 요약 생성 중...")
    try:
        preamble = document_text[:4000]
        prompt_global = f"Summarize the following Thai legal document preamble in Korean. Respond with the summary text only.\n\n[Preamble]\n{preamble}"
        response_summary = model.generate_content(prompt_global)
        global_summary = response_summary.text.strip()
    except Exception as e:
        global_summary = f"전역 요약 생성 중 오류 발생: {e}"

    # --- 2단계: 구조 분석 및 통합 ---
    # 2-1. 청크 분할
    status_container.write("2/4: 문서를 의미 기반 청크로 분할 중...")
    chunks = chunk_text_semantic(document_text)
    status_container.write(f"분할 완료: 총 {len(chunks)}개 청크 생성됨.")

    # 2-2. 각 청크별 구조 분석
    all_headers = []
    chunk_stats = []
    prompt_structure = """As an expert JSON generator, analyze the following Thai legal text chunk. Identify all headers like 'หมวด', 'ส่วน', 'มาตรา'. For each, return a JSON object with 'type', 'title', 'start_index', and 'end_index'. Return a JSON array of these objects. If no headers are found, return an empty array [].
[Text Chunk]
{text_chunk}"""
    
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        status_container.write(f"3/4: 청크 {chunk_num}/{len(chunks)} 구조 분석 중...")
        try:
            prompt = prompt_structure.format(text_chunk=chunk["text"])
            response = model.generate_content(prompt)
            headers_in_chunk = extract_json_from_response(response.text)
            
            if isinstance(headers_in_chunk, list):
                counts = Counter(h.get('type', 'unknown') for h in headers_in_chunk)
                chunk_stats.append({"청크 번호": chunk_num, "chapter": counts.get('หมวด', 0), "section": counts.get('ส่วน', 0), "article": counts.get('มาตรา', 0)})
                
                for header in headers_in_chunk:
                    if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                        header["global_start"] = header["start_index"] + chunk["global_start"]
                        header["global_end"] = header["end_index"] + chunk["global_start"]
                        all_headers.append(header)
        except Exception as e:
            status_container.error(f"청크 {chunk_num} 처리 중 오류: {e}")
            continue

    # 2-3. 중복 제거 및 최종 통계 계산
    status_container.write("4/4: 결과 통합 및 중복 제거 중...")
    original_counts = Counter(h.get('type', 'unknown') for h in all_headers)
    unique_headers_map = {(h['global_start'], h['title']): h for h in all_headers}
    unique_headers = sorted(list(unique_headers_map.values()), key=lambda x: x['global_start'])
    final_counts = Counter(h.get('type', 'unknown') for h in unique_headers)
    duplicate_counts = {
        "chapter": original_counts.get('หมวด', 0) - final_counts.get('หมวด', 0),
        "section": original_counts.get('ส่วน', 0) - final_counts.get('ส่วน', 0),
        "article": original_counts.get('มาตรา', 0) - final_counts.get('มาตรา', 0)
    }

    # 다운로드할 최종 결과 데이터 구성
    final_result_data = {
        "global_summary": global_summary,
        "document_title": "Document Analysis Result",
        "structure": unique_headers # 아직 평탄화된 구조
    }

    # 화면에 표시할 통계 데이터 구성
    stats_data = {
        "chunk_stats": chunk_stats,
        "duplicate_counts": duplicate_counts,
        "final_counts": {
            "chapter": final_counts.get('หมวด', 0),
            "section": final_counts.get('ส่วน', 0),
            "article": final_counts.get('มาตรา', 0)
        }
    }

    return final_result_data, stats_data