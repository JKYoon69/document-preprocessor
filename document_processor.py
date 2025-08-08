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

# 검증 완료된 의미 기반 청킹 함수
def chunk_text_semantic(text, chunk_size_chars=100000, overlap_chars=20000):
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text), "text": text, "global_start": 0}]
    chunks, start_char = [], 0
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

# ✅✅✅ 새로운 계층적 파이프라인 함수 ✅✅✅
def run_hierarchical_pipeline_step1(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    # 1. 전역 요약 생성
    status_container.write("1/4: 문서 전체 요약 생성 중...")
    preamble_text = document_text[:4000]
    prompt_global = f"Summarize the preamble of this Thai legal document in Korean.\n\n[Preamble]\n{preamble_text}"
    response_summary = model.generate_content(prompt_global, safety_settings=safety_settings)
    global_summary = response_summary.text.strip()

    # 2. 의미 기반 청킹
    status_container.write("2/4: 문서를 의미 기반 청크로 분할 중...")
    chunks = chunk_text_semantic(document_text)
    
    # 3. 슬라이딩 윈도우 방식으로 Chapter 정보 추출
    all_chapters = []
    
    # Chapter 추출에 최적화된 새로운 프롬프트
    prompt_chapter_extraction = """You are an expert in Thai legal document analysis. Your task is to identify ONLY the main Chapters ('หมวด') from the provided text.

Follow these rules:
1.  Identify the introductory text before the first 'หมวด' as 'chapter0' (Preamble).
2.  For each 'หมวด', create a JSON object with 'title', a Korean 'summary' (2-3 sentences), 'start_index', and 'end_index'.
3.  The `end_index` for a chapter is the character right before the next chapter begins.
4.  If there is concluding text after the last 'หมวด', identify it as 'chapterF' (Notes/References).
5.  Return ONLY a valid JSON array of these chapter objects.

[Text Chunk]
{text_chunk}"""

    # 슬라이딩 윈도우 생성
    windows = []
    if len(chunks) == 1:
        windows.append(chunks[0])
    else:
        for i in range(len(chunks) - 1):
            combined_text = chunks[i]['text'] + chunks[i+1]['text'][chunks[i+1]['global_start'] - (chunks[i]['global_start'] + len(chunks[i]['text'])):]
            windows.append({'text': combined_text, 'global_start': chunks[i]['global_start']})

    status_container.write("3/4: 슬라이딩 윈도우 방식으로 Chapter 분석 실행...")
    for i, window in enumerate(windows):
        status_container.write(f"... 윈도우 {i+1}/{len(windows)} 분석 중...")
        prompt = prompt_chapter_extraction.format(text_chunk=window["text"])
        response = model.generate_content(prompt, safety_settings=safety_settings)
        chapters_in_window = extract_json_from_response(response.text)
        
        if isinstance(chapters_in_window, list):
            for chap in chapters_in_window:
                if isinstance(chap, dict) and all(k in chap for k in ['title', 'summary', 'start_index', 'end_index']):
                    # 로컬 인덱스를 전역 인덱스로 변환
                    chap["global_start"] = chap["start_index"] + window["global_start"]
                    chap["global_end"] = chap["end_index"] + window["global_start"]
                    all_chapters.append(chap)

    # 4. 결과 통합 및 후처리
    status_container.write("4/4: Chapter 정보 통합 및 중복 제거...")
    if not all_chapters:
        raise ValueError("LLM이 문서에서 Chapter 구조를 추출하지 못했습니다.")
        
    # 중복 제거 (global_start 기준으로 고유한 chapter만 남김)
    unique_chapters = sorted(list({c['global_start']: c for c in all_chapters}.values()), key=lambda x: x['global_start'])

    # 인덱스 연결성 보장
    for i in range(len(unique_chapters) - 1):
        unique_chapters[i]["global_end"] = unique_chapters[i+1]["global_start"]
    if unique_chapters:
        unique_chapters[-1]["global_end"] = len(document_text)

    # 임시 키 제거
    for chap in unique_chapters:
        if "start_index" in chap: del chap["start_index"]
        if "end_index" in chap: del chap["end_index"]

    return {
        "global_summary": global_summary,
        "chapters": unique_chapters
    }