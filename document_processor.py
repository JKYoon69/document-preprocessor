# document_processor.py

import google.generativeai as genai
import json
import traceback

# 헬퍼 함수: LLM 응답에서 JSON 추출 (안정성 강화)
def extract_json_from_response(text):
    if '```json' in text:
        try:
            return json.loads(text.split('```json', 1)[1].split('```', 1)[0].strip())
        except (json.JSONDecodeError, IndexError):
            return None
    
    try: # 코드 블록이 없는 경우, 순수 JSON으로 가정
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# ✅✅✅ 헬퍼 함수: 청크 분할 로직 (완전히 새로 작성) ✅✅✅
def chunk_text(text, chunk_size=100000, overlap_ratio=0.2):
    if len(text) <= chunk_size:
        return [{"text": text, "global_start": 0}]

    chunks = []
    overlap = int(chunk_size * overlap_ratio)
    start = 0

    while True:
        end = start + chunk_size
        # 현재 청크의 텍스트와 시작 위치 저장
        chunk_text = text[start:end]
        chunks.append({"text": chunk_text, "global_start": start})
        
        # 다음 청크의 시작 위치 계산
        start += (chunk_size - overlap)
        
        # 다음 청크가 문서 끝을 넘어서면 반복 중단
        if start + overlap >= len(text):
            # 남은 텍스트가 의미 있는 길이라면 마지막 청크로 추가
            if len(text) - start > overlap:
                chunks.append({"text": text[start:], "global_start": start})
            break
            
    return chunks


# --- 메인 파이프라인 함수 (안정성 및 디버깅 강화) ---
def run_pipeline(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    debug_info = []

    # 1. 전역 요약
    status_container.write(f"1/4: **{model_name}** 모델로 **'전역 요약'** 생성 중...")
    try:
        preamble = document_text[:4000]
        prompt_global_summary = f"Summarize the following Thai legal document preamble in Korean. Respond with only the summary text.\n\n[Preamble]\n{preamble}"
        response_summary = model.generate_content(prompt_global_summary)
        generated_global_summary = response_summary.text.strip()
    except Exception as e:
        generated_global_summary = f"전역 요약 생성 중 오류 발생: {e}"

    # 2. 청크 분할
    status_container.write("2/4: 문서를 청크로 분할 중...")
    document_chunks = chunk_text(document_text) # ✅ 새로 작성된 함수 호출
    status_container.write(f"분할 완료: **총 {len(document_chunks)}개**의 청크 생성됨.")

    # 3. 각 청크별 구조 분석
    all_headers = []
    prompt_structure_template = """As an expert JSON generator, analyze the following Thai legal text chunk. Identify all headers like 'หมวด', 'ส่วน', 'มาตรา'. For each, return a JSON object with 'type', 'title', 'start_index', and 'end_index'. Return a JSON array of these objects. If no headers are found, return an empty array [].

[Text Chunk]
{text_chunk}"""

    for i, chunk in enumerate(document_chunks):
        chunk_num = i + 1
        status_container.write(f"3/4: **청크 {chunk_num}/{len(document_chunks)}** (전역 위치: {chunk['global_start']} 부터) 분석 중...")
        
        try:
            final_prompt = prompt_structure_template.format(text_chunk=chunk["text"])
            response = model.generate_content(final_prompt)
            
            # ✅ 디버깅 정보에 프롬프트와 응답을 모두 기록
            debug_info.append({
                f"chunk_{chunk_num}_prompt_snippet": final_prompt[:500] + "...",
                f"chunk_{chunk_num}_response": response.text
            })
            
            headers_in_chunk = extract_json_from_response(response.text)
            
            if isinstance(headers_in_chunk, list):
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

    # 4. 결과 통합
    status_container.write("4/4: 분석 완료. 결과 통합 중...")
    if not all_headers:
        return {
            "global_summary": generated_global_summary,
            "document_title": f"분석된 문서 (모델: {model_name})",
            "error": "문서 구조를 추출하지 못했습니다.",
            "debug_info": debug_info
        }

    unique_headers = list({h['global_start']: h for h in sorted(all_headers, key=lambda x: x['global_start'])}.values())
    
    # 지금은 안정성을 위해 평탄화된 리스트를 반환합니다.
    for header in unique_headers:
        header["text"] = document_text[header["global_start"]:header["global_end"]]

    return {
      "global_summary": generated_global_summary,
      "document_title": f"분석된 문서 (모델: {model_name})",
      "chapters": unique_headers, # 트리 변환은 다음 단계에서
      "debug_info": debug_info
    }