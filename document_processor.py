# document_processor.py

import google.generativeai as genai
import json
import traceback

def extract_json_from_response(text):
    # LLM 응답에서 JSON 코드 블록을 우선적으로 찾음
    if '```json' in text:
        start_index = text.find('```json') + len('```json')
        end_index = text.rfind('```')
        json_text = text[start_index:end_index].strip()
    else:
        # 코드 블록이 없으면, 첫 대괄호 '[' 를 기준으로 파싱 시도 (JSON 배열을 기대)
        start_index = text.find('[')
        if start_index == -1: return None
        json_text = text[start_index:]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None

# ✅ 청크 분할 로직을 더 단순하고 정확하게 수정
def chunk_text(text, chunk_size=100000, overlap=20000):
    if len(text) <= chunk_size:
        return [{"text": text, "global_start": 0}]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({"text": text[start:end], "global_start": start})
        start += chunk_size - overlap
    return chunks

def build_tree_from_flat_list(flat_list, document_text):
    # (이전 버전의 트리 구축 로직은 복잡하고 오류 가능성이 있어, 우선 평탄화된 리스트를 반환하는 것으로 안정화합니다.)
    # LLM으로부터 받은 구조 리스트에 실제 텍스트를 추가해서 반환
    for header in flat_list:
        if "global_start" in header and "global_end" in header:
            header["text"] = document_text[header["global_start"]:header["global_end"]]
    return flat_list


# --- 메인 파이프라인 함수 (전면 재구성) ---
def run_pipeline(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # 디버깅 정보를 담을 리스트
    debug_info = []

    # 1. 전역 요약
    status_container.write(f"1/4: **{model_name}** 모델로 **'전역 요약'** 생성 중...")
    preamble = document_text[:4000]
    prompt_global_summary = f"Summarize the following Thai legal document preamble in 2-3 sentences in Korean. Respond with only the summary text.\n\n[Preamble]\n{preamble}"
    
    try:
        response_summary = model.generate_content(prompt_global_summary)
        generated_global_summary = response_summary.text.strip()
    except Exception as e:
        generated_global_summary = f"전역 요약 생성 중 오류 발생: {e}"

    # 2. 청크 분할
    status_container.write("2/4: 문서를 청크로 분할 중...")
    document_chunks = chunk_text(document_text)
    status_container.write(f"분할 완료: **총 {len(document_chunks)}개**의 청크 생성됨.")

    # 3. 각 청크별 구조 분석
    all_headers = []
    prompt_structure_template = """As an expert JSON generator, analyze the following Thai legal text chunk. Identify all headers like 'หมวด', 'ส่วน', 'มาตรา'. For each, return a JSON object with 'type', 'title', 'start_index', and 'end_index'. Return a JSON array of these objects. If no headers are found, return an empty array [].

[Text Chunk]
{text_chunk}"""

    for i, chunk in enumerate(document_chunks):
        chunk_num = i + 1
        status_container.write(f"3/4: **청크 {chunk_num}/{len(document_chunks)}** (크기: {len(chunk['text'])} 바이트) 분석 중...")
        
        try:
            final_prompt = prompt_structure_template.format(text_chunk=chunk["text"])
            response = model.generate_content(final_prompt)
            
            # ✅ LLM의 모든 원본 응답을 디버깅 정보에 기록
            debug_info.append({f"chunk_{chunk_num}_response": response.text})
            
            headers_in_chunk = extract_json_from_response(response.text)
            
            if headers_in_chunk and isinstance(headers_in_chunk, list):
                for header in headers_in_chunk:
                    # ✅ 더 강력한 타입 및 키 존재 여부 확인
                    if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                        header["global_start"] = header["start_index"] + chunk["global_start"]
                        header["global_end"] = header["end_index"] + chunk["global_start"]
                        all_headers.append(header)
            else:
                debug_info.append({f"chunk_{chunk_num}_parsing_error": "Returned data is not a valid list of objects."})

        except Exception as e:
            debug_info.append({f"chunk_{chunk_num}_critical_error": traceback.format_exc()})
            continue # 오류 발생 시 다음 청크로 진행

    # 4. 결과 통합
    status_container.write("4/4: 분석 완료. 결과 통합 중...")
    if not all_headers:
        return {
            "global_summary": generated_global_summary,
            "document_title": f"분석된 문서 (모델: {model_name})",
            "error": "문서 구조를 추출하지 못했습니다.",
            "debug_info": debug_info
        }

    # 중복 제거 및 정렬
    unique_headers = list({h['global_start']: h for h in sorted(all_headers, key=lambda x: x['global_start'])}.values())
    
    # 최종 결과 생성 (안정화를 위해 우선 평탄화된 리스트 사용)
    final_result = build_tree_from_flat_list(unique_headers, document_text)
    
    return {
      "global_summary": generated_global_summary,
      "document_title": f"분석된 문서 (모델: {model_name})",
      "chapters": final_result,
      "debug_info": debug_info
    }