# document_processor.py

import google.generativeai as genai
import json
import time
import traceback # 상세한 오류 추적을 위해 추가

# 헬퍼 함수들 (이전과 동일)
def extract_json_from_response(text):
    try:
        if '```json' in text:
            start_index = text.find('```json') + len('```json')
            end_index = text.rfind('```')
            json_text = text[start_index:end_index].strip()
        else:
            first_bracket = text.find('{')
            first_square_bracket = text.find('[')
            if first_square_bracket != -1 and (first_square_bracket < first_bracket or first_bracket == -1):
                start_index = first_square_bracket
            elif first_bracket != -1:
                start_index = first_bracket
            else: return None
            json_text = text[start_index:]
        return json.loads(json_text)
    except json.JSONDecodeError: return None

def chunk_text(text, chunk_size=100000, overlap=20000):
    if len(text) <= chunk_size: return [{"text": text, "global_start": 0}]
    chunks = []; start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({"text": text[start:end], "global_start": start})
        if end == len(text): break
        start += chunk_size - overlap
    return chunks

def build_tree_from_flat_list(flat_list, document_text):
    if not flat_list: return []
    root_nodes = [node for node in flat_list if node.get('type') != 'article']
    articles = [node for node in flat_list if node.get('type') == 'article']
    for node in root_nodes:
        node['sections'] = []; node['articles'] = []
    for article in articles:
        parent = next((p for p in reversed(root_nodes) if p['global_start'] <= article['global_start']), None)
        if parent: parent['articles'].append(article)
    chapters = [n for n in root_nodes if n.get('type') == 'chapter']
    sections = [n for n in root_nodes if n.get('type') == 'section']
    for section in sections:
        parent_chapter = next((c for c in reversed(chapters) if c['global_start'] <= section['global_start']), None)
        if parent_chapter:
            parent_chapter['sections'].append(section)
            section['articles'] = [a for a in parent_chapter['articles'] if a['global_start'] >= section['global_start'] and a['global_end'] <= section['global_end']]
            parent_chapter['articles'] = [a for a in parent_chapter['articles'] if not (a['global_start'] >= section['global_start'] and a['global_end'] <= section['global_end'])]
    for node in flat_list:
        node["text"] = document_text[node["global_start"]:node["global_end"]]
        keys_to_del = ["start_index", "end_index", "global_start", "global_end"]
        for key in keys_to_del:
            if key in node: del node[key]
    return chapters

# --- 메인 파이프라인 함수 ---
def run_pipeline(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # 1. 전역 요약
    status_container.write(f"1/4: **{model_name}** 모델로 **'전역 요약'** 생성...")
    preamble = document_text[:4000]
    prompt_global_summary = f"Please analyze the following preamble of a Thai legal document and summarize its purpose, background, and core principles in 2-3 sentences in Korean. Respond with the summary text only, without any JSON formatting.\n\n[Preamble text]\n{preamble}"
    response_summary = model.generate_content(prompt_global_summary)
    generated_global_summary = response_summary.text.strip()

    # 2. 청크 분할
    status_container.write(f"2/4: 문서 분할...")
    document_chunks = chunk_text(document_text)
    status_container.write(f"총 {len(document_chunks)}개의 청크로 분할 완료. 구조 분석을 시작합니다.")

    # 3. 구조 분석
    all_headers = []
    prompt_structure_template = """As an expert in analyzing legal documents, your task is to return data in a clean, valid JSON format. From the following text chunk of a Thai legal document, identify all hierarchical headers ('หมวด', 'ส่วน', 'มาตรา'). For each header, extract its type ('chapter', 'section', 'article'), title, and its character start/end positions within this chunk. Return the result as a JSON array. Ensure every object in the array contains 'type', 'title', 'start_index', and 'end_index' keys. If no headers are found, return an empty array []. Example: [{"type": "chapter", "title": "หมวด 1", "start_index": 10, "end_index": 500}]\n\n[Text Chunk]\n{text_chunk}"""

    for i, chunk in enumerate(document_chunks):
        status_container.write(f"3/4: 청크 {i+1}/{len(document_chunks)} 분석 중...")
        try:
            final_prompt = prompt_structure_template.format(text_chunk=chunk["text"])
            response = model.generate_content(final_prompt)
            
            # ✅ 디버깅 강화: LLM의 원본 응답을 항상 보여주도록 변경
            st.info(f"청크 {i+1} LLM 원본 응답:\n```\n{response.text}\n```")
            
            headers_in_chunk = extract_json_from_response(response.text)
            
            if headers_in_chunk and isinstance(headers_in_chunk, list):
                for header in headers_in_chunk:
                    if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                        header["global_start"] = header["start_index"] + chunk["global_start"]
                        header["global_end"] = header["end_index"] + chunk["global_start"]
                        all_headers.append(header)
            else:
                st.warning(f"청크 {i+1}에서 유효한 JSON 리스트를 찾지 못했습니다.")

        except Exception as e:
            # ✅ 디버깅 강화: 어떤 종류의 오류인지, 어느 코드 라인에서 발생했는지 상세히 출력
            st.error(f"청크 {i+1} 처리 중 오류 발생!")
            st.error(f"오류 유형: {type(e).__name__}, 오류 메시지: {e}")
            # 전체 에러 스택을 텍스트로 변환하여 출력
            st.text(traceback.format_exc())
            continue

    status_container.write("4/4: 분석 완료. 결과 통합 및 트리 생성...")
    if not all_headers:
        return {"global_summary": generated_global_summary, "error": "문서 구조를 추출하지 못했습니다."}

    unique_headers = list({h['global_start']: h for h in sorted(all_headers, key=lambda x: x['global_start'])}.values())
    hierarchical_chapters = build_tree_from_flat_list(unique_headers, document_text)
    
    return {
      "global_summary": generated_global_summary,
      "document_title": f"분석된 문서 (모델: {model_name})",
      "chapters": hierarchical_chapters
    }