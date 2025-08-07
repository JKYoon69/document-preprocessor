# document_processor.py

import google.generativeai as genai
import json
import time

# 헬퍼 함수: LLM 응답에서 JSON만 깔끔하게 추출
def extract_json_from_response(text):
    # LLM 응답에 설명이 붙는 경우, ```json ... ``` 부분만 찾아서 파싱
    if '```json' in text:
        start_index = text.find('```json') + len('```json')
        end_index = text.rfind('```')
        json_text = text[start_index:end_index].strip()
    else:
        # ```json``` 마커가 없는 경우, 중괄호 또는 대괄호로 시작하는 부분을 찾음
        first_bracket = text.find('{')
        first_square_bracket = text.find('[')
        
        if first_square_bracket != -1 and (first_square_bracket < first_bracket or first_bracket == -1):
            start_index = first_square_bracket
        elif first_bracket != -1:
            start_index = first_bracket
        else:
            return None # JSON 시작을 찾을 수 없음
        
        json_text = text[start_index:]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}\n원본 텍스트: {text}")
        return None

# ✅ 헬퍼 함수: 텍스트를 청크로 분할 (로직 수정)
def chunk_text(text, chunk_size=100000, overlap=20000):
    if len(text) <= chunk_size:
        return [{"text": text, "global_start": 0}]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({"text": text[start:end], "global_start": start})
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks
    
# ✅ 헬퍼 함수: 평평한 리스트를 계층적 트리로 변환
def build_tree_from_flat_list(flat_list, document_text):
    if not flat_list:
        return []

    # 'มาตรา'가 아닌 노드(chapter, section)를 상위 노드로 간주
    root_nodes = [node for node in flat_list if node.get('type') != 'article']
    articles = [node for node in flat_list if node.get('type') == 'article']

    for node in root_nodes:
        node['sections'] = [] # section을 담을 리스트 초기화
        node['articles'] = [] # chapter 바로 밑에 article이 오는 경우

    # 각 article을 가장 적절한 상위 노드에 할당
    for article in articles:
        parent = None
        for potential_parent in reversed(root_nodes):
            if potential_parent['global_start'] <= article['global_start']:
                parent = potential_parent
                break
        if parent:
            parent['articles'].append(article)

    # section들을 chapter에 할당
    chapters = [node for node in root_nodes if node.get('type') == 'chapter']
    sections = [node for node in root_nodes if node.get('type') == 'section']

    for section in sections:
        parent_chapter = None
        for chapter in reversed(chapters):
            if chapter['global_start'] <= section['global_start']:
                parent_chapter = chapter
                break
        if parent_chapter:
            parent_chapter['sections'].append(section)
            # section에 속한 article들을 section 객체로 이동
            section['articles'] = [art for art in parent_chapter['articles'] if art['global_start'] >= section['global_start'] and art['global_end'] <= section['global_end']]
            # chapter 직속 article 리스트에서는 제거
            parent_chapter['articles'] = [art for art in parent_chapter['articles'] if not (art['global_start'] >= section['global_start'] and art['global_end'] <= section['global_end'])]
            
    # 최종적으로 각 노드에 텍스트 채우기 및 불필요한 인덱스 제거
    for node in flat_list:
        node["text"] = document_text[node["global_start"]:node["global_end"]]
        del node["start_index"], node["end_index"], node["global_start"], node["global_end"]

    return chapters

# --- 메인 파이프라인 함수 ---
def run_pipeline(document_text, api_key, status_container):
    
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # ✅ 1. Global Summary 생성 (프롬프트 수정, JSON 요청 제거)
    status_container.write(f"1/4: **{model_name}** 모델로 **'전역 요약'** 생성...")
    preamble = document_text[:4000]
    prompt_global_summary = f"Please analyze the following preamble of a Thai legal document and summarize its purpose, background, and core principles in 2-3 sentences in Korean. Respond with the summary text only, without any JSON formatting.\n\n[Preamble text]\n{preamble}"
    response_summary = model.generate_content(prompt_global_summary)
    generated_global_summary = response_summary.text.strip()

    # ✅ 2. 청크 분할 (수정된 로직 적용)
    status_container.write(f"2/4: 문서 분할...")
    document_chunks = chunk_text(document_text)
    status_container.write(f"총 {len(document_chunks)}개의 청크로 분할 완료. 구조 분석을 시작합니다.")
    time.sleep(1)

    # ✅ 3. 각 청크별 구조 분석
    all_headers = []
    # 시스템 명령어를 각 프롬프트에 명시적으로 통합하여 명확성 증대
    prompt_structure_template = """As an expert in analyzing legal documents, your task is to return data in a clean, valid JSON format.
From the following text chunk of a Thai legal document, identify all hierarchical headers ('หมวด', 'ส่วน', 'มาตรา').
For each header, extract its type ('chapter', 'section', 'article'), title, and its character start/end positions within this chunk.
Return the result as a JSON array. Ensure every object in the array contains 'type', 'title', 'start_index', and 'end_index' keys.
If no headers are found, return an empty array [].

Example: [{"type": "chapter", "title": "หมวด 1", "start_index": 10, "end_index": 500}]

[Text Chunk]
{text_chunk}"""

    for i, chunk in enumerate(document_chunks):
        status_container.write(f"3/4: 청크 {i+1}/{len(document_chunks)} 분석 중...")
        try:
            final_prompt = prompt_structure_template.format(text_chunk=chunk["text"])
            response = model.generate_content(final_prompt)
            headers_in_chunk = extract_json_from_response(response.text)
            
            if headers_in_chunk and isinstance(headers_in_chunk, list):
                for header in headers_in_chunk:
                    if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                        header["global_start"] = header["start_index"] + chunk["global_start"]
                        header["global_end"] = header["end_index"] + chunk["global_start"]
                        all_headers.append(header)
        except Exception as e:
            status_container.error(f"청크 {i+1} 처리 중 오류: {e}")
            continue

    status_container.write("4/4: 구조 분석 완료. 결과 통합 및 트리 생성...")
    if not all_headers:
        return {"global_summary": generated_global_summary, "error": "문서 구조를 추출하지 못했습니다."}

    unique_headers = list({h['global_start']: h for h in sorted(all_headers, key=lambda x: x['global_start'])}.values())
    
    # ✅ 최종적으로 트리 구조로 변환
    hierarchical_chapters = build_tree_from_flat_list(unique_headers, document_text)
    
    final_json = {
      "global_summary": generated_global_summary,
      "document_title": f"분석된 문서 (모델: {model_name})",
      "chapters": hierarchical_chapters
    }
    
    return final_json