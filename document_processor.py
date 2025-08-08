# document_processor.py
import google.generativeai as genai
import json
import traceback
import time

# --- Helper Functions ---

def extract_json_from_response(text):
    """LLM 응답에서 JSON 코드 블록을 안전하게 추출합니다."""
    if '```json' in text:
        try:
            return json.loads(text.split('```json', 1)[1].split('```', 1)[0].strip())
        except (json.JSONDecodeError, IndexError):
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def chunk_text_semantic(text, chunk_size_chars=30000, overlap_chars=3000):
    """
    텍스트를 의미 단위(문단, 문장)를 존중하며 지정된 크기로 청킹합니다.
    """
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
        
        search_start = max(start_char, ideal_end - 500)
        for sep in separators:
            best_sep_pos = text.rfind(sep, search_start, ideal_end)
            if best_sep_pos != -1:
                actual_end = best_sep_pos + len(sep)
                break
        
        if best_sep_pos == -1:
            actual_end = ideal_end

        chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end], "global_start": start_char})
        
        next_start = actual_end - overlap_chars
        if next_start <= (chunks[-1]['start_char'] if chunks else 0):
             start_char = actual_end
        else:
             start_char = next_start
             
    return chunks

def postprocess_nodes(nodes, parent_text, global_offset=0):
    """노드 리스트를 후처리하여 인덱스를 보정하고 텍스트를 채웁니다."""
    if not nodes:
        return []

    # global_start 키가 없는 비정상적인 노드 필터링 및 정렬
    unique_nodes = sorted(
        list({node['global_start']: node for node in nodes if 'global_start' in node}.values()),
        key=lambda x: x['global_start']
    )

    for i in range(len(unique_nodes) - 1):
        unique_nodes[i]['global_end'] = unique_nodes[i+1]['global_start']

    if unique_nodes:
        unique_nodes[-1]['global_end'] = global_offset + len(parent_text)

    for node in unique_nodes:
        local_start = node['global_start'] - global_offset
        local_end = node['global_end'] - global_offset
        node['text'] = parent_text[local_start:local_end]
        node['children'] = []

    return unique_nodes

# --- Core Extraction Logic ---

def _extract_structure(text_chunk, global_offset, model, safety_settings, prompt_template, debug_info, chunk_name):
    """지정된 프롬프트를 사용하여 텍스트 청크에서 구조를 추출하는 범용 함수"""
    extracted_nodes = []
    try:
        prompt = prompt_template.format(text_chunk=text_chunk)
        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        response_text = response.text
        debug_info.append({f"{chunk_name}_response": response_text})

        nodes_in_chunk = extract_json_from_response(response_text)

        if isinstance(nodes_in_chunk, list):
            for node in nodes_in_chunk:
                if isinstance(node, dict) and all(k in node for k in ['type', 'title', 'start_index']):
                    node['global_start'] = node['start_index'] + global_offset
                    extracted_nodes.append(node)
        else:
            debug_info.append({f"{chunk_name}_parsing_error": "응답이 유효한 JSON 리스트가 아닙니다."})

    except Exception as e:
        debug_info.append({f"{chunk_name}_critical_error": traceback.format_exc()})

    return extracted_nodes

def run_hybrid_pipeline(document_text, api_key, status_container):
    """Hybrid (Top-down + Post-processing) 파이프라인 실행"""
    # ▼▼▼ 모델명 수정 지점 ▼▼▼
    model_name = 'gemini-2.5-flash' 
    # ▲▲▲ 모델명 수정 지점 ▲▲▲
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    debug_info = []
    final_tree = []

    PROMPT_CHAPTERS = """You are a legal document parser for Thai law.
Analyze the provided text and identify ONLY the top-level structural headers: 'ภาค' (book), 'ลักษณะ' (part), 'หมวด' (chapter).
For each header, provide a JSON object with:
1. `type`: Mapped as 'ภาค'->'book', 'ลักษณะ'->'part', 'หมวด'->'chapter'.
2. `title`: The full header text (e.g., "หมวด ๑ บททั่วไป").
3. `start_index`: The starting character position of the header title within the given text.

Return a flat JSON array of these objects.
[Text Chunk]
{text_chunk}"""

    PROMPT_CHILDREN = """You are a legal document parser for Thai law.
Analyze the provided text, which is a single chapter. Identify all 'ส่วน' (section) and 'มาตรา' (article) headers within it.
For each header, provide a JSON object with:
1. `type`: Mapped as 'ส่วน'->'section', 'มาตรา'->'article'.
2. `title`: The full header text (e.g., "มาตรา ๑").
3. `start_index`: The starting character position of the header title within the given text.

Return a flat JSON array of these objects.
[Text Chunk]
{text_chunk}"""

    status_container.write("1/5: 문서를 의미 단위로 청킹하는 중...")
    chunks = chunk_text_semantic(document_text, chunk_size_chars=30000, overlap_chars=3000)
    chunking_details = [
        {"chunk": i+1, "global_start": c["global_start"], "length": len(c["text"])} 
        for i, c in enumerate(chunks)
    ]
    debug_info.append({"chunking_details": chunking_details})

    status_container.write(f"2/5: 최상위 구조 분석 중 ({len(chunks)}개 청크)...")
    top_level_nodes_raw = []
    for i, chunk in enumerate(chunks):
        status_container.write(f"청크 {i+1}/{len(chunks)}에서 Chapter 검색...")
        nodes_in_chunk = _extract_structure(
            chunk["text"], chunk["global_start"], model, safety_settings, 
            PROMPT_CHAPTERS, debug_info, f"chunk_{i+1}_chapters"
        )
        top_level_nodes_raw.extend(nodes_in_chunk)

    status_container.write("3/5: 추출된 Chapter 구조 후처리 및 보정 중...")
    
    unique_nodes_by_start = {node['global_start']: node for node in top_level_nodes_raw if 'global_start' in node}
    sorted_unique_nodes = sorted(unique_nodes_by_start.values(), key=lambda x: x['global_start'])

    if not sorted_unique_nodes or sorted_unique_nodes[0]['global_start'] > 0:
        preamble_node = {'type': 'preamble', 'title': 'Preamble', 'global_start': 0}
        sorted_unique_nodes.insert(0, preamble_node)
    
    top_level_nodes = postprocess_nodes(sorted_unique_nodes, document_text, 0)
    if not top_level_nodes:
        return {"error": "Failed to find any top-level structure."}, debug_info

    status_container.write(f"4/5: 총 {len(top_level_nodes)}개 Chapter 내부 구조 분석 중...")
    for i, parent_node in enumerate(top_level_nodes):
        status_container.write(f"Chapter {i+1}/{len(top_level_nodes)} ('{parent_node['title']}') 분석...")
        if parent_node['type'] == 'preamble':
            final_tree.append(parent_node)
            continue
        
        children_nodes_raw = _extract_structure(
            parent_node['text'], parent_node['global_start'], model, safety_settings, 
            PROMPT_CHILDREN, debug_info, f"parent_{i+1}_{parent_node['type']}"
        )
        
        if children_nodes_raw:
            children_nodes = postprocess_nodes(
                children_nodes_raw, parent_node['text'], parent_node['global_start']
            )
            parent_node['children'] = children_nodes
        
        final_tree.append(parent_node)

    status_container.write("5/5: 최종 트리 구조 완성!")
    return {"tree": final_tree}, debug_info