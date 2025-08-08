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
            # JSON 파싱 실패 시, 텍스트 자체를 로드 시도
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def postprocess_nodes(nodes, parent_text, global_offset=0):
    """
    노드 리스트를 후처리하여 인덱스를 보정하고 텍스트를 채웁니다.
    1. global_start 기준으로 정렬 및 중복 제거
    2. end_index를 다음 노드의 start_index로 설정하여 내용 누락 방지
    3. 각 노드에 해당하는 원본 텍스트 추가
    """
    if not nodes:
        return []

    # global_start를 기준으로 중복 제거 및 정렬
    unique_nodes = sorted(
        list({node['global_start']: node for node in nodes}.values()),
        key=lambda x: x['global_start']
    )

    # end_index 보정
    for i in range(len(unique_nodes) - 1):
        unique_nodes[i]['global_end'] = unique_nodes[i+1]['global_start']

    # 마지막 노드의 end_index는 부모 텍스트의 끝으로 설정
    if unique_nodes:
        unique_nodes[-1]['global_end'] = global_offset + len(parent_text)

    # 각 노드에 텍스트 채우기
    for node in unique_nodes:
        # global_start/end는 전체 문서 기준, 텍스트 슬라이싱은 parent_text 기준
        local_start = node['global_start'] - global_offset
        local_end = node['global_end'] - global_offset
        node['text'] = parent_text[local_start:local_end]
        node['children'] = [] # 하위 노드를 담을 리스트 초기화

    return unique_nodes


# --- Core Extraction Logic ---

def _extract_structure(text_chunk, global_offset, model, safety_settings, prompt_template, debug_info, chunk_name):
    """지정된 프롬프트를 사용하여 텍스트 청크에서 구조를 추출하는 범용 함수"""
    extracted_nodes = []
    try:
        prompt = prompt_template.format(text_chunk=text_chunk)
        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        # 디버깅을 위해 원본 응답 저장
        response_text = response.text
        debug_info.append({f"{chunk_name}_response": response_text})

        nodes_in_chunk = extract_json_from_response(response_text)

        if isinstance(nodes_in_chunk, list):
            for node in nodes_in_chunk:
                if isinstance(node, dict) and all(k in node for k in ['type', 'title', 'start_index']):
                    # 청크 내부 인덱스를 전역 인덱스로 변환
                    node['global_start'] = node['start_index'] + global_offset
                    # end_index는 후처리에서 결정하므로 여기서는 사용 안 함
                    extracted_nodes.append(node)
        else:
            debug_info.append({f"{chunk_name}_parsing_error": "응답이 유효한 JSON 리스트가 아닙니다."})

    except Exception as e:
        debug_info.append({f"{chunk_name}_critical_error": traceback.format_exc()})

    return extracted_nodes


def run_hybrid_pipeline(document_text, api_key, status_container):
    """
    Hybrid (Top-down + Post-processing) 파이프라인 실행
    1. 최상위 구조(Chapter 등)를 먼저 추출하고 인덱스를 보정.
    2. 각 최상위 구조 내부에서 하위 구조(Section, Article)를 추출하고 인덱스를 보정.
    """
    model_name = 'gemini-1.5-flash' # 모델명 변경 (gemini-2.5-flash -> gemini-1.5-flash)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    debug_info = []
    final_tree = []
    doc_len = len(document_text)

    # --- 프롬프트 정의 ---
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

    # --- 1단계: 최상위 구조 (Chapter 등) 추출 및 보정 ---
    status_container.write(f"1/3: **{model_name}** 모델로 최상위 구조(Chapter 등) 분석 중...")
    
    # 문서를 청킹하지 않고 전체를 한 번에 처리 (Gemini 1.5의 긴 컨텍스트 활용)
    top_level_nodes_raw = _extract_structure(document_text, 0, model, safety_settings, PROMPT_CHAPTERS, debug_info, "full_doc_chapters")
    
    # 서문(Preamble) 추가
    if not top_level_nodes_raw or top_level_nodes_raw[0]['global_start'] > 0:
        preamble_node = {
            'type': 'preamble', 
            'title': 'Preamble', 
            'global_start': 0,
            # end_index는 후처리에서 결정됨
        }
        top_level_nodes_raw.insert(0, preamble_node)
    
    # 후처리를 통해 top_level_nodes의 end_index 보정 및 텍스트 채우기
    status_container.write("2/3: 추출된 Chapter 구조 후처리 및 보정 중...")
    top_level_nodes = postprocess_nodes(top_level_nodes_raw, document_text, 0)

    if not top_level_nodes:
        status_container.error("문서에서 최상위 구조를 찾지 못했습니다.")
        return {"error": "Failed to find any top-level structure."}, debug_info

    # --- 2단계: 각 Chapter 내부의 하위 구조 (Section, Article) 추출 ---
    status_container.write(f"3/3: 총 {len(top_level_nodes)}개의 Chapter 내부 구조 분석 중...")
    
    for i, parent_node in enumerate(top_level_nodes):
        status_container.write(f"Chapter {i+1}/{len(top_level_nodes)} ('{parent_node['title']}') 분석...")
        
        # Preamble은 하위 구조 분석 생략
        if parent_node['type'] == 'preamble':
            final_tree.append(parent_node)
            continue

        # LLM으로 자식 노드 추출
        children_nodes_raw = _extract_structure(
            parent_node['text'], 
            parent_node['global_start'], 
            model, 
            safety_settings, 
            PROMPT_CHILDREN, 
            debug_info,
            f"parent_{i+1}_{parent_node['type']}"
        )
        
        # 후처리를 통해 자식 노드들의 인덱스 보정 및 텍스트 채우기
        if children_nodes_raw:
            children_nodes = postprocess_nodes(
                children_nodes_raw, 
                parent_node['text'], 
                parent_node['global_start']
            )
            parent_node['children'] = children_nodes
        
        final_tree.append(parent_node)

    return {"tree": final_tree}, debug_info