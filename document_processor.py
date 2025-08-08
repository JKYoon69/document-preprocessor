# document_processor.py
import google.generativeai as genai
import json
import traceback

# --- 기본 프롬프트 정의 ---

PROMPT_ARCHITECT = """You are a top-level document architect for Thai legal codes. Your mission is to identify ONLY the highest-level structural blocks.

1.  Analyze the entire document text provided.
2.  Identify all headers for 'ภาค' (book), 'ลักษณะ' (part), and 'หมวด' (chapter).
3.  **STRICTLY IGNORE** all lower-level headers like 'ส่วน' (section) and 'มาตรา' (article). Do not include them in your output.
4.  For each header found, create a JSON object with `type`, `title`, and its `start_index` within the full text.
5.  Return a single, flat JSON array of these objects. If no headers are found, return an empty array `[]`.

[DOCUMENT TEXT]
{text_chunk}"""

PROMPT_SURVEYOR = """You are a structural surveyor for a Thai legal chapter. Your mission is to map out the mid-level 'section' blocks within a given chapter.

1.  Analyze the provided text, which is a single chapter from a legal document.
2.  Identify all headers for 'ส่วน' (section).
3.  **STRICTLY IGNORE** 'มาตรา' (article) headers.
4.  For each 'ส่วน' (section) header found, create a JSON object with `type`: "section", `title`, and its `start_index` within the provided text.
5.  Return a single, flat JSON array of these objects. If no sections are found, return an empty array `[]`.

[CHAPTER TEXT]
{text_chunk}"""

PROMPT_DETAILER = """You are a meticulous clerk for a Thai legal section. Your mission is to find and list all 'article' blocks.

1.  Analyze the provided text, which is a single section or chapter from a legal document.
2.  Identify all headers for 'มาตรา' (article).
3.  For each 'มาตรา' (article) header found, create a JSON object with `type`: "article", `title`, and its `start_index` within the provided text.
4.  Return a single, flat JSON array of these objects. If no articles are found, return an empty array `[]`.

[SECTION/CHAPTER TEXT]
{text_chunk}"""


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

def postprocess_nodes(nodes, parent_text, global_offset=0):
    """노드 리스트를 후처리하여 인덱스를 보정하고 텍스트를 채웁니다."""
    if not nodes:
        return []

    parent_end = global_offset + len(parent_text)
    scoped_nodes = [
        node for node in nodes 
        if 'global_start' in node and global_offset <= node['global_start'] < parent_end
    ]

    unique_nodes = sorted(
        list({node['global_start']: node for node in scoped_nodes}.values()),
        key=lambda x: x['global_start']
    )

    for i in range(len(unique_nodes) - 1):
        unique_nodes[i]['global_end'] = unique_nodes[i+1]['global_start']

    if unique_nodes:
        unique_nodes[-1]['global_end'] = parent_end

    for node in unique_nodes:
        local_start = node['global_start'] - global_offset
        local_end = node['global_end'] - global_offset
        node['text'] = parent_text[local_start:local_end]
        node['children'] = []

    return unique_nodes

# --- Core Extraction Logic ---

def _extract_structure(text_chunk, global_offset, model, safety_settings, prompt_template, debug_info, step_name):
    """지정된 프롬프트를 사용하여 텍스트 청크에서 구조를 추출하는 범용 함수"""
    extracted_nodes = []
    try:
        prompt = prompt_template.format(text_chunk=text_chunk)
        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        response_text = response.text
        debug_info.append({f"{step_name}_response": response_text})

        nodes_in_chunk = extract_json_from_response(response_text)

        if isinstance(nodes_in_chunk, list):
            for node in nodes_in_chunk:
                if isinstance(node, dict) and all(k in node for k in ['type', 'title', 'start_index']):
                    node['global_start'] = node['start_index'] + global_offset
                    extracted_nodes.append(node)
        else:
            debug_info.append({f"{step_name}_parsing_error": "응답이 유효한 JSON 리스트가 아닙니다."})

    except Exception as e:
        debug_info.append({f"{step_name}_critical_error": traceback.format_exc()})

    return extracted_nodes

def run_pipeline(document_text, api_key, status_container, 
                 prompt_architect, prompt_surveyor, prompt_detailer):
    """3단계 계층적 파이프라인 실행"""
    model_name = 'gemini-2.5-flash'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    debug_info = []

    # === 1단계: 최상위 구조 (Architect) ===
    status_container.write(f"1/3: **Architect** - 문서 전체에서 최상위 구조(Book, Part, Chapter)를 추출합니다...")
    top_level_nodes_raw = _extract_structure(
        document_text, 0, model, safety_settings, 
        prompt_architect, debug_info, "step1_architect"
    )
    
    if not top_level_nodes_raw or top_level_nodes_raw[0].get('global_start', 0) > 0:
        preamble_node = {'type': 'preamble', 'title': 'Preamble', 'global_start': 0}
        top_level_nodes_raw.insert(0, preamble_node)
    
    top_level_nodes = postprocess_nodes(top_level_nodes_raw, document_text, 0)
    final_tree = top_level_nodes

    if not final_tree:
        return {"error": "1단계: 최상위 구조를 찾지 못했습니다."}, debug_info

    # === 2단계: 중간 구조 (Surveyor) ===
    status_container.write(f"2/3: **Surveyor** - {len(final_tree)}개의 최상위 구조 내부에서 중간 구조(Section)를 추출합니다...")
    for i, parent_node in enumerate(final_tree):
        # Preamble 같이 하위 구조가 없는 노드는 건너뜀
        if not parent_node.get('text', '').strip() or parent_node['type'] == 'preamble':
            continue
        
        mid_level_nodes_raw = _extract_structure(
            parent_node['text'], parent_node['global_start'], model, safety_settings,
            prompt_surveyor, debug_info, f"step2_surveyor_parent_{i+1}"
        )
        mid_level_nodes = postprocess_nodes(mid_level_nodes_raw, parent_node['text'], parent_node['global_start'])
        parent_node['children'] = mid_level_nodes

    # === 3단계: 최하위 구조 (Detailer) ===
    status_container.write(f"3/3: **Detailer** - 하위 구조(Article)를 추출하여 최종 트리를 완성합니다...")
    nodes_to_traverse = list(final_tree)
    while nodes_to_traverse:
        current_node = nodes_to_traverse.pop(0)
        
        # Section이 있으면 Section 내부에서 Article을 찾고, 없으면 Chapter 등에서 바로 Article을 찾음
        if current_node.get('children'):
            nodes_to_traverse.extend(current_node['children'])
            continue

        if not current_node.get('text', '').strip() or current_node['type'] == 'preamble':
            continue

        low_level_nodes_raw = _extract_structure(
            current_node['text'], current_node['global_start'], model, safety_settings,
            prompt_detailer, debug_info, f"step3_detailer_parent_{current_node['title']}"
        )
        low_level_nodes = postprocess_nodes(low_level_nodes_raw, current_node['text'], current_node['global_start'])
        current_node['children'] = low_level_nodes

    return {"tree": final_tree}, debug_info