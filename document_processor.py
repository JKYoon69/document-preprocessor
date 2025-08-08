# document_processor.py
import google.generativeai as genai
import json
import traceback
import time
from google.api_core.exceptions import InternalServerError

# ==============================================================================
# [ CONFIGURATION ] - 모델명과 프롬프트를 이곳에서 쉽게 수정하세요.
# ==============================================================================
MODEL_NAME = "gemini-1.5-flash"

TYPE_MAPPING = {
    "ภาค": "book",
    "ลักษณะ": "part",
    "หมวด": "chapter",
    "ส่วน": "section",
    "มาตรา": "article"
}

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
# ==============================================================================
# [ END OF CONFIGURATION ]
# ==============================================================================


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
    """LLM API를 호출하고 결과를 추출하며, 재시도 및 시간 측정을 포함합니다."""
    extracted_nodes = []
    duration = 0
    retries = 3
    for attempt in range(retries):
        try:
            prompt = prompt_template.format(text_chunk=text_chunk)
            start_time = time.perf_counter()
            response = model.generate_content(prompt, safety_settings=safety_settings)
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            response_text = response.text
            debug_info.append({f"{step_name}_response": response_text, "llm_duration_seconds": duration})

            nodes_in_chunk = extract_json_from_response(response_text)

            if isinstance(nodes_in_chunk, list):
                for node in nodes_in_chunk:
                    if isinstance(node, dict) and all(k in node for k in ['type', 'title', 'start_index']):
                        node['type'] = TYPE_MAPPING.get(node['type'], node['type'])
                        node['global_start'] = node['start_index'] + global_offset
                        extracted_nodes.append(node)
            else:
                debug_info.append({f"{step_name}_parsing_error": "응답이 유효한 JSON 리스트가 아닙니다."})
            
            return extracted_nodes # 성공 시 루프 탈출
        
        except InternalServerError as e:
            debug_info.append({f"{step_name}_retryable_error": f"Attempt {attempt + 1} failed: {e}"})
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e
        except Exception as e:
            debug_info.append({f"{step_name}_critical_error": traceback.format_exc()})
            return []
    return []

def run_pipeline(document_text, api_key, status_container, 
                 prompt_architect, prompt_surveyor, prompt_detailer,
                 intermediate_callback=None):
    """3단계 계층적 파이프라인을 실행합니다."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    debug_info = []
    timings = {}

    # === 1단계: Architect ===
    status_container.write(f"1/3: **Architect** - 최상위 구조 추출 중...")
    step1_start = time.perf_counter()
    top_level_nodes_raw = _extract_structure(
        document_text, 0, model, safety_settings, 
        prompt_architect, debug_info, "step1_architect"
    )
    
    if not top_level_nodes_raw or top_level_nodes_raw[0].get('global_start', 0) > 0:
        preamble_node = {'type': 'preamble', 'title': 'Preamble', 'global_start': 0}
        top_level_nodes_raw.insert(0, preamble_node)
    
    top_level_nodes = postprocess_nodes(top_level_nodes_raw, document_text, 0)
    final_tree = top_level_nodes
    step1_end = time.perf_counter()
    timings["step1_architect_duration"] = step1_end - step1_start

    if intermediate_callback:
        intermediate_callback(top_level_nodes, status_container, debug_info)
    
    if not final_tree:
        return {"error": "1단계: 최상위 구조를 찾지 못했습니다."}, debug_info

    # === 2단계: Surveyor ===
    status_container.write(f"2/3: **Surveyor** - 중간 구조 추출 중...")
    step2_start = time.perf_counter()
    for i, parent_node in enumerate(final_tree):
        if not parent_node.get('text', '').strip() or parent_node['type'] == 'preamble':
            continue
        
        mid_level_nodes_raw = _extract_structure(
            parent_node['text'], parent_node['global_start'], model, safety_settings,
            prompt_surveyor, debug_info, f"step2_surveyor_parent_{i+1}"
        )
        mid_level_nodes = postprocess_nodes(mid_level_nodes_raw, parent_node['text'], parent_node['global_start'])
        parent_node['children'] = mid_level_nodes
    step2_end = time.perf_counter()
    timings["step2_surveyor_duration"] = step2_end - step2_start

    # === 3단계: Detailer ===
    status_container.write(f"3/3: **Detailer** - 최하위 구조 추출 중...")
    step3_start = time.perf_counter()
    queue = list(final_tree)
    while queue:
        current_node = queue.pop(0)
        
        if current_node.get('children'):
            queue.extend(current_node['children'])
            continue

        if not current_node.get('text', '').strip() or current_node['type'] in ['preamble', 'article']:
            continue

        low_level_nodes_raw = _extract_structure(
            current_node['text'], current_node['global_start'], model, safety_settings,
            prompt_detailer, debug_info, f"step3_detailer_parent_{current_node.get('title', 'node')[:20]}"
        )
        low_level_nodes = postprocess_nodes(low_level_nodes_raw, current_node['text'], current_node['global_start'])
        current_node['children'] = low_level_nodes
    step3_end = time.perf_counter()
    timings["step3_detailer_duration"] = step3_end - step3_start
    
    debug_info.append({"performance_timings": timings})
    return {"tree": final_tree}, debug_info