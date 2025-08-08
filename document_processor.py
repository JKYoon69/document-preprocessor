# document_processor.py
import google.generativeai as genai
import json
import traceback
import time
from google.api_core.exceptions import InternalServerError

# ==============================================================================
# [ CONFIGURATION ]
# ==============================================================================
MODEL_NAME = "gemini-1.5-flash"

# 타입 매핑: LLM이 태국어로 타입을 반환할 경우 영어로 변환하기 위함
TYPE_MAPPING = {
    "ภาค": "book",
    "ลักษณะ": "part",
    "หมวด": "chapter",
    "ส่วน": "section",
    "มาตรา": "article"
}

PROMPT_ARCHITECT = "..." # 이전과 동일
PROMPT_SURVEYOR = "..."  # 이전과 동일
PROMPT_DETAILER = "..."  # 이전과 동일
# ==============================================================================
# [ END OF CONFIGURATION ]
# ==============================================================================

def extract_json_from_response(text):
    # ... 이전과 동일 ...

def postprocess_nodes(nodes, parent_text, global_offset=0):
    # ... 이전과 동일 ...

def _extract_structure(text_chunk, global_offset, model, safety_settings, prompt_template, debug_info, step_name):
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
            debug_info.append({f"{step_name}_response": response_text, "llm_duration": duration})

            nodes_in_chunk = extract_json_from_response(response_text)

            if isinstance(nodes_in_chunk, list):
                for node in nodes_in_chunk:
                    if isinstance(node, dict) and all(k in node for k in ['type', 'title', 'start_index']):
                        # 타입 매핑 적용
                        node['type'] = TYPE_MAPPING.get(node['type'], node['type'])
                        node['global_start'] = node['start_index'] + global_offset
                        extracted_nodes.append(node)
            else:
                debug_info.append({f"{step_name}_parsing_error": "응답이 유효한 JSON 리스트가 아닙니다."})
            
            return extracted_nodes # 성공 시 루프 탈출
        
        except InternalServerError as e:
            debug_info.append({f"{step_name}_retryable_error": f"Attempt {attempt + 1} failed: {e}"})
            if attempt < retries - 1:
                time.sleep(2) # 재시도 전 대기
            else:
                raise e # 마지막 시도도 실패하면 에러 발생
        
        except Exception as e:
            debug_info.append({f"{step_name}_critical_error": traceback.format_exc()})
            return [] # 치명적 에러 시 빈 리스트 반환
    
    return [] # 모든 재시도 실패 시

def run_pipeline(document_text, api_key, status_container, 
                 prompt_architect, prompt_surveyor, prompt_detailer,
                 intermediate_callback=None):
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    safety_settings = { ... } # 이전과 동일
    
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
        intermediate_callback(top_level_nodes, status_container)
    
    if not final_tree:
        return {"error": "1단계: 최상위 구조를 찾지 못했습니다."}, debug_info, timings

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