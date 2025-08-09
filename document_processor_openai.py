# document_processor.py
# This file contains the logic for processing documents using the Google Gemini API.
# VERSION: "Trust but Verify" Algorithm Applied

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import traceback
import time

# ==============================================================================
# [ CONFIGURATION ] - Gemini Version
# ==============================================================================
MODEL_NAME = "gemini-2.5-flash"
DETAIL_CHUNK_SIZE_THRESHOLD = 30000

TYPE_MAPPING = {
    "ภาค": "book", "ลักษณะ": "part", "หมวด": "chapter",
    "ส่วน": "section", "มาตรา": "article"
}

PROMPT_ARCHITECT = """You are a top-level document architect for Thai legal codes. Your mission is to identify ONLY the highest-level structural blocks.
1. Analyze the entire document text provided.
2. Identify all headers for 'ภาค' (book), 'ลักษณะ' (part), and 'หมวด' (chapter).
3. **STRICTLY IGNORE** all lower-level headers like 'ส่วน' (section) and 'มาตรา' (article).
4. For each header found, create a JSON object with `type`, `title`, and its `start_index`.
5. Return a single, flat JSON array of these objects. If no headers are found, return an empty array `[]`.

[DOCUMENT TEXT]
{text_chunk}"""

PROMPT_SURVEYOR = """You are a structural surveyor for a Thai legal chapter. Your mission is to map out the mid-level 'section' blocks within a given chapter.
1. Analyze the provided text, which is a single chapter from a legal document.
2. Identify all headers for 'ส่วน' (section).
3. **STRICTLY IGNORE** 'มาตรา' (article) headers.
4. For each 'ส่วน' (section) header found, create a JSON object with `type`: "section", `title`, and its `start_index`.
5. Return a single, flat JSON array of these objects. If no sections are found, return an empty array `[]`.

[CHAPTER TEXT]
{text_chunk}"""

PROMPT_DETAILER = """You are a meticulous clerk for a Thai legal section. Your mission is to find and list all 'article' blocks.
1. Analyze the provided text, which is a single section or chapter from a legal document.
2. Identify all headers for 'มาตรา' (article).
3. For each 'มาตรา' (article) header found, create a JSON object with `type`: "article", `title`, and its `start_index`.
4. Return a single, flat JSON array of these objects. If no articles are found, return an empty array `[]`.

[SECTION/CHAPTER TEXT]
{text_chunk}"""
# ==============================================================================
# [ END OF CONFIGURATION ]
# ==============================================================================

# Helper functions (model-agnostic)
def extract_json_from_response(text):
    if not text:
        return None
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
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "text": text}]
    chunks = []
    start_char = 0
    while start_char < len(text):
        ideal_end = start_char + chunk_size_chars
        if ideal_end >= len(text):
            chunks.append({"start_char": start_char, "text": text[start_char:]})
            break
        separators = ["\n\n", ". ", " ", ""]
        actual_end = -1
        for sep in separators:
            actual_end = text.rfind(sep, start_char, ideal_end)
            if actual_end != -1:
                break
        actual_end = ideal_end if actual_end == -1 else actual_end + len(sep)
        chunks.append({"start_char": start_char, "text": text[start_char:actual_end]})
        start_char = actual_end - overlap_chars
    return chunks

def postprocess_nodes(nodes, parent_text, global_offset=0):
    if not nodes: return []
    parent_end = global_offset + len(parent_text)
    scoped_nodes = [node for node in nodes if 'global_start' in node and global_offset <= node['global_start'] < parent_end]
    unique_nodes = sorted(list({node['global_start']: node for node in scoped_nodes}.values()), key=lambda x: x['global_start'])
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

# CORE API CALL LOGIC (FOR GEMINI)
def _extract_structure_gemini(text_chunk, global_offset, model, prompt_template, debug_info, step_name):
    retries = 3
    for attempt in range(retries):
        try:
            prompt = prompt_template.format(text_chunk=text_chunk)
            start_time = time.perf_counter()
            
            response = model.generate_content(prompt)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            response_text = response.text
            debug_info.append({f"{step_name}_response": response_text, "llm_duration_seconds": duration})

            nodes_in_chunk = extract_json_from_response(response_text)
            
            # ==================================================================
            # [ START OF ALGORITHM HARDENING - "Trust but Verify" ]
            # ==================================================================
            validated_nodes = []
            if isinstance(nodes_in_chunk, list):
                for node in nodes_in_chunk:
                    if not (isinstance(node, dict) and all(k in node for k in ['type', 'title', 'start_index'])):
                        debug_info.append({f"{step_name}_validation_skip": f"Skipping malformed node: {node}"})
                        continue
                    
                    local_start_index = node['start_index']
                    title_from_llm = node['title']

                    if not (0 <= local_start_index < len(text_chunk)):
                        debug_info.append({f"{step_name}_validation_fail_reason": "Index out of bounds", "failed_node": node})
                        continue

                    is_at_line_start = (local_start_index == 0) or (text_chunk[local_start_index - 1] == '\n')
                    if not is_at_line_start:
                        debug_info.append({f"{step_name}_validation_fail_reason": "Node not at line start", "failed_node": node})
                        continue

                    actual_text_at_index = text_chunk[local_start_index : local_start_index + len(title_from_llm)]
                    if not actual_text_at_index.strip() == title_from_llm.strip():
                        debug_info.append({
                            f"{step_name}_validation_fail_reason": "Title mismatch with source text",
                            "llm_title": title_from_llm, "actual_text": actual_text_at_index, "failed_node": node
                        })
                        continue
                    
                    node['type'] = TYPE_MAPPING.get(node['type'], node['type'])
                    node['global_start'] = node['start_index'] + global_offset
                    validated_nodes.append(node)
            else:
                 if response_text:
                    debug_info.append({f"{step_name}_parsing_error": "Response was not a valid JSON list."})
            
            return validated_nodes
            # ==================================================================
            # [ END OF ALGORITHM HARDENING ]
            # ==================================================================

        except Exception as e:
            debug_info.append({f"{step_name}_error": f"Attempt {attempt + 1} failed: {e}", "trace": traceback.format_exc()})
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e
    return []

# MAIN PIPELINE FUNCTION (FOR GEMINI)
def run_gemini_pipeline(document_text, api_key, status_container, 
                        prompt_architect, prompt_surveyor, prompt_detailer,
                        debug_info, intermediate_callback=None):
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        MODEL_NAME,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        },
        generation_config=genai.GenerationConfig(response_mime_type="application/json")
    )

    timings = {}

    status_container.write(f"1/3: **Architect (Gemini)** - Extracting top-level structure...")
    step1_start = time.perf_counter()
    top_level_nodes_raw = _extract_structure_gemini(document_text, 0, model, prompt_architect, debug_info, "step1_architect")
    
    if not top_level_nodes_raw or top_level_nodes_raw[0].get('global_start', 0) > 0:
        top_level_nodes_raw.insert(0, {'type': 'preamble', 'title': 'Preamble', 'global_start': 0})
    
    final_tree = postprocess_nodes(top_level_nodes_raw, document_text, 0)
    step1_end = time.perf_counter()
    timings["step1_architect_duration"] = step1_end - step1_start

    if intermediate_callback:
        intermediate_callback(final_tree)
    
    if not final_tree:
        return {"error": "Step 1 failed: Could not find any top-level structure."}

    status_container.write(f"2/3: **Surveyor (Gemini)** - Extracting mid-level structure...")
    step2_start = time.perf_counter()
    for i, parent_node in enumerate(final_tree):
        if not parent_node.get('text', '').strip() or parent_node['type'] == 'preamble': continue
        mid_level_nodes_raw = _extract_structure_gemini(parent_node['text'], parent_node['global_start'], model, prompt_surveyor, debug_info, f"step2_surveyor_parent_{i+1}")
        parent_node['children'] = postprocess_nodes(mid_level_nodes_raw, parent_node['text'], parent_node['global_start'])
    step2_end = time.perf_counter()
    timings["step2_surveyor_duration"] = step2_end - step2_start

    status_container.write(f"3/3: **Detailer (Gemini)** - Extracting lowest-level structure...")
    step3_start = time.perf_counter()
    
    def process_recursively(nodes):
        for i, node in enumerate(nodes):
            if node.get('children'):
                process_recursively(node['children'])
            elif node.get('text', '').strip() and node['type'] not in ['preamble', 'article']:
                all_articles_raw = []
                node_text = node['text']
                node_offset = node['global_start']
                
                sub_chunks = chunk_text_semantic(node_text, chunk_size_chars=DETAIL_CHUNK_SIZE_THRESHOLD) if len(node_text) > DETAIL_CHUNK_SIZE_THRESHOLD else [{'start_char': 0, 'text': node_text}]
                
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk_offset = node_offset + sub_chunk['start_char']
                    articles_in_chunk = _extract_structure_gemini(
                        sub_chunk['text'], chunk_offset, model, prompt_detailer, 
                        debug_info, f"step3_detailer_parent_{i+1}_subchunk_{j+1}"
                    )
                    all_articles_raw.extend(articles_in_chunk)
                node['children'] = postprocess_nodes(all_articles_raw, node_text, node_offset)

    process_recursively(final_tree)
    step3_end = time.perf_counter()
    timings["step3_detailer_duration"] = step3_end - step3_start
    
    debug_info.append({"performance_timings": timings})
    return {"tree": final_tree}