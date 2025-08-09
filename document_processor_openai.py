# document_processor_openai.py
# This file contains the logic for processing documents using the Parser-First Hybrid approach.

import os
from openai import OpenAI, APIError
import json
import traceback
import time
import re # Regular expression library

# ==============================================================================
# [ CONFIGURATION ] - v6.2 Corrected Post-processing
# ==============================================================================
MODEL_NAME = "gpt-4.1-2025-04-14"

TYPE_MAPPING = {
    "ภาค": "book", "ลักษณะ": "part", "หมวด": "chapter",
    "ส่วน": "section", "มาตรา": "article"
}

PROMPT_STRUCTURER = """You are an expert in Thai legal document structure. I have already parsed a document and extracted a flat list of all potential headers with their exact titles and start indexes.

Your task is to organize this flat list into a correct, nested hierarchical JSON tree.

RULES:
1.  The hierarchy is STRICTLY: ภาค (book) > ลักษณะ (part) > หมวด (chapter) > ส่วน (section) > มาตรา (article).
2.  A node is a child of the most recent, higher-level node that appeared before it. For example, the first 'หมวด' after a 'ลักษณะ' is a child of that 'ลักษณะ'.
3.  The final output MUST be a single JSON object with a root key "tree" which contains a list of the top-level nodes (which will always be 'ภาค' based on my input).
4.  Each node in the final tree must contain `type`, `title`, and `global_start`. It can also contain a `children` list.
5.  Do not invent new nodes. Only use the nodes provided in the input list.

Here is the flat list of candidate nodes:
{candidate_list_json}

Return ONLY the final, nested JSON object.
"""
# ==============================================================================
# [ END OF CONFIGURATION ]
# ==============================================================================

def _parse_candidate_nodes(text_chunk):
    candidate_nodes = []
    pattern = re.compile(r"^(ภาค|ลักษณะ|หมวด|ส่วน|มาตรา)\s+.*$", re.MULTILINE)
    for match in pattern.finditer(text_chunk):
        node_type_thai = match.group(1)
        node_type_eng = TYPE_MAPPING.get(node_type_thai, node_type_thai)
        candidate_nodes.append({
            "type": node_type_eng,
            "title": match.group(0).strip(),
            "global_start": match.start()
        })
    return candidate_nodes

# [!!! MODIFIED - The text extraction logic is now corrected !!!]
def _recursive_postprocess(nodes, full_document_text, parent_global_end):
    """
    Traverses the tree to calculate end indices and extract text from the FULL document.
    """
    for i, node in enumerate(nodes):
        next_node_start = nodes[i+1]['global_start'] if i + 1 < len(nodes) else parent_global_end
        
        if next_node_start <= node['global_start']:
            next_node_start = parent_global_end
            
        node['global_end'] = next_node_start

        # [!!! FIX !!!] Always slice from the original full_document_text
        # This ensures correct text extraction regardless of depth.
        node['text'] = full_document_text[node['global_start']:node['global_end']]

        if 'children' in node and node['children']:
            # Pass the node's own end as the boundary for its children
            _recursive_postprocess(node['children'], full_document_text, node['global_end'])
        else:
            node['children'] = []
    return nodes

def extract_json_from_response(text):
    if not text: return None
    if '```json' in text:
        try:
            return json.loads(text.split('```json', 1)[1].split('```', 1)[0].strip())
        except (json.JSONDecodeError, IndexError):
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def run_openai_pipeline(document_text, api_key, status_container, debug_info, **kwargs):
    client = OpenAI(api_key=api_key)
    timings = {}
    
    status_container.write("1/3: **Parser** - Extracting all candidate headers...")
    step1_start = time.perf_counter()
    all_candidate_nodes = _parse_candidate_nodes(document_text)
    step1_end = time.perf_counter()
    timings["step1_parser_duration"] = step1_end - step1_start
    debug_info.append({"step1_parser_results": all_candidate_nodes})
    
    if not all_candidate_nodes:
        return {"error": "Step 1 (Parser) failed: Could not find any potential headers."}
    
    first_book_index = -1
    for i, node in enumerate(all_candidate_nodes):
        if node['type'] == 'book':
            first_book_index = i
            break
            
    preamble_nodes = []
    main_content_nodes = []
    if first_book_index != -1:
        preamble_nodes = all_candidate_nodes[:first_book_index]
        main_content_nodes = all_candidate_nodes[first_book_index:]
    else:
        main_content_nodes = all_candidate_nodes

    debug_info.append({
        "pre_filtering_split": {
            "preamble_node_count": len(preamble_nodes),
            "main_content_node_count": len(main_content_nodes)
        }
    })

    status_container.write("2/3: **LLM Structurer** - Organizing main content into a tree...")
    step2_start = time.perf_counter()
    structured_main_tree = []
    if main_content_nodes:
        candidate_json_str = json.dumps(main_content_nodes, ensure_ascii=False, indent=2)
        prompt = PROMPT_STRUCTURER.format(candidate_list_json=candidate_json_str)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
            
            step2_end = time.perf_counter()
            timings["step2_llm_duration"] = step2_end - step2_start
            debug_info.append({"step2_llm_response": response_text, "llm_duration_seconds": timings["step2_llm_duration"]})
            
            structured_result = extract_json_from_response(response_text)
            
            if structured_result and "tree" in structured_result and isinstance(structured_result["tree"], list):
                structured_main_tree = structured_result["tree"]
            else:
                raise ValueError("LLM did not return a valid JSON object with a 'tree' key.")
        except Exception as e:
            debug_info.append({"step2_llm_critical_error": traceback.format_exc()})
            return {"error": f"Step 2 (LLM) failed: {e}"}

    status_container.write("3/3: **Post-processor** - Finalizing the complete document tree...")
    step3_start = time.perf_counter()
    final_tree_before_processing = []
    
    if preamble_nodes:
        preamble_container = {
            "type": "preamble", "title": "Preamble", "global_start": 0, "children": preamble_nodes
        }
        final_tree_before_processing.append(preamble_container)

    final_tree_before_processing.extend(structured_main_tree)
    
    if not preamble_nodes and final_tree_before_processing and final_tree_before_processing[0].get('global_start', 0) > 0:
        preamble_container = { "type": "preamble", "title": "Preamble", "global_start": 0, "children": [] }
        final_tree_before_processing.insert(0, preamble_container)

    # Call the corrected post-processor
    final_tree = _recursive_postprocess(final_tree_before_processing, document_text, len(document_text))
    
    step3_end = time.perf_counter()
    timings["step3_postprocess_duration"] = step3_end - step3_start
    
    debug_info.append({"performance_timings": timings})
    return {"tree": final_tree}