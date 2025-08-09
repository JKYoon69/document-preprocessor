# document_processor_openai.py
# This file contains the logic for processing documents using the Parser-First Hybrid approach.

import os
from openai import OpenAI, APIError
import json
import traceback
import time
import re # Regular expression library

# ==============================================================================
# [ CONFIGURATION ] - Parser-First Hybrid Version
# ==============================================================================
MODEL_NAME = "gpt-4.1-mini-2025-04-14"

# Type mapping remains useful for standardization
TYPE_MAPPING = {
    "ภาค": "book", "ลักษณะ": "part", "หมวด": "chapter",
    "ส่วน": "section", "มาตรา": "article"
}

# [!!! NEW - Single, powerful prompt for structuring pre-parsed nodes !!!]
PROMPT_STRUCTURER = """You are an expert in Thai legal document structure. I have already parsed a document and extracted a flat list of all potential headers with their exact titles and start indexes.

Your task is to organize this flat list into a correct, nested hierarchical JSON tree.

RULES:
1.  The hierarchy is STRICTLY: ภาค (book) > ลักษณะ (part) > หมวด (chapter) > ส่วน (section) > มาตรา (article).
2.  A node is a child of the most recent, higher-level node that appeared before it. For example, the first 'หมวด' after a 'ลักษณะ' is a child of that 'ลักษณะ'.
3.  The final output MUST be a single JSON object with a root key "tree" which contains a list of the top-level nodes (usually 'ภาค').
4.  Each node in the final tree must contain `type`, `title`, and `global_start`. It can also contain a `children` list.
5.  Do not invent new nodes. Only use the nodes provided in the input list.

Here is the flat list of candidate nodes:
{candidate_list_json}

Return ONLY the final, nested JSON object.
"""
# ==============================================================================
# [ END OF CONFIGURATION ]
# ==============================================================================

# [!!! NEW - Parser function using Regular Expressions !!!]
def _parse_candidate_nodes(text_chunk):
    """
    Uses regex to find all potential headers and their exact start index.
    This is deterministic and avoids LLM hallucination for positions.
    """
    candidate_nodes = []
    # This regex finds any line that STARTS with one of the keywords.
    pattern = re.compile(r"^(ภาค|ลักษณะ|หมวด|ส่วน|มาตรา)\s+.*$", re.MULTILINE)

    for match in pattern.finditer(text_chunk):
        node_type_thai = match.group(1)
        node_type_eng = TYPE_MAPPING.get(node_type_thai, node_type_thai)
        
        candidate_nodes.append({
            "type": node_type_eng,
            "title": match.group(0).strip(),
            "global_start": match.start() # Use global_start directly
        })
    return candidate_nodes

# [!!! NEW - Recursive function to post-process the LLM's tree output !!!]
def _recursive_postprocess(nodes, parent_text, parent_global_start, parent_global_end):
    """
    Traverses the tree returned by the LLM to calculate end indices and extract text.
    """
    for i, node in enumerate(nodes):
        # Determine the end of the current node
        next_node_start = nodes[i+1]['global_start'] if i + 1 < len(nodes) else parent_global_end
        node['global_end'] = next_node_start

        # Extract the node's own text from the parent's text
        local_start = node['global_start'] - parent_global_start
        local_end = node['global_end'] - parent_global_start
        node['text'] = parent_text[local_start:local_end]

        # If there are children, process them recursively
        if 'children' in node and node['children']:
            _recursive_postprocess(node['children'], node['text'], node['global_start'], node['global_end'])
        else:
            # ensure children key exists
            node['children'] = []
    return nodes

def extract_json_from_response(text):
    """Helper to extract JSON from LLM response."""
    if not text:
        return None
    # Standard markdown code block
    if '```json' in text:
        try:
            return json.loads(text.split('```json', 1)[1].split('```', 1)[0].strip())
        except (json.JSONDecodeError, IndexError):
            pass # Fallback to parsing the whole text
    # Direct JSON output
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# [!!! REWRITTEN - The main pipeline is now simpler and more robust !!!]
def run_openai_pipeline(document_text, api_key, status_container, debug_info, **kwargs):
    """
    Executes the new Parser-First Hybrid pipeline.
    kwargs are included to maintain compatibility with app.py's old call signature, but are not used.
    """
    client = OpenAI(api_key=api_key)
    timings = {}
    
    # --- Step 1: Parse all candidate nodes with Regex ---
    status_container.write("1/3: **Parser** - Extracting all candidate headers with exact locations...")
    step1_start = time.perf_counter()
    candidate_nodes = _parse_candidate_nodes(document_text)
    step1_end = time.perf_counter()
    timings["step1_parser_duration"] = step1_end - step1_start
    
    debug_info.append({"step1_parser_results": candidate_nodes})
    
    if not candidate_nodes:
        return {"error": "Step 1 (Parser) failed: Could not find any potential headers."}
    
    # --- Step 2: Use LLM to structure the candidates into a tree ---
    status_container.write("2/3: **LLM Structurer** - Organizing headers into a hierarchical tree...")
    step2_start = time.perf_counter()
    
    # Create the prompt for the LLM
    candidate_json_str = json.dumps(candidate_nodes, ensure_ascii=False, indent=2)
    prompt = PROMPT_STRUCTURER.format(candidate_list_json=candidate_json_str)
    
    final_tree = []
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
            final_tree = structured_result["tree"]
        else:
            raise ValueError("LLM did not return a valid JSON object with a 'tree' key.")
            
    except Exception as e:
        debug_info.append({"step2_llm_critical_error": traceback.format_exc()})
        return {"error": f"Step 2 (LLM) failed: {e}"}

    # --- Step 3: Post-process the tree to add text and end indices ---
    status_container.write("3/3: **Post-processor** - Calculating text blocks for each node...")
    step3_start = time.perf_counter()

    # Add a Preamble if the first node doesn't start at 0
    if final_tree and final_tree[0].get('global_start', 0) > 0:
        preamble_node = {
            "type": "preamble",
            "title": "Preamble",
            "global_start": 0,
            "children": []
        }
        final_tree.insert(0, preamble_node)
        
    final_tree = _recursive_postprocess(final_tree, document_text, 0, len(document_text))
    
    step3_end = time.perf_counter()
    timings["step3_postprocess_duration"] = step3_end - step3_start
    
    debug_info.append({"performance_timings": timings})
    return {"tree": final_tree}