# document_processor_openai.py
# This file contains the logic for processing documents using the Adaptive Parsing Strategy.

import os
from openai import OpenAI, APIError
import json
import traceback
import time
import re

# ==============================================================================
# [ CONFIGURATION ] - v7.1 Finalized
# ==============================================================================
MODEL_NAME = "gpt-4.1-2025-04-14"

HIERARCHY_LEVELS = {
    "book": 1,      # ภาค
    "part": 2,      # ลักษณะ
    "chapter": 3,   # หมวด
    "subheading": 4,# บทเฉพาะกาล 등
    "section": 5,   # ส่วน
    "article": 6    # มาตรา
}

TYPE_MAPPING = {
    "ภาค": "book", "ลักษณะ": "part", "หมวด": "chapter",
    "ส่วน": "section", "มาตรา": "article"
}

PROMPT_SUMMARIZER = """You are a legal assistant. Read the entire provided Thai legal document and generate a concise, one-paragraph summary. The summary should explain the main purpose and key subjects of the law or decree. Start the summary with "This document is about...".

[DOCUMENT TEXT]
{text_chunk}

Return ONLY the summary text.
"""

# ==============================================================================
# [ END OF CONFIGURATION ]
# ==============================================================================

def _parse_candidate_nodes(text_chunk):
    candidate_nodes = []
    pattern = re.compile(
        r"^(ภาค|ลักษณะ|หมวด|ส่วน|มาตรา|บทเฉพาะกาล|บทกำหนดโทษ|อัตราค่าธรรมเนียม)\s*.*$", 
        re.MULTILINE
    )
    
    for match in pattern.finditer(text_chunk):
        first_word = match.group(1)
        title = match.group(0).strip()
        
        node_type = TYPE_MAPPING.get(first_word)
        if not node_type:
            if "บท" in first_word or "อัตรา" in first_word:
                node_type = "subheading"
            else:
                node_type = "unknown"

        candidate_nodes.append({
            "type": node_type,
            "title": title,
            "global_start": match.start()
        })
    return candidate_nodes

def _build_tree_from_flat_list(nodes):
    if not nodes:
        return []
    root_nodes = []
    stack = []
    for node in nodes:
        node['children'] = []
        node_level = HIERARCHY_LEVELS.get(node['type'], 99)
        while stack and HIERARCHY_LEVELS.get(stack[-1]['type'], 99) >= node_level:
            stack.pop()
        if not stack:
            root_nodes.append(node)
        else:
            stack[-1]['children'].append(node)
        stack.append(node)
    return root_nodes

# [!!! FINAL CORRECTED VERSION of _recursive_postprocess !!!]
def _recursive_postprocess(nodes, full_document_text, parent_global_end):
    """
    Traverses the tree to calculate end indices and extract text from the FULL document.
    """
    for i, node in enumerate(nodes):
        # The end of the current node is the start of the next sibling, or the parent's end boundary
        next_sibling_start = nodes[i+1]['global_start'] if i + 1 < len(nodes) else parent_global_end
        node['global_end'] = next_sibling_start
        
        # FIX: Always slice from the original full_document_text using absolute global indices.
        # This prevents empty text fields for nested children.
        node['text'] = full_document_text[node['global_start']:node['global_end']]

        if node.get('children'):
            # FIX: Pass the current node's end as the explicit boundary for its children.
            # This prevents the last child from over-extending its text content.
            _recursive_postprocess(node['children'], full_document_text, node['global_end'])
    return nodes

def _run_deep_hierarchical_pipeline(document_text, debug_info):
    # 1. Parse all nodes deterministically
    all_nodes = _parse_candidate_nodes(document_text)
    if not all_nodes:
        return {"error": "Deep Parsing Failed: No structural nodes found."}
    debug_info.append({"pipeline_a_parsed_nodes": all_nodes})

    # 2. Build the tree using the stack algorithm
    tree = _build_tree_from_flat_list(all_nodes)
    
    # 3. Finalize tree by adding a preamble and post-processing
    final_tree = []
    if tree and tree[0].get('global_start', 0) > 0:
        preamble = {"type": "preamble", "title": "Preamble", "global_start": 0, "children": []}
        final_tree.append(preamble)
    final_tree.extend(tree)
    
    processed_tree = _recursive_postprocess(final_tree, document_text, len(document_text))
    return {"tree": processed_tree}

def _run_summary_chunking_pipeline(document_text, client, debug_info):
    summary = "Summary generation failed." # Default value
    try:
        summary_prompt = PROMPT_SUMMARIZER.format(text_chunk=document_text)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = response.choices[0].message.content
        debug_info.append({"pipeline_b_summary": summary})
    except Exception as e:
        debug_info.append({"pipeline_b_summary_error": str(e)})

    nodes = _parse_candidate_nodes(document_text)
    article_nodes = [n for n in nodes if n['type'] == 'article']
    
    if not article_nodes:
         # If no articles, chunk by paragraphs as a fallback
        chunks = document_text.split('\n\n')
        final_chunks = []
        current_pos = 0
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip(): continue
            final_chunks.append({
                "type": "paragraph",
                "title": f"Paragraph {i+1}",
                "global_start": current_pos,
                "global_end": current_pos + len(chunk_text),
                "text": f"Document Summary: {summary}\n\n---\n\n{chunk_text}",
                "children": []
            })
            current_pos += len(chunk_text) + 2 # Add 2 for the newline characters
        return {"tree": final_chunks}

    enriched_chunks = []
    doc_len = len(document_text)
    for i, node in enumerate(article_nodes):
        next_node_start = article_nodes[i+1]['global_start'] if i + 1 < len(article_nodes) else doc_len
        node['global_end'] = next_node_start
        article_text = document_text[node['global_start']:node['global_end']]
        node['text'] = f"Document Summary: {summary}\n\n---\n\n{article_text}"
        node['children'] = []
        enriched_chunks.append(node)
        
    return {"tree": enriched_chunks}

def run_openai_pipeline(document_text, api_key, status_container, debug_info, **kwargs):
    client = OpenAI(api_key=api_key)
    timings = {}
    
    status_container.write("1/3: **Profiler** - Analyzing document structure...")
    start_time = time.perf_counter()
    
    has_complex_structure = bool(re.search(r"^(ภาค|ลักษณะ)\s", document_text, re.MULTILINE))
    
    profiling_result = { "has_complex_structure": has_complex_structure, "char_count": len(document_text) }
    debug_info.append({"profiling_result": profiling_result})
    
    final_result = {}
    
    if has_complex_structure:
        status_container.write("-> Complex document detected. Running Deep Hierarchical Pipeline.")
        debug_info.append({"selected_pipeline": "A: Deep Hierarchical"})
        final_result = _run_deep_hierarchical_pipeline(document_text, debug_info)
    else:
        status_container.write("-> Simple document detected. Running Summary-Enriched Chunking Pipeline.")
        debug_info.append({"selected_pipeline": "B: Summary-Enriched Chunking"})
        final_result = _run_summary_chunking_pipeline(document_text, client, debug_info)

    end_time = time.perf_counter()
    timings["total_pipeline_duration"] = end_time - start_time
    debug_info.append({"performance_timings": timings})
    
    return final_result