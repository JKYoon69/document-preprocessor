# document_processor_openai.py
# This file contains the logic for processing documents using the Adaptive Parsing Strategy.

import os
from openai import OpenAI, APIError
import json
import traceback
import time
import re

# ==============================================================================
# [ CONFIGURATION ] - v7.0 Adaptive Strategy
# ==============================================================================
MODEL_NAME = "gpt-4.1-2025-04-14"

# Define the hierarchy level for deterministic tree building
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

# New prompt for summarizing simple documents (Pipeline B)
PROMPT_SUMMARIZER = """You are a legal assistant. Read the entire provided Thai legal document and generate a concise, one-paragraph summary. The summary should explain the main purpose and key subjects of the law or decree. Start the summary with "This document is about...".

[DOCUMENT TEXT]
{text_chunk}

Return ONLY the summary text.
"""

# ==============================================================================
# [ END OF CONFIGURATION ]
# ==============================================================================

def _parse_candidate_nodes(text_chunk):
    """
    Uses an enhanced regex to find all potential headers, including subheadings.
    """
    candidate_nodes = []
    # Regex updated to capture specific subheadings as well as standard structures
    pattern = re.compile(
        r"^(ภาค|ลักษณะ|หมวด|ส่วน|มาตรา|บทเฉพาะกาล|บทกำหนดโทษ|อัตราค่าธรรมเนียม)\s*.*$", 
        re.MULTILINE
    )
    
    for match in pattern.finditer(text_chunk):
        first_word = match.group(1)
        title = match.group(0).strip()
        
        node_type = TYPE_MAPPING.get(first_word)
        if not node_type:
            # Handle special cases like 'บทกำหนดโทษ' (Penalties)
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

# [!!! NEW - Deterministic Tree Building using a Stack Algorithm !!!]
def _build_tree_from_flat_list(nodes):
    """
    Builds a nested tree from a flat list of nodes using a stack.
    This is deterministic and replaces the unreliable LLM structuring step.
    """
    if not nodes:
        return []

    root_nodes = []
    stack = []  # Stack will hold parent nodes, e.g., [book, part, chapter]

    for node in nodes:
        node['children'] = []
        node_level = HIERARCHY_LEVELS.get(node['type'], 99)

        # Pop from stack until the current node can be a child of the node on top of the stack
        while stack and HIERARCHY_LEVELS.get(stack[-1]['type'], 99) >= node_level:
            stack.pop()

        if not stack:
            # If stack is empty, this is a top-level node (e.g., a 'book')
            root_nodes.append(node)
        else:
            # The node on top of the stack is the parent
            stack[-1]['children'].append(node)
        
        # Push the current node onto the stack to become a potential parent for subsequent nodes
        stack.append(node)

    return root_nodes

def _recursive_postprocess(nodes, full_document_text, parent_global_end):
    """
    Traverses the tree to calculate end indices and extract text from the FULL document.
    """
    for i, node in enumerate(nodes):
        # The end of the current node is the start of the next sibling, or the parent's end
        next_sibling_start = nodes[i+1]['global_start'] if i + 1 < len(nodes) else parent_global_end
        node['global_end'] = next_sibling_start
        node['text'] = full_document_text[node['global_start']:node['global_end']]

        if node.get('children'):
            # The boundary for children is the current node's end
            _recursive_postprocess(node['children'], full_document_text, node['global_end'])
    return nodes

def _run_deep_hierarchical_pipeline(document_text, debug_info):
    """
    PIPELINE A: For complex documents. Uses the parser and deterministic tree building.
    """
    # 1. Parse all nodes deterministically
    all_nodes = _parse_candidate_nodes(document_text)
    if not all_nodes:
        return {"error": "Deep Parsing Failed: No structural nodes found."}
    debug_info.append({"pipeline_a_parsed_nodes": all_nodes})

    # 2. Build the tree using the stack algorithm (no LLM needed here)
    tree = _build_tree_from_flat_list(all_nodes)
    
    # 3. Finalize tree by adding text and end indices
    final_tree = []
    if tree and tree[0].get('global_start', 0) > 0:
        preamble = {"type": "preamble", "title": "Preamble", "global_start": 0, "children": []}
        final_tree.append(preamble)
    final_tree.extend(tree)
    
    processed_tree = _recursive_postprocess(final_tree, document_text, len(document_text))
    return {"tree": processed_tree}


def _run_summary_chunking_pipeline(document_text, client, debug_info):
    """
    PIPELINE B: For simple documents. Summarizes, then creates enriched chunks.
    """
    # 1. Get a summary of the entire document
    summary_prompt = PROMPT_SUMMARIZER.format(text_chunk=document_text)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, # Use the powerful model for a good summary
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = response.choices[0].message.content
        debug_info.append({"pipeline_b_summary": summary})
    except Exception as e:
        summary = "Summary generation failed."
        debug_info.append({"pipeline_b_summary_error": str(e)})

    # 2. Parse only for 'มาตรา' (articles) to create chunks
    nodes = _parse_candidate_nodes(document_text)
    article_nodes = [n for n in nodes if n['type'] == 'article']
    
    if not article_nodes:
        return {"error": "Summary Chunking Failed: No 'มาตรา' (article) nodes found to create chunks."}

    # 3. Create enriched chunks
    enriched_chunks = []
    doc_len = len(document_text)
    for i, node in enumerate(article_nodes):
        next_node_start = article_nodes[i+1]['global_start'] if i + 1 < len(article_nodes) else doc_len
        node['global_end'] = next_node_start
        
        # Prepend the summary to the actual text of the article
        article_text = document_text[node['global_start']:node['global_end']]
        node['text'] = f"Document Summary: {summary}\n\n---\n\n{article_text}"
        node['children'] = []
        enriched_chunks.append(node)
        
    return {"tree": enriched_chunks}


def run_openai_pipeline(document_text, api_key, status_container, debug_info, **kwargs):
    """
    Main entry point that decides which pipeline to run.
    """
    client = OpenAI(api_key=api_key)
    timings = {}
    
    # --- Step 1: Document Profiling ---
    status_container.write("1/3: **Profiler** - Analyzing document structure and length...")
    start_time = time.perf_counter()
    
    # Check for complex structure keywords
    has_complex_structure = bool(re.search(r"^(ภาค|ลักษณะ)\s", document_text, re.MULTILINE))
    is_long_document = len(document_text) > 50000 # Define a threshold for long docs

    profiling_result = {
        "has_complex_structure": has_complex_structure,
        "is_long_document": is_long_document,
        "char_count": len(document_text)
    }
    debug_info.append({"profiling_result": profiling_result})
    
    final_result = {}
    
    # --- Step 2 & 3: Conditional Pipeline Execution ---
    if has_complex_structure or is_long_document:
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