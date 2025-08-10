# document_processor_openai.py
# This file contains the logic for processing documents using the Adaptive Parsing Strategy
# with Hierarchical Summarization and RAG-optimized chunking.

import os
from openai import OpenAI, APIError
import json
import traceback
import time
import re

# ==============================================================================
# [ CONFIGURATION ] - v17.0 Final Simplified Logic
# ==============================================================================
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
MAX_CHARS_FOR_SUMMARY = 64000

HIERARCHY_LEVELS = {
    "declaration": 0, "book": 1, "part": 2, "chapter": 3, "subheading": 4, "section": 5, "article": 6
}

TYPE_MAPPING = {
    "ภาค": "book", "ลักษณะ": "part", "หมวด": "chapter", "ส่วน": "section", "มาตรา": "article"
}

PROMPT_HIERARCHICAL_SUMMARIZER = """You are a Thai legal assistant. Your task is to create a concise, one-sentence summary for a specific section of a legal document based on its title and the titles of its direct children. This summary should explain the main purpose of the section.

-   **Current Section Title**: The title of the section you need to summarize (e.g., "ภาค 1 บททั่วไป").
-   **Sub-section Titles**: A list of titles under the current section (e.g., ["ลักษณะ 1 การจดทะเบียน", "ลักษณะ 2 บทกำหนดโทษ"]).

Based on BOTH the current section's title and its sub-section titles, what is the primary role of the current section? Return ONLY the one-sentence summary.

[Current Section Title]
{current_node_title}

[Sub-section Titles]
{child_titles}
"""

PROMPT_SUMMARIZER = """You are a legal assistant. Read the provided beginning of a Thai legal document and generate a concise, one-paragraph summary. The summary should explain the main purpose and key subjects of the law or decree. Start the summary with "This document is about...".

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
        r"^(ภาค|ลักษณะ|หมวด|ส่วน|มาตรา|บทเฉพาะกาล|บทกำหนดโทษ|อัตราค่าธรรมเนียม|หมายเหตุ)\s*.*$",
        re.MULTILINE
    )
    for match in pattern.finditer(text_chunk):
        first_word = match.group(1)
        title = match.group(0).strip()
        node_type = TYPE_MAPPING.get(first_word)
        if not node_type:
            node_type = "subheading"
        candidate_nodes.append({"type": node_type, "title": title, "global_start": match.start()})
    return candidate_nodes

def _build_tree_from_flat_list(nodes):
    if not nodes: return []
    root_nodes = []
    stack = []
    for node in nodes:
        node['children'] = []
        node_level = HIERARCHY_LEVELS.get(node['type'], 999)
        while stack and HIERARCHY_LEVELS.get(stack[-1]['type'], 999) >= node_level:
            stack.pop()
        if not stack:
            root_nodes.append(node)
        else:
            stack[-1]['children'].append(node)
        stack.append(node)
    return root_nodes

def _recursive_postprocess(nodes, full_document_text, parent_global_end):
    for i, node in enumerate(nodes):
        next_sibling_start = nodes[i+1].get('global_start', parent_global_end) if i + 1 < len(nodes) else parent_global_end
        start_pos = node['global_start']
        end_pos = node.get('global_end', next_sibling_start)
        node['global_end'] = end_pos
        node['text'] = full_document_text[start_pos:end_pos]
        if node.get('children'):
            _recursive_postprocess(node['children'], full_document_text, node['global_end'])

def _get_all_parent_nodes(nodes):
    parent_nodes = []
    def traverse(sub_nodes):
        for node in sub_nodes:
            if node.get('children'):
                parent_nodes.append(node)
                traverse(node['children'])
    traverse(nodes)
    return parent_nodes

def _flatten_tree_to_chunks(nodes, parent_summaries=None):
    if parent_summaries is None: parent_summaries = []
    final_chunks = []
    for node in nodes:
        current_summaries = parent_summaries
        if node.get('summary'):
            current_summaries = parent_summaries + [node.get('summary')]
        if not node.get('children'):
            node['context_summary'] = " > ".join(filter(None, current_summaries))
            node.pop('children', None)
            node.pop('summary', None)
            final_chunks.append(node)
        else:
            final_chunks.extend(_flatten_tree_to_chunks(node['children'], current_summaries))
    return final_chunks

def _run_deep_hierarchical_pipeline(document_text, client, debug_info, llm_log):
    # [!!! NEW SIMPLIFIED LOGIC !!!]
    
    # 1. Parse all structural nodes first.
    all_nodes = _parse_candidate_nodes(document_text)
    if not all_nodes:
        return {"chunks": [{"type": "full_document", "title": "Full Document", "global_start": 0, "global_end": len(document_text), "text": document_text}]}

    # 2. Check if there is text BEFORE the first structural node.
    if all_nodes[0]['global_start'] > 0:
        declaration_node = {
            'type': 'declaration',
            'title': 'Document Declaration',
            'global_start': 0,
            'children': [] 
        }
        # Insert this new node at the very beginning of the list.
        all_nodes.insert(0, declaration_node)

    # 3. Build ONE tree from the ENTIRE list of nodes. No more separation.
    final_tree = _build_tree_from_flat_list(all_nodes)
    
    # 4. Post-process the entire tree to get text content for each node.
    _recursive_postprocess(final_tree, document_text, len(document_text))

    # 5. Summarize parent nodes (same as before).
    nodes_to_summarize = _get_all_parent_nodes(final_tree)
    for node in nodes_to_summarize:
        if not node.get('children'): continue # Should not happen, but as a safeguard
        
        child_titles = [child.get('title', 'Untitled') for child in node.get('children')]
        child_titles_str = "\n".join(f"- {title}" for title in child_titles)
        prompt = PROMPT_HIERARCHICAL_SUMMARIZER.format(
            current_node_title=node.get('title', 'Untitled'),
            child_titles=child_titles_str
        )
        try:
            response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            summary = response.choices[0].message.content.strip()
            node['summary'] = summary
            llm_log["count"] += 1
            usage = response.usage
            call_log = {
                "call_number": llm_log["count"], "purpose": f"Summarize Parent Node: {node.get('title', 'Untitled')[:50]}...",
                "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens
            }
            llm_log["details"].append(call_log)
        except Exception as e:
            node['summary'] = f"Summary failed for {node.get('title', 'Untitled')}"
            error_msg = f"LLM call failed for node '{node.get('title', 'Untitled')}': {e}"
            llm_log["details"].append({"error": error_msg})

    debug_info.append({"pipeline_a_tree_with_summaries": final_tree})

    # 6. Flatten the final tree into chunks.
    rag_chunks = _flatten_tree_to_chunks(final_tree)
    return {"chunks": rag_chunks}


def _run_summary_chunking_pipeline(document_text, client, debug_info, llm_log):
    summary = "Summary generation failed."
    try:
        # ... implementation for simple documents ...
    except Exception as e:
        llm_log["details"].append({"error": f"Failed to summarize document: {e}"})

    # ... implementation ...
        
    return {"chunks": []} # Placeholder

def run_openai_pipeline(document_text, api_key, status_container, debug_info, **kwargs):
    client = OpenAI(api_key=api_key)
    timings = {}
    
    llm_log_container = {"llm_calls": {"count": 0, "details": []}}
    debug_info.append(llm_log_container)

    status_container.write("1/3: **Profiler** - Analyzing document structure...")
    start_time = time.perf_counter()
    
    has_complex_structure = bool(re.search(r"^(ภาค|ลักษณะ)\s", document_text, re.MULTILINE))
    debug_info.append({"profiling_result": {"has_complex_structure": has_complex_structure}})
    
    final_result = {}
    
    try:
        if has_complex_structure:
            status_container.write("-> Complex document. Running Hierarchical Summarization Pipeline.")
            debug_info.append({"selected_pipeline": "A: Deep Hierarchical"})
            final_result = _run_deep_hierarchical_pipeline(document_text, client, debug_info, llm_log_container["llm_calls"])
        else:
            status_container.write("-> Simple document. Running Summary-Enriched Chunking Pipeline.")
            debug_info.append({"selected_pipeline": "B: Summary-Enriched Chunking"})
            final_result = _run_summary_chunking_pipeline(document_text, client, debug_info, llm_log_container["llm_calls"])
    except Exception as e:
        return {"error": f"An error occurred in the selected pipeline: {e}", "traceback": traceback.format_exc()}

    end_time = time.perf_counter()
    timings["total_pipeline_duration"] = end_time - start_time
    debug_info.append({"performance_timings": timings})
    
    if final_result and final_result.get("chunks"):
        return {"tree": final_result["chunks"]}
    
    return final_result