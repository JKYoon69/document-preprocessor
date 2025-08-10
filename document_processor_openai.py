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
# [ CONFIGURATION ] - v15.0 Fix for Missing Preamble
# ==============================================================================
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
MAX_CHARS_FOR_SUMMARY = 64000

HIERARCHY_LEVELS = {
    "book": 1, "part": 2, "chapter": 3, "subheading": 4, "section": 5, "article": 6
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
        node_level = HIERARCHY_LEVELS.get(node['type'], 99)
        while stack and HIERARCHY_LEVELS.get(stack[-1]['type'], 99) >= node_level:
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
    all_nodes = _parse_candidate_nodes(document_text)
    if not all_nodes: return {"error": "Deep Parsing Failed: No structural nodes found."}
    
    first_book_index = next((i for i, node in enumerate(all_nodes) if node['type'] == 'book'), -1)
    
    preamble_nodes_flat = all_nodes[:first_book_index] if first_book_index != -1 else []
    main_content_nodes = all_nodes[first_book_index:] if first_book_index != -1 else all_nodes

    # [!!! FIXED !!!] Ensure preamble nodes are structurally consistent.
    # Add 'children' key to each preamble node so they are treated as proper leaf nodes.
    for node in preamble_nodes_flat:
        if 'children' not in node:
            node['children'] = []

    structured_main_tree = _build_tree_from_flat_list(main_content_nodes)
    
    final_tree = []
    if preamble_nodes_flat:
        preamble_container = {
            "type": "preamble_container", 
            "title": "Preamble Section", 
            "global_start": 0,
            "children": preamble_nodes_flat
        }
        final_tree.append(preamble_container)

    final_tree.extend(structured_main_tree)
    
    _recursive_postprocess(final_tree, document_text, len(document_text))

    nodes_to_summarize = _get_all_parent_nodes(final_tree)
    
    for node in nodes_to_summarize:
        child_titles = [child.get('title', 'Untitled') for child in node.get('children', [])]
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
                "call_number": llm_log["count"],
                "purpose": f"Summarize Parent Node: {node.get('title', 'Untitled')[:50]}...",
                "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
            llm_log["details"].append(call_log)
        except Exception as e:
            node['summary'] = f"Summary failed for {node.get('title', 'Untitled')}"
            error_msg = f"LLM call failed for node '{node.get('title', 'Untitled')}': {e}"
            llm_log["details"].append({"error": error_msg})

    debug_info.append({"pipeline_a_tree_with_summaries": final_tree})
    rag_chunks = _flatten_tree_to_chunks(final_tree)
    return {"chunks": rag_chunks}

def _run_summary_chunking_pipeline(document_text, client, debug_info, llm_log):
    summary = "Summary generation failed."
    try:
        truncated_text = document_text[:MAX_CHARS_FOR_SUMMARY]
        if len(document_text) > MAX_CHARS_FOR_SUMMARY:
            debug_info.append({"warning": f"Document text was truncated to {MAX_CHARS_FOR_SUMMARY} chars."})
        summary_prompt = PROMPT_SUMMARIZER.format(text_chunk=truncated_text)
        response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": summary_prompt}])
        summary = response.choices[0].message.content.strip()
        llm_log["count"] += 1
        # ... logging ...
    except Exception as e:
        llm_log["details"].append({"error": f"Failed to summarize document: {e}"})

    nodes = _parse_candidate_nodes(document_text)
    article_nodes = [n for n in nodes if n.get('type') == 'article']
    
    if not article_nodes:
        chunks = document_text.split('\n\n')
        final_chunks = []
        current_pos = 0
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                current_pos += len(chunk_text) + 2; continue
            final_chunks.append({
                "type": "paragraph", "title": f"Paragraph {i+1}", 
                "global_start": current_pos, "global_end": current_pos + len(chunk_text), 
                "text": f"Document Summary: {summary}\n\n---\n\n{chunk_text}"
            })
            current_pos += len(chunk_text) + 2
        return {"chunks": final_chunks}

    enriched_chunks = []
    doc_len = len(document_text)
    for i, node in enumerate(article_nodes):
        next_node_start = article_nodes[i+1]['global_start'] if i + 1 < len(article_nodes) else doc_len
        node['global_end'] = next_node_start
        article_text = document_text[node['global_start']:node['global_end']]
        node['text'] = f"Document Summary: {summary}\n\n---\n\n{article_text}"
        node.pop('children', None)
        enriched_chunks.append(node)
        
    return {"chunks": enriched_chunks}

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