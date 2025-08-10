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
# [ CONFIGURATION ] - v10.1 Final Corrected Version
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

PROMPT_HIERARCHICAL_SUMMARIZER = """You are a Thai legal assistant. Your task is to create a concise, one-sentence summary for a specific section of a legal document.

You will be given the parent section's summary for context, and the full text of the current section.

-   **Parent Context**: This provides the broader legal context.
-   **Current Section Text**: This is the text you need to summarize.

Based on BOTH pieces of information, create a summary that explains the main purpose of the **Current Section Text** within the **Parent Context**.

[Parent Context Summary]
{parent_summary}

[Current Section Text to Summarize]
{text_chunk}

Return ONLY the one-sentence summary of the Current Section Text.
"""

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
        r"^(ภาค|ลักษณะ|หมวด|ส่วน|มาตรา|บทเฉพาะกาล|บทกำหนดโทษ|อัตราค่าธรรมเนียม|หมายเหตุ)\s*.*$", 
        re.MULTILINE
    )
    for match in pattern.finditer(text_chunk):
        first_word = match.group(1)
        title = match.group(0).strip()
        node_type = TYPE_MAPPING.get(first_word)
        if not node_type:
            if "บท" in first_word or "อัตรา" in first_word or "หมายเหตุ" in first_word:
                node_type = "subheading"
            else:
                node_type = "unknown"
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
        next_sibling_start = nodes[i+1]['global_start'] if i + 1 < len(nodes) else parent_global_end
        node['global_end'] = next_sibling_start
        node['text'] = full_document_text[node['global_start']:node['global_end']]
        if node.get('children'):
            _recursive_postprocess(node['children'], full_document_text, node['global_end'])

def _generate_hierarchical_summaries(nodes, client, llm_log, parent_summary="This document is a Thai legal code."):
    for node in nodes:
        if node.get('children'):
            prompt = PROMPT_HIERARCHICAL_SUMMARIZER.format(parent_summary=parent_summary, text_chunk=node['text'])
            try:
                response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
                summary = response.choices[0].message.content
                node['summary'] = summary
                usage = response.usage
                llm_log["count"] += 1
                call_log = {"call_number": llm_log["count"], "purpose": f"Summarize Node: {node['title'][:50]}...", "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
                llm_log["details"].append(call_log)
                _generate_hierarchical_summaries(node['children'], client, llm_log, parent_summary=summary)
            except Exception as e:
                node['summary'] = "Summary generation failed."
                llm_log["details"].append({"error": f"Failed to summarize {node['title']}: {e}"})

def _flatten_tree_to_chunks(nodes, parent_breadcrumbs=None, parent_summaries=None):
    if parent_breadcrumbs is None: parent_breadcrumbs = []
    if parent_summaries is None: parent_summaries = []
    final_chunks = []
    for node in nodes:
        current_breadcrumbs = parent_breadcrumbs + [node.get('title', '')]
        current_summaries = parent_summaries + [node.get('summary', '')]
        if not node.get('children'):
            node['context_path'] = " > ".join(filter(None, parent_breadcrumbs))
            node['context_summary'] = " ".join(filter(None, current_summaries))
            node.pop('children', None)
            node.pop('summary', None) # Clean up summary from leaf node
            final_chunks.append(node)
        else:
            final_chunks.extend(_flatten_tree_to_chunks(node['children'], current_breadcrumbs, current_summaries))
    return final_chunks

def _run_deep_hierarchical_pipeline(document_text, client, debug_info, llm_log):
    all_nodes = _parse_candidate_nodes(document_text)
    if not all_nodes: return {"error": "Deep Parsing Failed: No structural nodes found."}
    debug_info.append({"pipeline_a_parsed_nodes": all_nodes})

    first_book_index = next((i for i, node in enumerate(all_nodes) if node['type'] == 'book'), -1)
    
    preamble_nodes = all_nodes[:first_book_index] if first_book_index != -1 else []
    main_content_nodes = all_nodes[first_book_index:] if first_book_index != -1 else all_nodes

    structured_main_tree = _build_tree_from_flat_list(main_content_nodes)
    
    final_tree = []
    if preamble_nodes:
        preamble_container = {"type": "preamble", "title": "Preamble", "global_start": 0, "children": preamble_nodes}
        final_tree.append(preamble_container)
    final_tree.extend(structured_main_tree)
    
    _recursive_postprocess(final_tree, document_text, len(document_text))
    _generate_hierarchical_summaries(final_tree, client, llm_log)
    debug_info.append({"pipeline_a_full_hierarchical_tree_with_summaries": final_tree})

    rag_chunks = _flatten_tree_to_chunks(final_tree)
    return {"chunks": rag_chunks}

def _run_summary_chunking_pipeline(document_text, client, debug_info, llm_log):
    summary = "Summary generation failed."
    try:
        summary_prompt = PROMPT_SUMMARIZER.format(text_chunk=document_text)
        response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": summary_prompt}])
        summary = response.choices[0].message.content
        usage = response.usage
        llm_log["count"] += 1
        call_log = {"call_number": llm_log["count"], "purpose": "Full Document Summary", "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
        llm_log["details"].append(call_log)
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
                current_pos += len(chunk_text) + 2
                continue
            final_chunks.append({"type": "paragraph", "title": f"Paragraph {i+1}", "global_start": current_pos, "global_end": current_pos + len(chunk_text), "text": f"Document Summary: {summary}\n\n---\n\n{chunk_text}"})
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
    
    # [!!! FIX !!!] - Initialize the llm_log dict and append it to the list.
    # Pass the dict itself to sub-functions, but append other logs to the main list.
    llm_log = {"llm_calls": {"count": 0, "details": []}}
    debug_info.append(llm_log)

    status_container.write("1/3: **Profiler** - Analyzing document structure...")
    start_time = time.perf_counter()
    
    has_complex_structure = bool(re.search(r"^(ภาค|ลักษณะ)\s", document_text, re.MULTILINE))
    debug_info.append({"profiling_result": {"has_complex_structure": has_complex_structure}})
    
    final_result = {}
    if has_complex_structure:
        status_container.write("-> Complex document. Running Hierarchical Summarization Pipeline.")
        debug_info.append({"selected_pipeline": "A: Deep Hierarchical"})
        final_result = _run_deep_hierarchical_pipeline(document_text, client, debug_info, llm_log["llm_calls"])
    else:
        status_container.write("-> Simple document. Running Summary-Enriched Chunking Pipeline.")
        debug_info.append({"selected_pipeline": "B: Summary-Enriched Chunking"})
        final_result = _run_summary_chunking_pipeline(document_text, client, debug_info, llm_log["llm_calls"])

    end_time = time.perf_counter()
    timings["total_pipeline_duration"] = end_time - start_time
    debug_info.append({"performance_timings": timings})
    
    if "chunks" in final_result:
        return {"tree": final_result["chunks"]}
    return final_result