# document_processor.py

import google.generativeai as genai
import json
import traceback
from collections import Counter
import time

# HELPER: Extracts JSON from the LLM's text response.
def extract_json_from_response(text):
    if '```json' in text:
        try:
            return json.loads(text.split('```json', 1)[1].split('```', 1)[0].strip())
        except (json.JSONDecodeError, IndexError):
            return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# HELPER: Splits text into semantic chunks.
def chunk_text_semantic(text, chunk_size_chars=100000, overlap_chars=20000):
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text), "text": text, "global_start": 0}]
    chunks, start_char = [], 0
    while start_char < len(text):
        ideal_end = start_char + chunk_size_chars
        actual_end = min(ideal_end, len(text))
        if ideal_end >= len(text):
            chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end], "global_start": start_char})
            break
        separators = ["\n\n", ". ", " ", ""]
        best_sep_pos = -1
        for sep in separators:
            search_start = max(start_char, ideal_end - 5000)
            best_sep_pos = text.rfind(sep, search_start, ideal_end)
            if best_sep_pos != -1:
                actual_end = best_sep_pos + len(sep)
                break
        if best_sep_pos == -1:
            actual_end = ideal_end
        chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end], "global_start": start_char})
        start_char = actual_end - overlap_chars
    return chunks

# FINAL STAGE HELPER: Builds the hierarchical tree from the flat list.
def build_tree_from_flat_list(flat_list):
    if not flat_list:
        return []

    # Use a stack to manage the hierarchy. The root is a dummy node.
    root = {"children": [], "level": -1}
    stack = [root]
    
    # Define the hierarchy levels for sorting nodes correctly.
    hierarchy_levels = {"book": 0, "part": 1, "chapter": 2, "section": 3, "article": 4, "preamble": 5}
    
    # Clean up and assign levels to each node.
    for node in flat_list:
        # Re-classify erroneous preambles at the end of the doc as articles
        if node.get("type") == "preamble" and node.get("global_start", 0) > 10000:
             node["type"] = "article"
        node["level"] = hierarchy_levels.get(node.get("type"), 5)

    for node in flat_list:
        # Find the correct parent on the stack for the current node.
        while stack[-1]["level"] >= node["level"]:
            stack.pop()
        
        parent = stack[-1]
        if "children" not in parent:
            parent["children"] = []
        parent["children"].append(node)
        
        # If the current node can have children, push it to the stack.
        if node["level"] < 4: # Books, parts, chapters, sections can be parents
            stack.append(node)

    return root["children"]

# FINAL STAGE HELPER: Recursively summarizes the nodes in the tree.
def summarize_nodes_recursively(node, model, global_summary):
    # Base Case: If the node is a leaf (an article), summarize its text.
    if "children" not in node or not node["children"]:
        prompt_template = """This legal document's purpose is {global_summary}.
The text below is one of its articles titled "{node_title}".
Based on the document's overall theme, concisely summarize the core content of this article in 1-2 sentences in Korean.

[Article Text]
{node_text}
"""
        prompt = prompt_template.format(
            global_summary=global_summary,
            node_title=node.get("title", ""),
            node_text=node.get("text", "")
        )
        response = model.generate_content(prompt)
        node["summary"] = response.text.strip()
        return

    # Recursive Step: Summarize all children first.
    for child in node["children"]:
        summarize_nodes_recursively(child, model, global_summary)
        
    # Now, summarize the parent node based on its children's summaries.
    child_summaries = "\n".join([f"- {child.get('title', '')}: {child.get('summary', '')}" for child in node["children"]])
    
    prompt_template = """This legal document's purpose is {global_summary}.
Below are summaries of all child nodes belonging to "{parent_title}".
Synthesize these summaries to create a comprehensive summary for the entire parent node in Korean.

[Summaries of child nodes]
{child_summaries}
"""
    prompt = prompt_template.format(
        global_summary=global_summary,
        parent_title=node.get("title", ""),
        child_summaries=child_summaries
    )
    response = model.generate_content(prompt)
    node["summary"] = response.text.strip()
    
# --- The Main Pipeline Function ---
def run_full_pipeline(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    all_headers = []

    # Step 1: Global Summary
    status_container.write("1/5: Generating global summary...")
    # ... (code for global summary generation)
    preamble = document_text[:4000]
    prompt_global = f"Summarize the preamble of this Thai legal document in Korean.\n\n[Preamble]\n{preamble}"
    response_summary = model.generate_content(prompt_global)
    global_summary = response_summary.text.strip()

    # Step 2: Chunking
    status_container.write("2/5: Chunking document...")
    chunks = chunk_text_semantic(document_text)
    
    # Step 3: Structure Extraction from Chunks
    prompt_structure = """You are a precise data extraction tool... [Text Chunk]\n{text_chunk}""" # (Using the last verified prompt)
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        status_container.write(f"3/5: Analyzing structure in chunk {chunk_num}/{len(chunks)}...")
        prompt = prompt_structure.format(text_chunk=chunk["text"])
        response = model.generate_content(prompt)
        headers_in_chunk = extract_json_from_response(response.text)
        if isinstance(headers_in_chunk, list):
            for header in headers_in_chunk:
                if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                    header["global_start"] = header["start_index"] + chunk["global_start"]
                    header["global_end"] = header["end_index"] + chunk["global_start"]
                    all_headers.append(header)

    # Step 4: Build Hierarchical Tree
    status_container.write("4/5: Building hierarchical tree...")
    unique_headers = list({h['global_start']: h for h in sorted(all_headers, key=lambda x: x['global_start'])}.values())
    hierarchical_tree = build_tree_from_flat_list(unique_headers)

    # Step 5: Recursive Summarization
    for root_node in hierarchical_tree:
        status_container.write(f"5/5: Summarizing nodes in '{root_node.get('title', '...')}'...")
        summarize_nodes_recursively(root_node, model, global_summary)

    # Final Cleanup
    final_data = {
        "global_summary": global_summary,
        "document_title": "Thai Narcotics Code Analysis",
        "chapters": hierarchical_tree
    }

    return final_data