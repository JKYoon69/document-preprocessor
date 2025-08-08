# document_processor.py

import google.generativeai as genai
import json
import traceback
from collections import Counter
import time

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

def build_tree(flat_list):
    if not flat_list: return []
    hierarchy = {"book": 0, "part": 1, "chapter": 2, "section": 3, "article": 4, "preamble": 5}
    for node in flat_list:
        node["level"] = hierarchy.get(node.get("type"), 5)
    root = {"children": [], "level": -1}
    stack = [root]
    for node in flat_list:
        while stack[-1]["level"] >= node["level"]:
            stack.pop()
        parent = stack[-1]
        if "children" not in parent: parent["children"] = []
        parent["children"].append(node)
        if node["level"] < 4: stack.append(node)
    def finalize_structure(node):
        if "children" in node:
            node["sections"] = [child for child in node["children"] if child.get("level") < 4]
            node["articles"] = [child for child in node["children"] if child.get("level") >= 4]
            for section in node["sections"]: finalize_structure(section)
            if not node["sections"]: del node["sections"]
            if not node["articles"]: del node["articles"]
            del node["children"]
        del node["level"], node["global_start"], node["global_end"]
    for child in root["children"]: finalize_structure(child)
    return root["children"]

def summarize_nodes_recursively(node, model, global_summary, status_container):
    if "sections" in node:
        for section in node["sections"]:
            summarize_nodes_recursively(section, model, global_summary, status_container)
    if "articles" in node:
        for article in node["articles"]:
            status_container.write(f"...Summarizing: {article.get('title', 'Unknown Article')}")
            prompt = f"Based on the global summary '{global_summary}', please summarize the following article '{article.get('title')}' in one sentence in Korean.\n\nText:\n{article.get('text')}"
            response = model.generate_content(prompt)
            article["summary"] = response.text.strip()
    if "sections" in node or "articles" in node:
        child_summaries = ""
        if "sections" in node: child_summaries += "\n".join([f"- {s.get('title')}: {s.get('summary', '')}" for s in node["sections"]])
        if "articles" in node: child_summaries += "\n".join([f"- {a.get('title')}: {a.get('summary', '')}" for a in node["articles"]])
        status_container.write(f"...Summarizing parent node: {node.get('title', 'Unknown Node')}")
        prompt = f"Based on the global summary '{global_summary}', please synthesize the following summaries of child nodes into a comprehensive summary for the parent node '{node.get('title')}' in Korean.\n\nChild Summaries:\n{child_summaries}"
        response = model.generate_content(prompt)
        node["summary"] = response.text.strip()

# ✅✅✅ 함수 이름을 'run_full_pipeline'으로 수정 ✅✅✅
def run_full_pipeline(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    status_container.write("1/5: Generating global summary...")
    preamble_text = document_text[:4000]
    prompt_global = f"Summarize the preamble of this Thai legal document in Korean.\n\n[Preamble]\n{preamble_text}"
    response_summary = model.generate_content(prompt_global)
    global_summary = response_summary.text.strip()

    status_container.write("2/5: Chunking and analyzing structure...")
    chunks = chunk_text_semantic(document_text)
    all_headers = []
    prompt_structure = """You are a highly accurate legal document parsing tool. Your task is to analyze the following chunk of a Thai legal document and identify all structural elements.

Follow these rules with extreme precision:
1.  Identify the introductory text before the first formal article as `preamble`.
2.  Identify all headers such as 'ภาค', 'ลักษณะ', 'หมวด', 'ส่วน', and 'มาตรา'.
3.  For each element, create a JSON object.
4.  The `title` field MUST contain ONLY the short header text (e.g., "มาตรา ๑").
5.  The `end_index` of an element MUST extend to the character right before the `start_index` of the NEXT element. If it is the last element in the chunk, its `end_index` is the end of the chunk. This is crucial to avoid missing any text.
6.  Map the Thai header to the `type` field using these exact rules:
    - Text before 'มาตรา ๑' -> 'preamble'
    - 'ภาค' -> 'book'
    - 'ลักษณะ' -> 'part'
    - 'หมวด' -> 'chapter'
    - 'ส่วน' -> 'section'
    - 'มาตรา' -> 'article'
7.  Return a single JSON array of these objects.

Example of expected output for a chunk:
[
  {{
    "type": "preamble",
    "title": "Preamble",
    "start_index": 0,
    "end_index": 533
  }},
  {{
    "type": "article",
    "title": "มาตรา ๑",
    "start_index": 534,
    "end_index": 611
  }},
  {{
    "type": "article",
    "title": "มาตรา ๒",
    "start_index": 612,
    "end_index": 688
  }}
]

[Text Chunk]
{text_chunk}"""
    for i, chunk in enumerate(chunks):
        prompt = prompt_structure.format(text_chunk=chunk["text"])
        response = model.generate_content(prompt)
        headers_in_chunk = extract_json_from_response(response.text)
        if isinstance(headers_in_chunk, list):
            for header in headers_in_chunk:
                if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                    header["global_start"] = header["start_index"] + chunk["global_start"]
                    header["global_end"] = header["end_index"] + chunk["global_start"]
                    all_headers.append(header)

    status_container.write("3/5: Post-processing data...")
    if not all_headers: raise ValueError("LLM did not extract any structure.")
    unique_headers = sorted(list({h['global_start']: h for h in all_headers}.values()), key=lambda x: x['global_start'])
    if unique_headers[0]["global_start"] > 0:
        preamble_node = {"type": "preamble", "title": "Preamble", "global_start": 0, "global_end": unique_headers[0]["global_start"]}
        unique_headers.insert(0, preamble_node)
    for i in range(len(unique_headers) - 1):
        unique_headers[i]["global_end"] = unique_headers[i+1]["global_start"]
    unique_headers[-1]["global_end"] = len(document_text)
    for header in unique_headers:
        header["text"] = document_text[header["global_start"]:header["global_end"]]

    status_container.write("4/5: Building hierarchical tree...")
    hierarchical_tree = build_tree(unique_headers)

    status_container.write("5/5: Summarizing nodes...")
    for root_node in hierarchical_tree:
        summarize_nodes_recursively(root_node, model, global_summary, status_container)

    return {
        "global_summary": global_summary,
        "document_title": "Thai Legal Document Analysis",
        "chapters": hierarchical_tree
    }