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

# The Main Pipeline Function
def run_full_pipeline(document_text, api_key, status_container):
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    debug_info = []
    all_headers = []
    chunk_stats = []

    # 1. Global Summary
    status_container.write("1/4: Generating global summary...")
    try:
        preamble = document_text[:4000]
        prompt_global = f"Summarize the preamble of this Thai legal document in Korean.\n\n[Preamble]\n{preamble}"
        
        start_time = time.time()
        response_summary = model.generate_content(prompt_global)
        end_time = time.time()
        
        global_summary = response_summary.text.strip()
        debug_info.append({"global_summary_response_time": f"{end_time - start_time:.2f} seconds"})
        
    except Exception as e:
        global_summary = f"Error during global summary generation: {e}"
        debug_info.append({"global_summary_error": traceback.format_exc()})

    # 2. Chunking
    status_container.write("2/4: Chunking document...")
    chunks = chunk_text_semantic(document_text)
    status_container.write(f"Chunking complete: {len(chunks)} chunks created.")
    
    # 3. Structure Analysis per Chunk
    prompt_structure = """You are a precise data extraction tool. Your task is to analyze the following chunk of a Thai legal document and identify all hierarchical headers.

Follow these rules STRICTLY:
1.  Identify headers such as 'ภาค', 'ลักษณะ', 'หมวด', 'ส่วน', and 'มาตรา'.
2.  For each header, create a JSON object.
3.  The `title` field MUST contain ONLY the header text itself (e.g., "มาตรา ๑"), NOT the full text of the article.
4.  Map the Thai header to the `type` field using these exact rules:
    - 'ภาค' -> 'book'
    - 'ลักษณะ' -> 'part'
    - 'หมวด' -> 'chapter'
    - 'ส่วน' -> 'section'
    - 'มาตรา' -> 'article'
5.  Provide the character `start_index` and `end_index` for the entire element (header + its content).
6.  Return a single JSON array of these objects. If no headers are found, return an empty array [].

Example:
[
  {{
    "type": "chapter",
    "title": "หมวด ๑ บทบัญญัติทั่วไป",
    "start_index": 120,
    "end_index": 950
  }},
  {{
    "type": "article",
    "title": "มาตรา ๑",
    "start_index": 250,
    "end_index": 400
  }}
]

[Text Chunk]
{text_chunk}"""
    
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        status_container.write(f"3/4: Analyzing structure in chunk {chunk_num}/{len(chunks)}...")
        try:
            prompt = prompt_structure.format(text_chunk=chunk["text"])
            
            start_time = time.time()
            response = model.generate_content(prompt)
            end_time = time.time()
            
            debug_info.append({
                f"chunk_{chunk_num}_response_time": f"{end_time - start_time:.2f} seconds",
                f"chunk_{chunk_num}_response": response.text
            })

            headers_in_chunk = extract_json_from_response(response.text)
            
            if isinstance(headers_in_chunk, list):
                counts = Counter(h.get('type', 'unknown') for h in headers_in_chunk)
                chunk_stats.append({"Chunk Number": chunk_num, "book": counts.get('book',0), "part": counts.get('part',0), "chapter": counts.get('chapter', 0), "section": counts.get('section', 0), "article": counts.get('article', 0)})
                
                for header in headers_in_chunk:
                    if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                        header["global_start"] = header["start_index"] + chunk["global_start"]
                        header["global_end"] = header["end_index"] + chunk["global_start"]
                        all_headers.append(header)
            else:
                debug_info.append({f"chunk_{chunk_num}_parsing_error": "Response was not a valid list of objects."})
        except Exception as e:
            status_container.error(f"Error processing chunk {chunk_num}: {e}")
            debug_info.append({f"chunk_{chunk_num}_critical_error": traceback.format_exc()})
            continue

    # 4. Result Aggregation and Deduplication
    status_container.write("4/4: Aggregating results and removing duplicates...")
    
    unique_headers, duplicate_counts, final_counts = [], {}, {}
    try:
        original_counts = Counter(h.get('type', 'unknown') for h in all_headers)
        unique_headers_map = {(h['global_start'], h['title']): h for h in all_headers}
        unique_headers = sorted(list(unique_headers_map.values()), key=lambda x: x['global_start'])
        final_counts = Counter(h.get('type', 'unknown') for h in unique_headers)
        duplicate_counts = {
            "book": original_counts.get('book', 0) - final_counts.get('book', 0),
            "part": original_counts.get('part', 0) - final_counts.get('part', 0),
            "chapter": original_counts.get('chapter', 0) - final_counts.get('chapter', 0),
            "section": original_counts.get('section', 0) - final_counts.get('section', 0),
            "article": original_counts.get('article', 0) - final_counts.get('article', 0)
        }
    except KeyError as e:
        debug_info.append({"deduplication_error": f"Key error during deduplication: {e}"})

    # Prepare final outputs
    final_result_data = {
        "global_summary": global_summary,
        "structure": unique_headers
    }
    stats_data = {
        "chunk_stats": chunk_stats,
        "duplicate_counts": duplicate_counts,
        "final_counts": final_counts
    }

    return final_result_data, stats_data, debug_info