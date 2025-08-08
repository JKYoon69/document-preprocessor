# document_processor.py

import google.generativeai as genai
import json
import traceback
from collections import Counter
import time

# 헬퍼 함수: LLM 응답에서 JSON 추출
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

# 헬퍼 함수: 의미 기반 청킹
def chunk_text_semantic(text, chunk_size_chars=100000, overlap_chars=20000):
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text), "text": text, "global_start": 0}]

    chunks = []
    start_char = 0
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

# 3단계: 계층적 트리 구축 로직
def build_tree(flat_list):
    if not flat_list:
        return []

    hierarchy = {"book": 0, "part": 1, "chapter": 2, "section": 3, "article": 4, "preamble": 5}
    
    # 정형화된 제목이 아닌 설명 구절 필터링 및 레벨 할당
    cleaned_list = []
    for node in flat_list:
        title = node.get("title", "")
        node_type = node.get("type")
        if (title.startswith(('ภาค', 'ลักษณะ', 'หมวด', 'ส่วน', 'มาตรา')) or node_type == "preamble"):
            if node_type == "preamble":
                node["title"] = "Preamble" # Preamble 제목 정리
            node["level"] = hierarchy.get(node_type, 5)
            cleaned_list.append(node)
    
    root = {"children": [], "level": -1}
    stack = [root]

    for node in cleaned_list:
        while stack[-1]["level"] >= node["level"]:
            stack.pop()
        
        parent = stack[-1]
        if "children" not in parent:
            parent["children"] = []
        parent["children"].append(node)
        
        if node["level"] < 4: # book, part, chapter, section은 부모가 될 수 있음
            stack.append(node)
            
    # 최종 JSON 형식에 맞게 구조 정리
    def finalize_structure(node):
        if "children" in node:
            node["sections"] = [child for child in node["children"] if child.get("level") < 4]
            node["articles"] = [child for child in node["children"] if child.get("level") >= 4]
            
            for section in node["sections"]:
                finalize_structure(section)
            
            if not node["sections"]: del node["sections"]
            if not node["articles"]: del node["articles"]
            del node["children"]
        
        # 임시 키 삭제
        if "level" in node: del node["level"]
        if "global_start" in node: del node["global_start"]
        if "global_end" in node: del node["global_end"]
        
    for child in root["children"]:
        finalize_structure(child)

    return root["children"]

# 4단계: 재귀적 요약 로직
def summarize_nodes_recursively(node, model, global_summary, status_container):
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }

    # 하위 노드부터 재귀적으로 요약
    if "sections" in node:
        for section in node["sections"]:
            summarize_nodes_recursively(section, model, global_summary, status_container)
    
    if "articles" in node:
        for article in node["articles"]:
            status_container.write(f"...요약 중: {article.get('title', '알 수 없는 Article')}")
            prompt = f"Based on the global summary '{global_summary}', please summarize the following article '{article.get('title')}' in one sentence in Korean.\n\nText:\n{article.get('text')}"
            try:
                response = model.generate_content(prompt, safety_settings=safety_settings)
                article["summary"] = response.text.strip()
            except Exception:
                article["summary"] = "요약 생성 실패"

    # 하위 노드 요약이 끝나면 현재 노드 요약
    if "sections" in node or "articles" in node:
        child_summaries = ""
        if "sections" in node:
            child_summaries += "\n".join([f"- {s.get('title')}: {s.get('summary', '')}" for s in node["sections"]])
        if "articles" in node:
            child_summaries += "\n".join([f"- {a.get('title')}: {a.get('summary', '')}" for a in node["articles"]])
        
        # 프롬프트가 너무 길어지는 것을 방지
        if len(child_summaries) > 30000:
            child_summaries = child_summaries[:30000] + "\n... (요약 내용이 너무 길어 생략)"

        status_container.write(f"...상위 노드 요약 중: {node.get('title', '알 수 없는 Node')}")
        prompt = f"Based on the global summary '{global_summary}', please synthesize the following summaries of child nodes into a comprehensive summary for the parent node '{node.get('title')}' in Korean.\n\nChild Summaries:\n{child_summaries}"
        try:
            response = model.generate_content(prompt, safety_settings=safety_settings)
            node["summary"] = response.text.strip()
        except Exception:
            node["summary"] = "요약 생성 실패"


# ✅✅✅ app.py에서 호출하는 함수 이름과 정확히 일치합니다 ✅✅✅
def run_final_pipeline(document_text, api_key, status_container, file_name):
    model_name = 'gemini-2.5-flash'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE", "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    # 1. 전역 요약
    status_container.write("1/5: 전역 요약 생성...")
    preamble_text = document_text[:4000]
    prompt_global = f"Summarize the preamble of this Thai legal document in Korean.\n\n[Preamble]\n{preamble_text}"
    response_summary = model.generate_content(prompt_global, safety_settings=safety_settings)
    global_summary = response_summary.text.strip()

    # 2. 청킹 및 구조 분석
    status_container.write("2/5: 청킹 및 구조 분석...")
    chunks = chunk_text_semantic(document_text)
    all_headers = []
    prompt_structure = """You are a highly accurate legal document parsing tool. Your task is to analyze the following chunk of a Thai legal document and identify all structural elements.

Follow these rules with extreme precision:
1.  Identify the introductory text before the first formal article as `preamble`.
2.  Identify all headers such as 'ภาค', 'ลักษณะ', 'หมวด', 'ส่วน', and 'มาตรา'.
3.  For each element, create a JSON object.
4.  The `title` field MUST contain ONLY the short header text (e.g., "มาตรา ๑").
5.  The `end_index` of an element MUST extend to the character right before the `start_index` of the NEXT element. If it is the last element in the chunk, its `end_index` is the end of the chunk.
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
        status_container.write(f"청크 {i+1}/{len(chunks)} 분석 중...")
        prompt = prompt_structure.format(text_chunk=chunk["text"])
        response = model.generate_content(prompt, safety_settings=safety_settings)
        headers_in_chunk = extract_json_from_response(response.text)
        if isinstance(headers_in_chunk, list):
            for header in headers_in_chunk:
                if isinstance(header, dict) and all(k in header for k in ['type', 'title', 'start_index', 'end_index']):
                    header["global_start"] = header["start_index"] + chunk["global_start"]
                    header["global_end"] = header["end_index"] + chunk["global_start"]
                    all_headers.append(header)

    # 3. 후처리
    status_container.write("3/5: 데이터 후처리 및 정제...")
    if not all_headers:
        raise ValueError("LLM이 문서에서 구조를 추출하지 못했습니다.")
    
    unique_headers = sorted(list({h['global_start']: h for h in all_headers}.values()), key=lambda x: x['global_start'])
    
    if unique_headers and unique_headers[0]["global_start"] > 0:
        preamble_node = {"type": "preamble", "title": "Preamble", "global_start": 0, "global_end": unique_headers[0]["global_start"]}
        unique_headers.insert(0, preamble_node)
        
    for i in range(len(unique_headers) - 1):
        unique_headers[i]["global_end"] = unique_headers[i+1]["global_start"]
    if unique_headers:
        unique_headers[-1]["global_end"] = len(document_text)

    for header in unique_headers:
        header["text"] = document_text[header["global_start"]:header["global_end"]]

    # 4. 계층 트리 구축
    status_container.write("4/5: 계층 트리 구축...")
    hierarchical_tree = build_tree(unique_headers)

    # 5. 재귀적 요약
    status_container.write("5/5: 계층 구조 요약...")
    for root_node in hierarchical_tree:
        summarize_nodes_recursively(root_node, model, global_summary, status_container)

    return {
        "global_summary": global_summary,
        "document_title": file_name.split('.')[0],
        "chapters": hierarchical_tree
    }