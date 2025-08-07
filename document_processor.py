# document_processor.py (전역 요약 생성 기능 추가)

import google.generativeai as genai

# ✅ 검증 완료된 의미 기반 청킹 함수
def chunk_text_semantic(text, chunk_size_chars=100000, overlap_chars=20000):
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text), "text": text}]

    chunks = []
    start_char = 0
    while start_char < len(text):
        ideal_end = start_char + chunk_size_chars
        actual_end = min(ideal_end, len(text))
        
        if ideal_end >= len(text):
            chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end]})
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

        chunks.append({"start_char": start_char, "end_char": actual_end, "text": text[start_char:actual_end]})
        start_char = actual_end - overlap_chars
    return chunks

# ✅✅✅ 새로 추가된 전역 요약 생성 함수 ✅✅✅
def get_global_summary(text, api_key):
    """
    입력된 텍스트(주로 문서의 첫 부분)를 바탕으로 전역 요약을 생성합니다.
    """
    model_name = 'gemini-2.5-flash-lite'
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # 문서의 첫 4000자를 서문(preamble)으로 간주
    preamble = text[:4000]
    
    prompt = f"""Analyze the preamble of the Thai legal document provided below. 
Summarize the document's purpose, background, and core principles in 2-3 sentences in Korean. 
Return the summary in a JSON format with a single key "global_summary".

[Preamble text]
{preamble}
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"