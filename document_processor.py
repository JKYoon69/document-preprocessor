# document_processor.py (의미 기반 청킹 테스트용)

def chunk_text_by_char(text, chunk_size_chars=100000, overlap_chars=20000):
    # (이전 버전의 글자 수 기준 함수는 그대로 둡니다.)
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text)}]
    chunks_indices = []
    start_char = 0
    while start_char < len(text):
        end_char = start_char + chunk_size_chars
        chunks_indices.append({"start_char": start_char, "end_char": end_char})
        next_start = start_char + chunk_size_chars - overlap_chars
        if next_start <= start_char: break
        start_char = next_start
    return chunks_indices

# ✅✅✅ 새로 추가된 의미 기반 청킹 함수 ✅✅✅
def chunk_text_semantic(text, chunk_size_chars=100000, overlap_chars=20000):
    """
    문장, 문단 등 의미있는 경계를 찾아 청크를 나누는 함수.
    """
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text)}]

    chunks_indices = []
    start_char = 0
    
    while start_char < len(text):
        # 1. 이상적인 청크의 끝 위치를 계산
        ideal_end = start_char + chunk_size_chars
        
        # 2. 실제 텍스트의 끝을 넘지 않도록 조정
        actual_end = min(ideal_end, len(text))
        
        # 3. 만약 마지막 청크라면, 그대로 추가하고 종료
        if ideal_end >= len(text):
            chunks_indices.append({"start_char": start_char, "end_char": actual_end})
            break

        # 4. 의미있는 분리 지점 찾기 (뒤에서부터 탐색)
        # 분리 기호 우선순위: 문단 > 문장 > 공백
        separators = ["\n\n", ". ", " ", ""]
        best_sep_pos = -1

        for sep in separators:
            # ideal_end 위치에서 뒤로 5000자 범위 내에서만 탐색
            search_start = max(start_char, ideal_end - 5000)
            best_sep_pos = text.rfind(sep, search_start, ideal_end)
            
            if best_sep_pos != -1:
                actual_end = best_sep_pos + len(sep)
                break
        
        # 적절한 분리 지점을 못 찾으면 그냥 ideal_end에서 자름
        if best_sep_pos == -1:
            actual_end = ideal_end

        chunks_indices.append({"start_char": start_char, "end_char": actual_end})
        
        # 5. 다음 청크의 시작 위치 계산
        start_char = actual_end - overlap_chars

    return chunks_indices