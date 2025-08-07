# document_processor.py (청크 분할 테스트용)

def chunk_text(text, chunk_size=100000, overlap_ratio=0.2):
    """
    텍스트를 지정된 크기와 중첩 비율로 나누는, 새롭고 명확한 함수.
    """
    # 0. 중첩 크기 계산
    overlap = int(chunk_size * overlap_ratio)
    
    # 1. 텍스트가 청크 크기보다 작거나 같으면, 그대로 단일 청크로 반환
    if len(text) <= chunk_size:
        return [{"text": text, "global_start": 0, "size": len(text)}]

    # 2. 반복문을 위한 단계(step) 크기 계산
    step = chunk_size - overlap
    
    chunks = []
    # 3. for 반복문을 사용하여 명확한 단계로 텍스트 분할
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk_content = text[start:end]
        
        # 청크가 비어있지 않은 경우에만 추가
        if chunk_content:
            chunks.append({
                "text": chunk_content,
                "global_start": start,
                "size": len(chunk_content)
            })
            
    return chunks