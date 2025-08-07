# document_processor.py (최종 청크 분할 테스트용)

def chunk_text(text, chunk_size=100000, overlap_ratio=0.2):
    """
    텍스트를 지정된 크기와 중첩 비율로 나누는, 검증된 최종 함수.
    """
    overlap = int(chunk_size * overlap_ratio)
    step = chunk_size - overlap
    
    # 텍스트가 청크 크기보다 작거나 같으면, 그대로 단일 청크로 반환
    if len(text) <= chunk_size:
        return [{"start": 0, "end": len(text)}]

    chunks_indices = []
    
    # for 반복문을 사용하여 명확한 단계로 텍스트 분할
    for start in range(0, len(text), step):
        end = start + chunk_size
        
        # 마지막 청크가 문서 끝을 초과하지 않도록 조정
        if end > len(text):
            end = len(text)
        
        chunks_indices.append({
            "start": start,
            "end": end
        })
        
        # 현재 청크의 끝이 텍스트의 끝에 도달했거나 넘어섰다면 중단
        if end == len(text):
            break
            
    return chunks_indices