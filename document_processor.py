# document_processor.py (글자 수 기준 최종 테스트용)

def chunk_text_by_char(text, chunk_size_chars=100000, overlap_chars=20000):
    """
    텍스트를 '글자 수' 기준으로 나누는, 새롭고 검증된 최종 함수.
    """
    # 텍스트가 청크 크기보다 작거나 같으면, 그대로 단일 청크로 반환
    if len(text) <= chunk_size_chars:
        return [{"start_char": 0, "end_char": len(text)}]

    chunks_indices = []
    start_char = 0
    
    # while 루프를 사용하여, 텍스트의 끝에 도달할 때까지 반복
    while start_char < len(text):
        # 현재 청크의 끝 위치 계산
        end_char = start_char + chunk_size_chars
        
        # 청크의 시작과 끝 위치 저장
        chunks_indices.append({
            "start_char": start_char,
            "end_char": end_char 
            # 실제 텍스트의 끝을 넘지 않도록 min()을 사용해도 되지만, 
            # 슬라이싱 [start:end]은 end가 길이를 넘어도 자동으로 처리해 주므로 
            # 로직을 단순하게 유지합니다.
        })
        
        # 다음 청크의 시작 위치를 중첩을 고려하여 계산
        next_start = start_char + chunk_size_chars - overlap_chars
        
        # 다음 시작 위치가 이전과 같다면 무한 루프이므로 중단
        if next_start <= start_char:
            break
            
        start_char = next_start
            
    return chunks_indices