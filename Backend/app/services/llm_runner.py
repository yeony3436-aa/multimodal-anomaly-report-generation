# app/services/llm_runner.py ->임시

async def generate_inspection_report(image_path: str, score: float):
    """
    지금은 가짜 데이터를 리턴하지만, 
    나중에는 여기에 InternVL/Llava 모델 추론 코드를 넣습니다.
    """
    # [Mocking] 점수가 높으면 불량 멘트, 낮으면 정상 멘트 리턴
    if score > 3.0:
        return f"주의: {image_path}에서 심각한 표면 긁힘이 감지되었습니다. (점수: {score})"
    else:
        return "정상: 특이사항이 발견되지 않았습니다."