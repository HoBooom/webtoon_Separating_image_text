#!/usr/bin/env python3
"""
말풍선 감지 및 텍스트 필터링 파라미터 튜닝 도구

이 스크립트는 다양한 말풍선 감지 신뢰도 및 텍스트-말풍선 중첩 비율 임계값을 
테스트하여 최적의 결과를 찾을 수 있게 도와줍니다.
"""

import os
import cv2
import numpy as np
import argparse
from speech_bubble_ocr import SpeechBubbleOCR

def parse_args():
    parser = argparse.ArgumentParser(description='말풍선 감지 파라미터 튜닝 도구')
    parser.add_argument('--image', type=str, required=True, help='테스트할 웹툰 이미지 경로')
    parser.add_argument('--model', type=str, default='kitsumed/yolov8m_seg-speech-bubble', 
                        help='사용할 Hugging Face 말풍선 감지 모델')
    parser.add_argument('--api_key', type=str, required=True, help='Azure Computer Vision API 키')
    parser.add_argument('--endpoint', type=str, required=True, help='Azure Computer Vision API 엔드포인트')
    parser.add_argument('--output_dir', type=str, default='tune_results', help='결과 저장 디렉토리')
    parser.add_argument('--confidence_steps', type=int, default=5, 
                        help='테스트할 감지 신뢰도 단계 수 (0.3-0.8 범위에서 균등 분할)')
    parser.add_argument('--overlap_steps', type=int, default=5, 
                        help='테스트할 중첩 비율 단계 수 (0.2-0.9 범위에서 균등 분할)')
    return parser.parse_args()

def create_comparison_grid(images, titles, output_path, grid_size=None):
    """
    여러 이미지를 그리드로 배치하여 비교 이미지 생성
    
    Args:
        images: 이미지 목록
        titles: 각 이미지의 제목 목록
        output_path: 결과 이미지 저장 경로
        grid_size: 그리드 크기 (rows, cols), None이면 자동 계산
    """
    n_images = len(images)
    if n_images == 0:
        return
    
    # 이미지 크기 확인 (모든 이미지가 같은 크기라고 가정)
    height, width, channels = images[0].shape
    
    # 그리드 크기 계산
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # 결과 이미지 크기 계산
    title_height = 30  # 제목을 위한 추가 공간
    result_height = rows * (height + title_height)
    result_width = cols * width
    
    # 결과 이미지 생성
    result = np.ones((result_height, result_width, channels), dtype=np.uint8) * 255
    
    # 이미지 배치
    for i in range(n_images):
        row = i // cols
        col = i % cols
        
        # 이미지 위치 계산
        y = row * (height + title_height)
        x = col * width
        
        # 이미지 복사
        result[y+title_height:y+height+title_height, x:x+width] = images[i]
        
        # 제목 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, titles[i], (x+10, y+20), font, 0.5, (0, 0, 0), 1)
    
    # 결과 저장
    cv2.imwrite(output_path, result)
    return result

def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 파일명 추출
    image_basename = os.path.basename(args.image)
    name, ext = os.path.splitext(image_basename)
    
    # 기본 OCR 프로세서 초기화
    processor = SpeechBubbleOCR(
        model_path=args.model,
        azure_api_key=args.api_key,
        azure_endpoint=args.endpoint
    )
    
    print(f"이미지 로드 중: {args.image}")
    original_image = cv2.imread(args.image)
    if original_image is None:
        print(f"Error: Could not read image from {args.image}")
        return
    
    # 테스트할 파라미터 값 생성
    confidence_values = np.linspace(0.3, 0.8, args.confidence_steps)
    overlap_values = np.linspace(0.2, 0.9, args.overlap_steps)
    
    print("파라미터 튜닝 시작...")
    
    # 신뢰도 임계값 테스트 (중첩 비율 기본값 0.5 사용)
    confidence_images = []
    confidence_titles = []
    
    for conf in confidence_values:
        print(f"감지 신뢰도 테스트 중: {conf:.2f}")
        processor.confidence_threshold = conf
        
        # 결과 이미지 생성
        result_image = processor.visualize_results(args.image)
        confidence_images.append(result_image)
        confidence_titles.append(f"Confidence: {conf:.2f}")
        
        # 개별 결과 저장
        output_path = os.path.join(args.output_dir, f"{name}_conf_{conf:.2f}{ext}")
        cv2.imwrite(output_path, result_image)
    
    # 중첩 비율 임계값 테스트 (기본 신뢰도 0.5 사용)
    overlap_images = []
    overlap_titles = []
    processor.confidence_threshold = 0.5
    
    for overlap in overlap_values:
        print(f"중첩 비율 테스트 중: {overlap:.2f}")
        
        # 말풍선 내 텍스트만 처리
        text_results, _ = processor.process_image(args.image, overlap_threshold=overlap)
        
        # 원본 이미지에 결과 시각화
        result_image = original_image.copy()
        
        # 말풍선 감지
        speech_bubbles = processor.detect_speech_bubbles(original_image)
        
        # 말풍선 그리기 (파란색)
        for bubble in speech_bubbles:
            minr, minc, maxr, maxc = bubble
            cv2.rectangle(result_image, (minc, minr), (maxc, maxr), (255, 0, 0), 2)
        
        # 텍스트 영역 그리기 (녹색)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for idx, (key, item) in enumerate(text_results.items()):
            bbox = item['bbox']
            minr, minc, maxr, maxc = bbox
            cv2.rectangle(result_image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            cv2.putText(result_image, f"{idx}", (minc, minr-10), font, 0.5, (0, 255, 0), 1)
        
        overlap_images.append(result_image)
        overlap_titles.append(f"Overlap: {overlap:.2f} (Count: {len(text_results)})")
        
        # 개별 결과 저장
        output_path = os.path.join(args.output_dir, f"{name}_overlap_{overlap:.2f}{ext}")
        cv2.imwrite(output_path, result_image)
    
    # 비교 그리드 생성
    conf_grid_path = os.path.join(args.output_dir, f"{name}_confidence_comparison{ext}")
    overlap_grid_path = os.path.join(args.output_dir, f"{name}_overlap_comparison{ext}")
    
    create_comparison_grid(confidence_images, confidence_titles, conf_grid_path)
    create_comparison_grid(overlap_images, overlap_titles, overlap_grid_path)
    
    print(f"튜닝 완료!")
    print(f"감지 신뢰도 비교: {conf_grid_path}")
    print(f"중첩 비율 비교: {overlap_grid_path}")
    print(f"개별 결과는 '{args.output_dir}' 디렉토리에 저장되었습니다.")
    
    # 최적 파라미터 추천
    print("\n=== 권장 파라미터 ===")
    print("각 웹툰 스타일에 맞는 최적의 파라미터를 비교 이미지에서 선택하세요:")
    print("1. 감지 신뢰도(confidence): 말풍선이 정확하게 감지되는 최소값 선택")
    print("2. 중첩 비율(overlap): 말풍선 내 텍스트만 정확히 포함되는 값 선택")
    print("\n사용 예시:")
    print(f"python example_usage.py --image {args.image} --api_key your_api_key --endpoint your_endpoint --confidence 0.5 --overlap 0.6")

if __name__ == "__main__":
    main() 