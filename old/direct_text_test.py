import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from text_detector import TextDetector
from text_processor import TextProcessor

def show_image(img, title="Image", figsize=(10, 10)):
    """이미지를 화면에 표시합니다."""
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def test_text_detection(image_path):
    """텍스트 감지 테스트"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: 이미지를 불러올 수 없습니다: {image_path}")
        return
    
    print("텍스트 영역 감지 테스트 중...")
    detector = TextDetector()
    
    # 전처리된 이미지 확인 (디버깅용)
    preprocessed = detector.preprocess_image(image)
    show_image(preprocessed, "전처리된 이미지")
    
    # 텍스트 영역 감지
    text_regions = detector.detect_text_regions(image)
    print(f"감지된 텍스트 영역 수: {len(text_regions)}")
    
    # 결과 시각화
    result = detector.visualize_text_regions(image, text_regions)
    show_image(result, "텍스트 영역 감지 결과")
    
    return image, text_regions

def test_text_processing(image, text_regions, lang='korean'):
    """텍스트 처리 테스트"""
    print("텍스트 처리 테스트 중 (PaddleOCR 사용)...")
    
    # 텍스트 처리기 초기화
    processor = TextProcessor(lang=lang)
    
    # 모든 텍스트 영역 처리
    results, clean_image = processor.process_all_regions(image, text_regions)
    
    # 결과 출력
    for i, data in results.items():
        print(f"텍스트 영역 {i}:")
        print(f"  위치: {data['bbox']}")
        print(f"  텍스트: {data['text']}")
        print("---")
    
    # 결과 시각화
    show_image(image, "원본 이미지")
    show_image(clean_image, "텍스트 제거 결과")
    
    return results, clean_image

def parse_args():
    parser = argparse.ArgumentParser(description='웹툰 텍스트 처리 테스트')
    parser.add_argument('--image', type=str, required=True, help='테스트할 이미지 경로')
    parser.add_argument('--lang', type=str, default='korean', help='OCR 언어 (korean, en, japan 등)')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'detect', 'process'],
                        help='테스트 모드 (all: 모든 기능, detect: 텍스트 감지, '
                             'process: 텍스트 처리)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 모드에 따라 적절한 테스트 실행
    if args.mode in ['all', 'detect']:
        # 텍스트 감지 테스트
        image, text_regions = test_text_detection(args.image)
        
        if args.mode == 'detect':
            return
    
    if args.mode in ['all', 'process']:
        # 텍스트 처리 테스트
        if 'image' not in locals():
            image, text_regions = test_text_detection(args.image)
        
        results, clean_image = test_text_processing(image, text_regions, args.lang)

if __name__ == "__main__":
    main() 