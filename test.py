import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from bubble_detector import BubbleDetector
from advanced_bubble_detector import AdvancedBubbleDetector
from text_extractor import TextExtractor
from text_remover import TextRemover

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

def test_bubble_detection(image_path, use_advanced=True):
    """말풍선 감지 테스트"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: 이미지를 불러올 수 없습니다: {image_path}")
        return
    
    # 말풍선 감지
    if use_advanced:
        detector = AdvancedBubbleDetector()
        title_prefix = "고급 "
    else:
        detector = BubbleDetector()
        title_prefix = "기본 "
    
    print(f"{title_prefix}말풍선 감지 테스트 중...")
    bubbles = detector.detect_bubbles(image)
    print(f"감지된 말풍선 수: {len(bubbles)}")
    
    # 결과 시각화
    result = detector.visualize_bubbles(image, bubbles)
    show_image(result, f"{title_prefix}말풍선 감지 결과")
    
    return image, bubbles

def test_text_extraction(image, bubbles, tesseract_path=None):
    """텍스트 추출 테스트"""
    print("텍스트 추출 테스트 중...")
    
    # 텍스트 추출
    extractor = TextExtractor(tesseract_cmd=tesseract_path)
    results = extractor.extract_all_texts(image, bubbles)
    
    # 결과 출력
    for i, data in results.items():
        print(f"말풍선 {i}:")
        print(f"  위치: {data['bbox']}")
        print(f"  텍스트: {data['text']}")
        print("---")
    
    return results

def test_text_removal(image, bubbles):
    """텍스트 제거 테스트"""
    print("텍스트 제거 테스트 중...")
    
    # 텍스트 제거
    remover = TextRemover()
    clean_image = remover.remove_all_texts(image, bubbles)
    
    # 결과 시각화
    show_image(image, "원본 이미지")
    show_image(clean_image, "텍스트 제거 결과")
    
    return clean_image

def parse_args():
    parser = argparse.ArgumentParser(description='웹툰 말풍선 처리 테스트')
    parser.add_argument('--image', type=str, required=True, help='테스트할 이미지 경로')
    parser.add_argument('--tesseract', type=str, default=None, help='Tesseract 실행 파일 경로')
    parser.add_argument('--basic', action='store_true', help='기본 말풍선 감지 사용')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'detect', 'extract', 'remove'],
                        help='테스트 모드 (all: 모든 기능, detect: 말풍선 감지, '
                             'extract: 텍스트 추출, remove: 텍스트 제거)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 모드에 따라 적절한 테스트 실행
    if args.mode in ['all', 'detect']:
        # 말풍선 감지 테스트
        image, bubbles = test_bubble_detection(args.image, not args.basic)
        
        if args.mode == 'detect':
            return
    
    if args.mode in ['all', 'extract']:
        # 텍스트 추출 테스트
        if 'image' not in locals():
            image, bubbles = test_bubble_detection(args.image, not args.basic)
        
        results = test_text_extraction(image, bubbles, args.tesseract)
        
        if args.mode == 'extract':
            return
    
    if args.mode in ['all', 'remove']:
        # 텍스트 제거 테스트
        if 'image' not in locals():
            image, bubbles = test_bubble_detection(args.image, not args.basic)
        
        clean_image = test_text_removal(image, bubbles)

if __name__ == "__main__":
    main() 