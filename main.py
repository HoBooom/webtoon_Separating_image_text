import os
import argparse
import cv2
import json
from datetime import datetime

from bubble_detector import BubbleDetector
from advanced_bubble_detector import AdvancedBubbleDetector
from text_extractor import TextExtractor
from text_remover import TextRemover

def parse_args():
    parser = argparse.ArgumentParser(description='웹툰 말풍선 텍스트 추출 및 제거 프로그램')
    parser.add_argument('--input', type=str, required=True, help='입력 이미지 파일 경로')
    parser.add_argument('--output_dir', type=str, default='output', help='결과 저장 디렉토리')
    parser.add_argument('--tesseract_path', type=str, default=None, 
                        help='Tesseract 실행 파일 경로 (Windows에서 필요)')
    parser.add_argument('--lang', type=str, default='kor+eng', 
                        help='OCR 언어 (기본값: 한국어+영어)')
    parser.add_argument('--visualize', action='store_true', 
                        help='말풍선 감지 결과 시각화')
    parser.add_argument('--use_advanced', action='store_true', default=True,
                        help='고급 말풍선 감지 알고리즘 사용 (기본값: True)')
    
    return parser.parse_args()

def main():
    # 명령줄 인자 파싱
    args = parse_args()
    
    # 입력 이미지 로드
    image_path = args.input
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: 이미지를 불러올 수 없습니다: {image_path}")
        return
    
    # 결과 저장 디렉토리 생성
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 타임스탬프 생성 (결과 파일 이름용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 말풍선 감지
    print("말풍선 감지 중...")
    if args.use_advanced:
        detector = AdvancedBubbleDetector()
    else:
        detector = BubbleDetector()
    bubbles = detector.detect_bubbles(image)
    print(f"감지된 말풍선 수: {len(bubbles)}")
    
    # 말풍선 시각화 (옵션)
    if args.visualize:
        print("말풍선 시각화 중...")
        viz_image = detector.visualize_bubbles(image, bubbles)
        viz_path = os.path.join(output_dir, f"{timestamp}_bubbles.jpg")
        cv2.imwrite(viz_path, viz_image)
        print(f"말풍선 시각화 저장: {viz_path}")
    
    # 텍스트 추출
    print("텍스트 추출 중...")
    extractor = TextExtractor(tesseract_cmd=args.tesseract_path, lang=args.lang)
    results = extractor.extract_all_texts(image, bubbles)
    
    # 추출된 텍스트 저장
    text_path = os.path.join(output_dir, f"{timestamp}_texts.json")
    with open(text_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"추출된 텍스트 저장: {text_path}")
    
    # 텍스트 제거
    print("텍스트 제거 중...")
    remover = TextRemover()
    clean_image = remover.remove_all_texts(image, bubbles)
    
    # 텍스트가 제거된 이미지 저장
    clean_path = os.path.join(output_dir, f"{timestamp}_clean.jpg")
    cv2.imwrite(clean_path, clean_image)
    print(f"텍스트가 제거된 이미지 저장: {clean_path}")
    
    print("처리 완료!")

if __name__ == "__main__":
    main() 