import os
import argparse
import cv2
import json
from datetime import datetime

from text_detector import TextDetector
from text_processor import TextProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='웹툰 텍스트 추출 및 제거 프로그램')
    parser.add_argument('--input', type=str, required=True, help='입력 이미지 파일 경로')
    parser.add_argument('--output_dir', type=str, default='output', help='결과 저장 디렉토리')
    parser.add_argument('--tesseract_path', type=str, default=None, 
                        help='더 이상 사용되지 않음 (Tesseract 호환성 유지)')
    parser.add_argument('--lang', type=str, default='korean', 
                        help='OCR 언어 (기본값: korean, 가능한 값: korean, en, japan 등)')
    parser.add_argument('--visualize', action='store_true', 
                        help='텍스트 감지 결과 시각화')
    
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
    
    # 텍스트 영역 감지
    print("텍스트 영역 감지 중...")
    detector = TextDetector()
    text_regions = detector.detect_text_regions(image)
    print(f"감지된 텍스트 영역 수: {len(text_regions)}")
    
    # 텍스트 영역 시각화 (옵션)
    if args.visualize:
        print("텍스트 영역 시각화 중...")
        viz_image = detector.visualize_text_regions(image, text_regions)
        viz_path = os.path.join(output_dir, f"{timestamp}_text_regions.jpg")
        cv2.imwrite(viz_path, viz_image)
        print(f"텍스트 영역 시각화 저장: {viz_path}")
    
    # 텍스트 처리 (추출 및 제거)
    print("텍스트 처리 중 (PaddleOCR 사용)...")
    processor = TextProcessor(tesseract_cmd=args.tesseract_path, lang=args.lang)
    results, clean_image = processor.process_all_regions(image, text_regions)
    
    # 추출된 텍스트 저장
    text_path = os.path.join(output_dir, f"{timestamp}_texts.json")
    with open(text_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"추출된 텍스트 저장: {text_path}")
    
    # 텍스트가 제거된 이미지 저장
    clean_path = os.path.join(output_dir, f"{timestamp}_clean.jpg")
    cv2.imwrite(clean_path, clean_image)
    print(f"텍스트가 제거된 이미지 저장: {clean_path}")
    
    print("처리 완료!")

if __name__ == "__main__":
    main() 