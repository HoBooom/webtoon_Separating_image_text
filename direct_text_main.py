import os
import argparse
import cv2
import json
import shutil
from datetime import datetime

from azure_text_processor import AzureTextProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='웹툰 텍스트 추출 및 제거 프로그램')
    parser.add_argument('--input', type=str, required=True, help='입력 이미지 파일 경로')
    parser.add_argument('--output_dir', type=str, default='output', help='결과 저장 디렉토리')
    parser.add_argument('--api_key', type=str, required=True, 
                        help='Azure Computer Vision API Key')
    parser.add_argument('--endpoint', type=str, required=True, 
                        help='Azure Computer Vision API Endpoint')
    parser.add_argument('--lang', type=str, default='ko', 
                        help='OCR 언어 (기본값: ko, 가능한 값: ko, en, ja 등)')
    parser.add_argument('--visualize', action='store_true', 
                        help='텍스트 감지 결과 시각화 (콘솔 출력용, 저장은 항상 됨)')
    parser.add_argument('--use_local_detection', action='store_true',
                        help='로컬 텍스트 감지 사용 (기본: Azure OCR 감지 사용)')
    
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
    
    # 원본 이미지 저장
    original_path = os.path.join(output_dir, f"{timestamp}_original.jpg")
    cv2.imwrite(original_path, image)
    print(f"원본 이미지 저장: {original_path}")
    
    # Azure OCR 프로세서 초기화
    processor = AzureTextProcessor(
        api_key=args.api_key,
        endpoint=args.endpoint,
        lang=args.lang
    )
    
    # 로컬 텍스트 감지 사용 여부 확인
    if args.use_local_detection:
        # 로컬 텍스트 영역 감지 사용
        from text_detector import TextDetector
        print("로컬 방식으로 텍스트 영역 감지 중...")
        detector = TextDetector()
        text_regions = detector.detect_text_regions(image)
        print(f"감지된 텍스트 영역 수: {len(text_regions)}")
        
        # 텍스트 영역 시각화 이미지 생성 및 저장
        viz_image = detector.visualize_text_regions(image, text_regions)
        viz_path = os.path.join(output_dir, f"{timestamp}_text_regions.jpg")
        cv2.imwrite(viz_path, viz_image)
        print(f"텍스트 영역 시각화 저장: {viz_path}")
        
        # 콘솔 출력용 시각화 옵션
        if args.visualize:
            print("텍스트 영역 시각화 콘솔 출력...")
            # 여기에 콘솔 출력 코드 추가 가능 (필요시)
        
        # 이미지 처리 (영역별 OCR 및 제거)
        print("Azure OCR로 텍스트 처리 중...")
        results, clean_image = processor.process_image(image, text_regions)
    else:
        # Azure OCR API를 사용하여 텍스트 감지 및 인식
        print("Azure OCR API로 텍스트 감지 및 인식 중...")
        results = processor.process_entire_image(image_path)
        print(f"감지된 텍스트 영역 수: {len(results)}")
        
        # 텍스트 영역 시각화 이미지 생성 및 저장
        viz_image = processor.visualize_text_regions(image, results)
        viz_path = os.path.join(output_dir, f"{timestamp}_text_regions.jpg")
        cv2.imwrite(viz_path, viz_image)
        print(f"텍스트 영역 시각화 저장: {viz_path}")
        
        # 콘솔 출력용 시각화 옵션
        if args.visualize:
            print("텍스트 영역 시각화 콘솔 출력...")
            # 여기에 콘솔 출력 코드 추가 가능 (필요시)
        
        # 텍스트 제거
        print("텍스트 영역 제거 중...")
        clean_image = processor.remove_all_text(image, results)
    
    # 추출된 텍스트 저장
    text_path = os.path.join(output_dir, f"{timestamp}_texts.json")
    with open(text_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"추출된 텍스트 저장: {text_path}")
    
    # 텍스트가 제거된 이미지 저장
    clean_path = os.path.join(output_dir, f"{timestamp}_clean.jpg")
    cv2.imwrite(clean_path, clean_image)
    print(f"텍스트가 제거된 이미지 저장: {clean_path}")
    
    print(f"\n모든 결과물이 '{output_dir}' 디렉토리에 저장되었습니다:")
    print(f"1. 원본 이미지: {os.path.basename(original_path)}")
    print(f"2. 텍스트 영역 표시된 이미지: {os.path.basename(viz_path)}")
    print(f"3. 텍스트가 제거된 이미지: {os.path.basename(clean_path)}")
    print(f"4. 추출된 텍스트 데이터: {os.path.basename(text_path)}")
    
    print("처리 완료!")

if __name__ == "__main__":
    main() 