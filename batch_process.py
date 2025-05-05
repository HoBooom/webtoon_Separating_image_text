#!/usr/bin/env python3
"""
웹툰 이미지 배치 처리 스크립트

여러 웹툰 이미지를 한 번에 처리하여 말풍선 내 텍스트를 추출하고 제거합니다.
"""

import os
import cv2
import argparse
import glob
import time
import json
from tqdm import tqdm
from speech_bubble_ocr import SpeechBubbleOCR

def parse_args():
    parser = argparse.ArgumentParser(description='웹툰 이미지 배치 처리 도구')
    parser.add_argument('--input_dir', type=str, required=True, help='처리할 웹툰 이미지가 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, default='batch_output', help='결과 저장 디렉토리')
    parser.add_argument('--model', type=str, default='./models/model.pt', 
                        help='사용할 YOLO 모델 파일 경로 (기본값: ./models/model.pt)')
    parser.add_argument('--api_key', type=str, required=True, help='Azure Computer Vision API 키')
    parser.add_argument('--endpoint', type=str, required=True, help='Azure Computer Vision API 엔드포인트')
    parser.add_argument('--lang', type=str, default='ko', help='OCR 언어 (ko, en, ja 등)')
    parser.add_argument('--confidence', type=float, default=0.5, help='말풍선 감지 신뢰도 임계값 (0.0-1.0)')
    parser.add_argument('--overlap', type=float, default=0.5, help='텍스트-말풍선 중첩 비율 임계값 (0.0-1.0)')
    parser.add_argument('--pattern', type=str, default='*.jpg,*.jpeg,*.png', 
                        help='처리할 이미지 파일 패턴 (쉼표로 구분)')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화 이미지 생성')
    return parser.parse_args()

def process_image(processor, image_path, output_dir, overlap, visualize=False):
    """
    단일 이미지 처리 및 결과 저장
    
    Args:
        processor: SpeechBubbleOCR 인스턴스
        image_path: 처리할 이미지 경로
        output_dir: 결과 저장 디렉토리
        overlap: 텍스트-말풍선 중첩 비율 임계값
        visualize: 시각화 이미지 생성 여부
        
    Returns:
        추출된 텍스트 수
    """
    # 파일명 추출
    image_basename = os.path.basename(image_path)
    name, ext = os.path.splitext(image_basename)
    
    # 출력 경로 설정
    clean_output_path = os.path.join(output_dir, f"{name}_clean{ext}")
    json_output_path = os.path.join(output_dir, f"{name}_text.json")
    
    # 말풍선 내 텍스트만 처리
    text_results, clean_image = processor.process_image(image_path, overlap_threshold=overlap)
    
    # 결과 저장
    cv2.imwrite(clean_output_path, clean_image)
    
    # JSON 형식으로 텍스트 결과 저장
    json_data = {
        "image": image_path,
        "total_texts": len(text_results),
        "texts": []
    }
    
    for idx, (key, item) in enumerate(text_results.items()):
        text_entry = {
            "id": idx,
            "text": item.get('text', ''),
            "bbox": item.get('bbox', []),
            "confidence": item.get('confidence', 0)
        }
        json_data["texts"].append(text_entry)
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # 시각화 이미지 생성 (선택사항)
    if visualize:
        vis_output_path = os.path.join(output_dir, f"{name}_visualized{ext}")
        result_image = processor.visualize_results(image_path)
        cv2.imwrite(vis_output_path, result_image)
    
    return len(text_results)

def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 이미지 파일 패턴 확장
    patterns = args.pattern.split(',')
    image_files = []
    for pattern in patterns:
        pattern_path = os.path.join(args.input_dir, pattern.strip())
        image_files.extend(glob.glob(pattern_path))
    
    # 중복 제거 및 정렬
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"Error: 지정된 패턴({args.pattern})과 일치하는 이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")
    
    # 말풍선 OCR 프로세서 초기화
    processor = SpeechBubbleOCR(
        model_path=args.model,
        azure_api_key=args.api_key,
        azure_endpoint=args.endpoint,
        lang=args.lang,
        confidence_threshold=args.confidence
    )
    
    # 결과 요약 파일
    summary_path = os.path.join(args.output_dir, "summary.json")
    
    # 처리 시작 시간
    start_time = time.time()
    
    # 모든 이미지 처리
    total_texts = 0
    results = []
    
    for image_path in tqdm(image_files, desc="이미지 처리"):
        try:
            text_count = process_image(
                processor, 
                image_path, 
                args.output_dir, 
                args.overlap, 
                args.visualize
            )
            total_texts += text_count
            results.append({
                "image": image_path,
                "filename": os.path.basename(image_path),
                "texts_count": text_count,
                "status": "success"
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                "image": image_path,
                "filename": os.path.basename(image_path),
                "texts_count": 0,
                "status": "error",
                "error_message": str(e)
            })
    
    # 처리 시간 계산
    elapsed_time = time.time() - start_time
    
    # 요약 저장 (JSON 형식)
    summary_data = {
        "total_images": len(image_files),
        "total_texts_extracted": total_texts,
        "processing_time_seconds": elapsed_time,
        "average_time_per_image": elapsed_time / len(image_files) if image_files else 0,
        "settings": {
            "model": args.model,
            "confidence_threshold": args.confidence,
            "overlap_threshold": args.overlap,
            "language": args.lang
        },
        "results": results
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n처리 완료!")
    print(f"총 추출된 텍스트 수: {total_texts}")
    print(f"처리 시간: {elapsed_time:.2f}초")
    print(f"결과는 '{args.output_dir}' 디렉토리에 저장되었습니다.")
    print(f"요약 정보 (JSON): {summary_path}")

if __name__ == "__main__":
    main() 