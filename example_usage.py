import os
import cv2
import json
import time
from speech_bubble_ocr import SpeechBubbleOCR
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Webtoon 이미지 내 말풍선 텍스트 추출 도구')
    parser.add_argument('--image', type=str, required=True, help='처리할 웹툰 이미지 경로')
    parser.add_argument('--model', type=str, default='./models/model.pt', 
                        help='사용할 YOLO 모델 파일 경로 (기본값: ./models/model.pt)')
    parser.add_argument('--api_key', type=str, required=True, help='Azure Computer Vision API 키')
    parser.add_argument('--endpoint', type=str, required=True, help='Azure Computer Vision API 엔드포인트')
    parser.add_argument('--lang', type=str, default='ko', help='OCR 언어 (ko, en, ja 등)')
    parser.add_argument('--output_dir', type=str, default='output', help='결과 저장 디렉토리')
    parser.add_argument('--overlap', type=float, default=0.5, 
                        help='텍스트가 말풍선 내에 있다고 판단할 최소 중첩 비율 (0.0-1.0)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='말풍선 감지 신뢰도 임계값 (0.0-1.0)')
    parser.add_argument('--merge', action='store_true', default=True,
                        help='가까운 텍스트 박스 병합 활성화 (기본값: 활성화)')
    parser.add_argument('--no-merge', dest='merge', action='store_false',
                        help='텍스트 박스 병합 비활성화')
    parser.add_argument('--merge_distance', type=int, default=20,
                        help='텍스트 박스 병합 거리 임계값 (픽셀, 기본값: 20)')
    parser.add_argument('--merge_any_overlap', action='store_true', default=True,
                        help='겹치는 부분이 있는 모든 텍스트 박스 병합 (기본값: 활성화)')
    parser.add_argument('--no-merge_any_overlap', dest='merge_any_overlap', action='store_false',
                        help='겹침 기반 병합 비활성화, 거리만 사용')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 파일명 추출
    image_basename = os.path.basename(args.image)
    name, ext = os.path.splitext(image_basename)
    
    # 현재 시간을 파일명에 추가
    current_time = time.strftime("%Y%m%d_%H%M%S")
    
    # 출력 경로 설정 (시간 추가 및 PNG 고정)
    vis_output_path = os.path.join(args.output_dir, f"{current_time}_{name}_visualized{ext}")
    clean_output_path = os.path.join(args.output_dir, f"{current_time}_{name}_clean.png") # .png로 고정
    json_output_path = os.path.join(args.output_dir, f"{current_time}_{name}_text.json")
    
    # 말풍선 OCR 프로세서 초기화
    processor = SpeechBubbleOCR(
        model_path=args.model,
        azure_api_key=args.api_key,
        azure_endpoint=args.endpoint,
        lang=args.lang,
        confidence_threshold=args.confidence
    )
    
    # 텍스트 박스 병합 설정
    processor.merge_boxes = args.merge
    processor.merge_distance_threshold = args.merge_distance
    processor.merge_any_overlap = args.merge_any_overlap
    
    print(f"이미지 처리 중: {args.image}")
    print(f"사용 모델: {args.model}")
    print(f"말풍선 감지 신뢰도 임계값: {args.confidence}")
    print(f"텍스트-말풍선 중첩 비율 임계값: {args.overlap}")
    
    merge_mode = "비활성화"
    if args.merge:
        if args.merge_any_overlap:
            merge_mode = f"활성화 (겹치는 모든 박스 병합, 거리 임계값: {args.merge_distance}px)"
        else:
            merge_mode = f"활성화 (거리 기반 병합, 임계값: {args.merge_distance}px)"
    
    print(f"텍스트 박스 병합: {merge_mode}")
    
    # 말풍선 내 텍스트만 처리 (중첩 임계값과 병합 거리 적용)
    text_results, clean_image = processor.process_image(
        args.image, 
        overlap_threshold=args.overlap,
        merge_distance=args.merge_distance
    )
    
    # 결과 시각화
    result_image = processor.visualize_results(args.image)
    
    # 결과 저장
    cv2.imwrite(vis_output_path, result_image)
    cv2.imwrite(clean_output_path, clean_image)
    
    # JSON 형식으로 텍스트 결과 저장
    json_data = {
        "image": args.image,
        "total_texts": len(text_results),
        "merge_enabled": args.merge,
        "merge_distance": args.merge_distance if args.merge else None,
        "merge_any_overlap": args.merge_any_overlap if args.merge else None,
        "texts": []
    }
    
    for idx, (key, item) in enumerate(text_results.items()):
        text_entry = {
            "id": idx,
            "text": item.get('text', ''),
            "bbox": item.get('bbox', []),
            "confidence": item.get('confidence', 0.0), # batch_process.py와 동일하게 float으로
            # 상세 OCR 정보 추가
            #"font_style_details": item.get('font_style_details', {}),
            #"appearance_details": item.get('appearance_details', {}),
            #"geometry_details": item.get('geometry_details', {})
        }
        json_data["texts"].append(text_entry)
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"처리 완료!")
    print(f"결과 시각화 이미지: {vis_output_path}")
    print(f"텍스트 삭제된 이미지: {clean_output_path}")
    print(f"추출된 텍스트 (JSON): {json_output_path}")
    print(f"말풍선 내 텍스트 개수: {len(text_results)}")

if __name__ == "__main__":
    main() 