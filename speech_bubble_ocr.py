import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import huggingface_hub
from azure_text_processor import AzureTextProcessor

class SpeechBubbleOCR:
    def __init__(self, model_path, azure_api_key, azure_endpoint, lang='ko', confidence_threshold=0.5):
        """
        말풍선 감지 및 OCR 통합 모듈 초기화
        
        Args:
            model_path: YOLOv8 말풍선 감지 모델 경로 (Hugging Face model ID 또는 로컬 경로)
            azure_api_key: Azure Computer Vision API 키
            azure_endpoint: Azure Computer Vision API 엔드포인트
            lang: OCR 인식 언어 (기본값: 한국어)
            confidence_threshold: 말풍선 감지 신뢰도 임계값
        """
        # YOLO 모델 로드 - Hugging Face ID인 경우 먼저 다운로드
        if '/' in model_path and not os.path.exists(model_path):
            try:
                print(f"Hugging Face에서 모델 다운로드 중: {model_path}")
                model_file = huggingface_hub.hf_hub_download(
                    repo_id=model_path,
                    filename="model.pt"
                )
                self.model = YOLO(model_file)
                print("모델 다운로드 및 로드 완료")
            except Exception as e:
                print(f"Hugging Face 모델 다운로드 오류: {e}")
                raise
        else:
            self.model = YOLO(model_path)
            
        self.confidence_threshold = confidence_threshold
        
        # Azure OCR 프로세서 초기화
        self.ocr_processor = AzureTextProcessor(azure_api_key, azure_endpoint, lang)
        
        # 기본 중첩 임계값 설정
        self.overlap_threshold = 0.5
    
    def detect_speech_bubbles(self, image):
        """
        YOLOv8 모델을 사용하여 이미지에서 말풍선을 감지합니다.
        
        Args:
            image: OpenCV 이미지 (BGR)
            
        Returns:
            말풍선 영역 목록 (각 항목은 [minr, minc, maxr, maxc] 형식)
        """
        # OpenCV 이미지를 PIL로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 모델을 사용하여 말풍선 감지
        results = self.model(pil_image)
        
        # 감지된 말풍선 영역 추출
        speech_bubbles = []
        
        for result in results:
            # 세그멘테이션 결과 가져오기 (mask 있는 경우)
            if hasattr(result, 'masks') and result.masks is not None:
                for i, box in enumerate(result.boxes):
                    # 신뢰도 임계값 확인
                    conf = float(box.conf)
                    if conf >= self.confidence_threshold:
                        # 바운딩 박스 좌표 추출 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # OpenCV 형식으로 변환 (min_row, min_col, max_row, max_col)
                        speech_bubbles.append([y1, x1, y2, x2])
            else:
                # 세그멘테이션 없는 경우 바운딩 박스만 사용
                for i, box in enumerate(result.boxes):
                    # 신뢰도 임계값 확인
                    conf = float(box.conf)
                    if conf >= self.confidence_threshold:
                        # 바운딩 박스 좌표 추출 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # OpenCV 형식으로 변환 (min_row, min_col, max_row, max_col)
                        speech_bubbles.append([y1, x1, y2, x2])
        
        return speech_bubbles
    
    def is_text_in_speech_bubble(self, text_bbox, speech_bubbles, overlap_threshold=None):
        """
        텍스트 영역이 말풍선 내에 있는지 확인합니다.
        
        Args:
            text_bbox: 텍스트 영역 [minr, minc, maxr, maxc]
            speech_bubbles: 말풍선 영역 목록
            overlap_threshold: 텍스트가 말풍선 내에 있다고 판단할 최소 중첩 비율
            
        Returns:
            텍스트가 말풍선 내에 있으면 True, 그렇지 않으면 False
        """
        # 중첩 임계값이 지정되지 않은 경우 기본값 사용
        if overlap_threshold is None:
            overlap_threshold = self.overlap_threshold
            
        text_minr, text_minc, text_maxr, text_maxc = text_bbox
        text_area = (text_maxr - text_minr) * (text_maxc - text_minc)
        
        if text_area <= 0:
            return False
        
        for bubble in speech_bubbles:
            bubble_minr, bubble_minc, bubble_maxr, bubble_maxc = bubble
            
            # 중첩 영역 계산
            overlap_minr = max(text_minr, bubble_minr)
            overlap_minc = max(text_minc, bubble_minc)
            overlap_maxr = min(text_maxr, bubble_maxr)
            overlap_maxc = min(text_maxc, bubble_maxc)
            
            # 중첩이 있는 경우
            if overlap_minr < overlap_maxr and overlap_minc < overlap_maxc:
                overlap_area = (overlap_maxr - overlap_minr) * (overlap_maxc - overlap_minc)
                overlap_ratio = overlap_area / text_area
                
                # 중첩 비율이 임계값보다 크면 말풍선 내에 있다고 판단
                if overlap_ratio >= overlap_threshold:
                    return True
        
        return False
    
    def process_image(self, image_path, overlap_threshold=None):
        """
        이미지에서 말풍선을 감지하고 그 내부 텍스트만 추출합니다.
        
        Args:
            image_path: 이미지 경로
            overlap_threshold: 텍스트가 말풍선 내에 있다고 판단할 최소 중첩 비율
            
        Returns:
            텍스트 추출 결과와 처리된 이미지 (말풍선 내 텍스트가 삭제된)
        """
        # 중첩 임계값이 지정되지 않은 경우 기본값 사용
        if overlap_threshold is not None:
            self.overlap_threshold = overlap_threshold
            
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return {}, None
        
        # 1. 말풍선 감지
        speech_bubbles = self.detect_speech_bubbles(image)
        
        # 2. Azure OCR로 전체 이미지 텍스트 인식
        all_text_results = self.ocr_processor.process_entire_image(image_path)
        
        # 3. 말풍선 내 텍스트만 필터링
        filtered_results = {}
        count = 0
        for _, item in all_text_results.items():
            if isinstance(item, dict) and 'bbox' in item:
                bbox = item['bbox']
                # 텍스트가 말풍선 내에 있는지 확인
                if self.is_text_in_speech_bubble(bbox, speech_bubbles):
                    filtered_results[count] = item
                    count += 1
        
        # 4. 말풍선 내 텍스트만 삭제한 이미지 생성
        clean_image = self.ocr_processor.remove_all_text(image, filtered_results)
        
        return filtered_results, clean_image
    
    def visualize_results(self, image_path, output_path=None):
        """
        결과를 시각화하여 표시합니다.
        
        Args:
            image_path: 원본 이미지 경로
            output_path: 결과 이미지 저장 경로 (없으면 저장하지 않음)
            
        Returns:
            시각화된 이미지
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # 말풍선 감지
        speech_bubbles = self.detect_speech_bubbles(image)
        
        # 텍스트 인식 및 필터링
        text_results, _ = self.process_image(image_path)
        
        # 결과 이미지 생성
        result_image = image.copy()
        
        # 말풍선 그리기 (파란색)
        for bubble in speech_bubbles:
            minr, minc, maxr, maxc = bubble
            cv2.rectangle(result_image, (minc, minr), (maxc, maxr), (255, 0, 0), 2)
        
        # 텍스트 영역 그리기 (녹색)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for idx, (key, item) in enumerate(text_results.items()):
            bbox = item['bbox']
            text = item.get('text', '')
            minr, minc, maxr, maxc = bbox
            
            # 텍스트 영역 표시
            cv2.rectangle(result_image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            
            # 텍스트 내용 표시
            cv2.putText(result_image, f"{idx}: {text[:20]}", 
                       (minc, minr-10), font, 0.5, (0, 255, 0), 1)
        
        # 결과 저장
        if output_path:
            cv2.imwrite(output_path, result_image)
        
        return result_image


# 사용 예시:
if __name__ == "__main__":
    # 모델 경로 및 API 키 설정
    model_path = "kitsumed/yolov8m_seg-speech-bubble"  # Hugging Face 모델
    azure_api_key = "your_azure_api_key"
    azure_endpoint = "your_azure_endpoint"
    
    # 이미지 경로 설정
    image_path = "path_to_webtoon_image.jpg"
    output_path = "result.jpg"
    
    # 모듈 초기화 및 처리
    processor = SpeechBubbleOCR(model_path, azure_api_key, azure_endpoint)
    
    # 말풍선 내 텍스트만 처리 (중첩 임계값을 0.7로 설정하여 더 엄격하게)
    text_results, clean_image = processor.process_image(image_path, overlap_threshold=0.7)
    
    # 결과 출력
    print(f"말풍선 내 텍스트 개수: {len(text_results)}")
    for idx, (key, item) in enumerate(text_results.items()):
        print(f"텍스트 {idx}: {item['text']}")
    
    # 결과 시각화
    result_image = processor.visualize_results(image_path, output_path)
    
    # 결과 이미지 및 클린 이미지 저장
    cv2.imwrite("clean_" + os.path.basename(image_path), clean_image)
    cv2.imwrite(output_path, result_image)
    
    print(f"처리 완료: {output_path}") 