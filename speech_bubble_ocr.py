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
        
        # 텍스트 박스 병합 설정
        self.merge_boxes = True  # 기본적으로 병합 활성화
        self.merge_distance_threshold = 20  # 텍스트 박스 간 최대 병합 거리 (픽셀)
        self.merge_any_overlap = True  # 겹치는 부분이 있으면 모두 병합
    
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
    
    def has_overlap(self, box1, box2):
        """
        두 박스 간에 겹치는 부분이 있는지 확인합니다.
        
        Args:
            box1: 첫 번째 박스 [minr, minc, maxr, maxc]
            box2: 두 번째 박스 [minr, minc, maxr, maxc]
            
        Returns:
            겹치는 부분이 있으면 True, 없으면 False
        """
        box1_minr, box1_minc, box1_maxr, box1_maxc = box1
        box2_minr, box2_minc, box2_maxr, box2_maxc = box2
        
        # 두 박스가 겹치지 않는 경우
        if (box1_maxr <= box2_minr or  # box1이 box2의 위쪽에 있음
            box1_minr >= box2_maxr or  # box1이 box2의 아래쪽에 있음
            box1_maxc <= box2_minc or  # box1이 box2의 왼쪽에 있음
            box1_minc >= box2_maxc):   # box1이 box2의 오른쪽에 있음
            return False
        
        return True
    
    def should_merge_boxes(self, box1, box2, distance_threshold):
        """
        두 박스를 병합해야 하는지 결정합니다.
        
        Args:
            box1: 첫 번째 박스 [minr, minc, maxr, maxc]
            box2: 두 번째 박스 [minr, minc, maxr, maxc]
            distance_threshold: 병합 거리 임계값
            
        Returns:
            병합해야 하면 True, 아니면 False
        """
        # 겹치는 부분이 있는 경우 바로 병합
        if self.merge_any_overlap and self.has_overlap(box1, box2):
            return True
            
        # 거리 기반 병합 로직 (이전과 동일)
        box1_minr, box1_minc, box1_maxr, box1_maxc = box1
        box2_minr, box2_minc, box2_maxr, box2_maxc = box2
        
        # 두 박스 간의 거리 계산 (수직, 수평 거리)
        horizontal_distance = min(
            abs(box1_minc - box2_maxc),  # box1 왼쪽 - box2 오른쪽
            abs(box1_maxc - box2_minc)   # box1 오른쪽 - box2 왼쪽
        )
        vertical_distance = min(
            abs(box1_minr - box2_maxr),  # box1 위 - box2 아래
            abs(box1_maxr - box2_minr)   # box1 아래 - box2 위
        )
        
        # 수직으로 겹치는 경우 수평 거리만 고려
        if (box1_minr <= box2_maxr and box1_maxr >= box2_minr):
            return horizontal_distance <= distance_threshold
        # 수평으로 겹치는 경우 수직 거리만 고려
        elif (box1_minc <= box2_maxc and box1_maxc >= box2_minc):
            return vertical_distance <= distance_threshold
        # 둘 다 겹치지 않는 경우 대각선 거리 계산
        else:
            diagonal_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
            return diagonal_distance <= distance_threshold
    
    def merge_text_boxes(self, text_results, speech_bubbles, distance_threshold=None):
        """
        텍스트 박스들을 병합합니다. 단, 같은 말풍선 내에 있는 텍스트 박스들만 병합합니다.
        
        Args:
            text_results: OCR 결과 사전
            speech_bubbles: 감지된 말풍선 영역 목록
            distance_threshold: 텍스트 박스 간 최대 병합 거리 (픽셀)
            
        Returns:
            병합된 텍스트 결과 사전
        """
        if not self.merge_boxes:
            return text_results
        
        if distance_threshold is None:
            distance_threshold = self.merge_distance_threshold
        
        # 텍스트 박스와 관련 정보를 리스트로 변환
        boxes = []
        for key, item in text_results.items():
            if isinstance(item, dict) and 'bbox' in item:
                minr, minc, maxr, maxc = item['bbox']
                boxes.append({
                    'key': key,
                    'bbox': [minr, minc, maxr, maxc],
                    'text': item.get('text', ''),
                    'confidence': item.get('confidence', 0),
                    'merged': False,  # 병합 여부 표시
                    'cluster_id': -1,  # 클러스터 ID 초기화
                    'bubble_id': -1  # 속한 말풍선 ID
                })
        
        # 각 텍스트 박스가 어떤 말풍선에 속하는지 식별
        for i, box in enumerate(boxes):
            for j, bubble in enumerate(speech_bubbles):
                if self.is_text_in_speech_bubble(box['bbox'], [bubble]):
                    boxes[i]['bubble_id'] = j
                    break
        
        # 클러스터링 알고리즘을 사용하여 병합할 박스 그룹 식별 (같은 말풍선에 있는 박스들만)
        cluster_id = 0
        
        # 첫 번째 단계: 같은 말풍선 내에 있는 박스들 간의 연결 관계 확인
        for i in range(len(boxes)):
            if boxes[i]['cluster_id'] == -1:
                boxes[i]['cluster_id'] = cluster_id
                
                # 이 박스와 연결된 다른 모든 박스들을 찾기 위한 BFS
                queue = [i]
                while queue:
                    current_idx = queue.pop(0)
                    current_box = boxes[current_idx]
                    current_bubble_id = current_box['bubble_id']
                    
                    # 같은 말풍선에 속하지 않는 텍스트 박스는 병합하지 않음
                    if current_bubble_id == -1:
                        continue
                    
                    for j in range(len(boxes)):
                        # 같은 말풍선에 속하는지 확인
                        if boxes[j]['bubble_id'] != current_bubble_id:
                            continue
                            
                        # 아직 클러스터에 할당되지 않았거나, 다른 클러스터에 속한 박스
                        if boxes[j]['cluster_id'] == -1 or boxes[j]['cluster_id'] != current_box['cluster_id']:
                            # 두 박스가 병합 조건을 만족하는지 확인
                            if self.should_merge_boxes(current_box['bbox'], boxes[j]['bbox'], distance_threshold):
                                # 다른 클러스터에 이미 속해 있으면 두 클러스터를 병합
                                if boxes[j]['cluster_id'] != -1:
                                    old_cluster_id = boxes[j]['cluster_id']
                                    for k in range(len(boxes)):
                                        if boxes[k]['cluster_id'] == old_cluster_id:
                                            boxes[k]['cluster_id'] = current_box['cluster_id']
                                else:
                                    # 클러스터에 할당되지 않은 경우 현재 클러스터에 추가
                                    boxes[j]['cluster_id'] = current_box['cluster_id']
                                    queue.append(j)
                
                cluster_id += 1
        
        # 두 번째 단계: 클러스터별로 박스 병합
        clusters = {}
        for i, box in enumerate(boxes):
            cluster = box['cluster_id']
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(box)
        
        # 각 클러스터에서 모든 박스를 병합
        merged_boxes = []
        for cluster_boxes in clusters.values():
            if not cluster_boxes:
                continue
                
            # 클러스터의 모든 박스를 포함하는 최대 경계 계산
            min_r = min(box['bbox'][0] for box in cluster_boxes)
            min_c = min(box['bbox'][1] for box in cluster_boxes)
            max_r = max(box['bbox'][2] for box in cluster_boxes)
            max_c = max(box['bbox'][3] for box in cluster_boxes)
            
            # 모든 텍스트 결합 (줄바꿈 유지)
            # 각 box['text']는 Azure에서 이미 한 줄 단위로 인식된 텍스트이므로, 
            # 이를 \n으로 연결하면 원래의 줄바꿈을 유지할 수 있습니다.
            merged_text = "\n".join(box['text'] for box in cluster_boxes)
            
            # 신뢰도 평균 계산
            total_confidence = sum(box['confidence'] for box in cluster_boxes)
            avg_confidence = total_confidence / len(cluster_boxes) if cluster_boxes else 0
            
            # 병합된 박스 저장
            merged_boxes.append({
                'bbox': [min_r, min_c, max_r, max_c],
                'text': merged_text,
                'confidence': avg_confidence
            })
        
        # 새로운 결과 사전 생성
        merged_results = {}
        for i, box in enumerate(merged_boxes):
            merged_results[i] = box
        
        return merged_results
    
    def process_image(self, image_path, overlap_threshold=None, merge_distance=None):
        """
        이미지에서 말풍선을 감지하고 그 내부 텍스트만 추출합니다.
        
        Args:
            image_path: 이미지 경로
            overlap_threshold: 텍스트가 말풍선 내에 있다고 판단할 최소 중첩 비율
            merge_distance: 텍스트 박스 병합 거리 임계값
            
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
        
        # 4. 같은 말풍선 내 가까운 텍스트 박스 병합 (옵션)
        if self.merge_boxes:
            if merge_distance is not None:
                self.merge_distance_threshold = merge_distance
            filtered_results = self.merge_text_boxes(filtered_results, speech_bubbles)
        
        # 5. 말풍선 내 텍스트만 삭제한 이미지 생성
        clean_image = self.ocr_processor.remove_all_text(image, filtered_results)
        
        return filtered_results, clean_image
    
    def visualize_results(self, image_path, output_path=None, show_merged=True):
        """
        결과를 시각화하여 표시합니다.
        
        Args:
            image_path: 원본 이미지 경로
            output_path: 결과 이미지 저장 경로 (없으면 저장하지 않음)
            show_merged: 병합된 텍스트 박스 표시 여부
            
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
        if show_merged:
            # 텍스트 박스 병합 포함된 결과
            text_results, _ = self.process_image(image_path)
        else:
            # 병합 없이 텍스트 감지만 수행
            temp_merge_setting = self.merge_boxes
            self.merge_boxes = False
            text_results, _ = self.process_image(image_path)
            self.merge_boxes = temp_merge_setting
        
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
            confidence = item.get('confidence', 0)
            minr, minc, maxr, maxc = bbox
            
            # 텍스트 영역 표시
            cv2.rectangle(result_image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            
            # 텍스트 내용 표시 (길이 제한)
            display_text = text[:20] + "..." if len(text) > 20 else text
            cv2.putText(result_image, f"{idx}: {display_text}", 
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
    
    # 텍스트 박스 병합 설정
    processor.merge_boxes = True
    processor.merge_distance_threshold = 15  # 병합 거리 임계값 (픽셀)
    processor.merge_any_overlap = True  # 겹치는 부분이 있으면 모두 병합
    
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