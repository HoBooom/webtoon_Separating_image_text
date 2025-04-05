import cv2
import numpy as np
from skimage import measure

class TextDetector:
    def __init__(self):
        """텍스트 감지 모듈 초기화"""
        self.min_area = 20  # 텍스트 영역으로 인식할 최소 영역 크기
        self.max_area = 10000  # 텍스트 영역으로 인식할 최대 영역 크기
        self.aspect_ratio_threshold = 10.0  # 가로세로 비율 임계값
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러 적용
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 적응형 이진화 (텍스트는 주로 어두운 부분)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 텍스트 영역 확장 (문자 연결)
        dilated = cv2.dilate(opening, kernel, iterations=3)
        
        return dilated
    
    def detect_text_regions(self, image):
        """이미지에서 텍스트 영역을 감지합니다."""
        # 이미지 전처리
        binary = self.preprocess_image(image)
        
        # 연결된 영역 찾기
        labels = measure.label(binary)
        
        # 영역 속성 얻기
        regions = measure.regionprops(labels)
        
        # 텍스트 영역 저장
        text_regions = []
        
        for region in regions:
            # 영역 크기 필터링
            if self.min_area < region.area < self.max_area:
                # 바운딩 박스 (min_row, min_col, max_row, max_col)
                minr, minc, maxr, maxc = region.bbox
                
                # 가로세로 비율 계산
                width = maxc - minc
                height = maxr - minr
                aspect_ratio = max(width, height) / min(width, height)
                
                # 가로세로 비율이 임계값 미만인 영역 선택
                if aspect_ratio < self.aspect_ratio_threshold:
                    text_regions.append(region.bbox)
        
        # 인접한 텍스트 영역 병합
        merged_regions = self.merge_text_regions(text_regions)
        
        return merged_regions
    
    def merge_text_regions(self, regions, distance_threshold=20):
        """인접한 텍스트 영역을 병합합니다."""
        if not regions:
            return []
        
        # 초기 병합 결과
        merged = [regions[0]]
        
        for current in regions[1:]:
            merged_with_existing = False
            
            # 현재 영역의 좌표
            curr_minr, curr_minc, curr_maxr, curr_maxc = current
            
            # 병합된 영역 목록을 순회하며 병합 가능 여부 확인
            for i, existing in enumerate(merged):
                exist_minr, exist_minc, exist_maxr, exist_maxc = existing
                
                # 두 영역 사이의 수직/수평 거리 계산
                vertical_dist = min(abs(curr_minr - exist_maxr), abs(curr_maxr - exist_minr))
                horizontal_dist = min(abs(curr_minc - exist_maxc), abs(curr_maxc - exist_minc))
                
                # 두 영역이 충분히 가까우면 병합
                if vertical_dist <= distance_threshold and horizontal_dist <= distance_threshold:
                    # 새로운 병합된 바운딩 박스 계산
                    new_minr = min(exist_minr, curr_minr)
                    new_minc = min(exist_minc, curr_minc)
                    new_maxr = max(exist_maxr, curr_maxr)
                    new_maxc = max(exist_maxc, curr_maxc)
                    
                    # 병합된 영역 업데이트
                    merged[i] = (new_minr, new_minc, new_maxr, new_maxc)
                    merged_with_existing = True
                    break
            
            # 기존 영역과 병합되지 않았으면 새 영역으로 추가
            if not merged_with_existing:
                merged.append(current)
        
        return merged
    
    def visualize_text_regions(self, image, regions):
        """감지된 텍스트 영역을 시각화합니다."""
        result = image.copy()
        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region
            cv2.rectangle(result, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            cv2.putText(result, str(i), (minc, minr-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return result 