import cv2
import numpy as np
from skimage import measure, filters

class TextDetector:
    def __init__(self):
        """텍스트 감지 모듈 초기화"""
        self.min_area = 15  # 텍스트 영역으로 인식할 최소 영역 크기 (감소)
        self.max_area = 15000  # 텍스트 영역으로 인식할 최대 영역 크기 (증가)
        self.aspect_ratio_threshold = 15.0  # 가로세로 비율 임계값 (증가)
        self.solidity_threshold = 0.1  # 솔리디티 임계값 (영역의 밀집도)
    
    def preprocess_image(self, image):
        """이미지 전처리 (개선된 버전)"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 가우시안 블러 적용 (약하게)
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 다양한 이진화 방법 시도
        # 1. 적응형 이진화
        binary_adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 2. Otsu 이진화
        _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 두 이진화 결과 결합
        binary = cv2.bitwise_or(binary_adaptive, binary_otsu)
        
        # 노이즈 제거
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 텍스트 영역 확장 (문자 연결)
        dilated = cv2.dilate(opening, kernel, iterations=2)
        
        return dilated
    
    def is_inside_speech_bubble(self, image, region, threshold=200):
        """영역이 말풍선 내부에 있는지 확인 (흰색 배경 기준)"""
        minr, minc, maxr, maxc = region
        h, w = image.shape[:2]
        
        # 영역 주변을 검사하기 위한 확장 영역
        ext_minr = max(0, minr - 5)
        ext_maxr = min(h, maxr + 5)
        ext_minc = max(0, minc - 5)
        ext_maxc = min(w, maxc + 5)
        
        # 확장 영역 추출
        roi = image[ext_minr:ext_maxr, ext_minc:ext_maxc]
        
        if roi.size == 0:
            return False
            
        # 그레이스케일 변환
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
            
        # 평균 밝기 계산
        avg_brightness = np.mean(gray)
        
        # 밝기가 임계값보다 높으면 말풍선 내부로 간주
        return avg_brightness > threshold
    
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
                
                # 솔리디티 계산 (영역의 밀집도)
                solidity = region.area / region.convex_area if region.convex_area > 0 else 0
                
                # 필터링 조건 적용
                if (aspect_ratio < self.aspect_ratio_threshold and 
                    solidity > self.solidity_threshold and
                    width > 5 and height > 5):  # 최소 크기 조건 추가
                    
                    # 말풍선 내부 여부 확인 (옵션)
                    # 말풍선 내부 확인을 비활성화하거나 말풍선 내부인 경우만 추가
                    #if self.is_inside_speech_bubble(image, region.bbox):
                    text_regions.append(region.bbox)
        
        # 인접한 텍스트 영역 병합
        merged_regions = self.merge_text_regions(text_regions)
        
        # 크기 기준 정렬 (큰 영역부터)
        merged_regions.sort(key=lambda box: (box[2]-box[0])*(box[3]-box[1]), reverse=True)
        
        return merged_regions
    
    def merge_text_regions(self, regions, distance_threshold=15):
        """인접한 텍스트 영역을 병합합니다. (개선된 버전)"""
        if not regions:
            return []
        
        # 초기 병합 결과
        merged = [regions[0]]
        
        for current in regions[1:]:
            merged_with_existing = False
            
            # 현재 영역의 좌표
            curr_minr, curr_minc, curr_maxr, curr_maxc = current
            curr_height = curr_maxr - curr_minr
            curr_width = curr_maxc - curr_minc
            
            # 병합된 영역 목록을 순회하며 병합 가능 여부 확인
            for i, existing in enumerate(merged):
                exist_minr, exist_minc, exist_maxr, exist_maxc = existing
                exist_height = exist_maxr - exist_minr
                exist_width = exist_maxc - exist_minc
                
                # 두 영역 사이의 수직/수평 거리 계산
                vertical_dist = min(abs(curr_minr - exist_maxr), abs(curr_maxr - exist_minr))
                horizontal_dist = min(abs(curr_minc - exist_maxc), abs(curr_maxc - exist_minc))
                
                # 수직 방향으로 정렬된 텍스트인 경우 수직 거리 임계값 증가
                v_threshold = distance_threshold
                h_threshold = distance_threshold
                
                # 수직 정렬 텍스트 감지 (비슷한 너비, 수직으로 가까운 경우)
                if (abs(curr_width - exist_width) < 0.5 * max(curr_width, exist_width) and
                    horizontal_dist < 0.5 * max(curr_width, exist_width)):
                    v_threshold = 2 * distance_threshold
                
                # 수평 정렬 텍스트 감지 (비슷한 높이, 수평으로 가까운 경우)
                if (abs(curr_height - exist_height) < 0.5 * max(curr_height, exist_height) and
                    vertical_dist < 0.5 * max(curr_height, exist_height)):
                    h_threshold = 2 * distance_threshold
                
                # 두 영역이 충분히 가까우면 병합
                if vertical_dist <= v_threshold and horizontal_dist <= h_threshold:
                    # 새로운 병합된 바운딩 박스 계산
                    new_minr = min(exist_minr, curr_minr)
                    new_minc = min(exist_minc, curr_minc)
                    new_maxr = max(exist_maxr, curr_maxr)
                    new_maxc = max(exist_maxc, curr_maxc)
                    
                    # 새로운 병합 영역의 크기가 너무 크지 않은지 확인
                    new_width = new_maxc - new_minc
                    new_height = new_maxr - new_minr
                    new_aspect_ratio = max(new_width, new_height) / min(new_width, new_height)
                    
                    # 병합 후 영역이 너무 길쭉하지 않으면 병합
                    if new_aspect_ratio < 1.5 * self.aspect_ratio_threshold:
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
            # 크기에 따라 색상 변경 (큰 영역은 빨간색, 작은 영역은 녹색)
            area = (maxr - minr) * (maxc - minc)
            if area > 5000:
                color = (0, 0, 255)  # 빨간색 (큰 영역)
            elif area > 1000:
                color = (0, 255, 255)  # 노란색 (중간 영역)
            else:
                color = (0, 255, 0)  # 녹색 (작은 영역)
                
            cv2.rectangle(result, (minc, minr), (maxc, maxr), color, 2)
            cv2.putText(result, str(i), (minc, minr-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return result 