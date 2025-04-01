import cv2
import numpy as np
from skimage import measure, morphology
from skimage.filters import threshold_otsu

class AdvancedBubbleDetector:
    def __init__(self):
        self.min_area = 800  # 말풍선으로 인식할 최소 영역 크기
        self.max_area = 200000  # 말풍선으로 인식할 최대 영역 크기
        self.white_threshold = 190  # 말풍선 내부 흰색 영역 임계값
        self.aspect_ratio_threshold = 5.0  # 가로세로 비율 임계값
        self.solidity_threshold = 0.7  # 영역의 솔리디티 임계값 (말풍선의 형태 특성)
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 적응형 이진화 (대비 향상 후)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        
        # 이진화
        try:
            thresh = threshold_otsu(enhanced)
            binary = enhanced > thresh
        except:
            # Otsu 이진화가 실패하면 일반적인 임계값 사용
            _, binary = cv2.threshold(enhanced, 230, 255, cv2.THRESH_BINARY)
        
        return binary.astype(np.uint8) * 255
    
    def filter_regions(self, regions, gray):
        """지정된 기준에 따라 말풍선 후보 영역을 필터링합니다."""
        bubbles = []
        
        for region in regions:
            # 영역 크기 필터링
            if self.min_area < region.area < self.max_area:
                # 바운딩 박스 (min_row, min_col, max_row, max_col)
                minr, minc, maxr, maxc = region.bbox
                
                # 가로세로 비율 계산
                width = maxc - minc
                height = maxr - minr
                aspect_ratio = max(width, height) / min(width, height)
                
                # 솔리디티(형태의 밀집도) 계산
                solidity = region.area / region.convex_area
                
                # 영역 내부 평균 밝기 계산
                roi = gray[minr:maxr, minc:maxc]
                avg_intensity = np.mean(roi)
                
                # 필터링 조건 적용
                if (aspect_ratio < self.aspect_ratio_threshold and 
                    solidity > self.solidity_threshold and 
                    avg_intensity > self.white_threshold):
                    
                    # 말풍선 후보에 추가
                    bubbles.append(region.bbox)
        
        return bubbles
    
    def merge_overlapping_bubbles(self, bubbles):
        """겹치는 말풍선 영역을 병합합니다."""
        if not bubbles:
            return []
        
        # 바운딩 박스 좌표 변환 (min_row, min_col, max_row, max_col) -> (x, y, w, h)
        rects = []
        for bubble in bubbles:
            minr, minc, maxr, maxc = bubble
            rects.append([minc, minr, maxc - minc, maxr - minr])
        
        # 겹치는 영역 병합
        merged_rects = []
        suppressed = np.zeros(len(rects), dtype=np.int32)
        
        for i in range(len(rects)):
            if suppressed[i] == 1:
                continue
            
            r1 = rects[i]
            for j in range(i + 1, len(rects)):
                if suppressed[j] == 1:
                    continue
                
                r2 = rects[j]
                
                # 겹침 여부 확인
                x1 = max(r1[0], r2[0])
                y1 = max(r1[1], r2[1])
                x2 = min(r1[0] + r1[2], r2[0] + r2[2])
                y2 = min(r1[1] + r1[3], r2[1] + r2[3])
                
                if x1 < x2 and y1 < y2:
                    overlap_area = (x2 - x1) * (y2 - y1)
                    r1_area = r1[2] * r1[3]
                    r2_area = r2[2] * r2[3]
                    
                    # 영역의 20% 이상이 겹치면 병합
                    if overlap_area > 0.2 * min(r1_area, r2_area):
                        x = min(r1[0], r2[0])
                        y = min(r1[1], r2[1])
                        w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
                        h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
                        
                        r1 = [x, y, w, h]
                        suppressed[j] = 1
            
            # (x, y, w, h) -> (min_row, min_col, max_row, max_col)
            merged_rects.append((r1[1], r1[0], r1[1] + r1[3], r1[0] + r1[2]))
        
        return merged_rects
    
    def detect_bubbles(self, image):
        """이미지에서 말풍선 영역을 감지합니다."""
        # 이미지 전처리
        binary = self.preprocess_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 모폴로지 연산으로 노이즈 제거 및 말풍선 영역 강화
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 작은 구멍 채우기
        binary = morphology.remove_small_holes(binary.astype(bool), area_threshold=100)
        binary = binary.astype(np.uint8) * 255
        
        # 연결된 영역 찾기
        labels = measure.label(binary)
        
        # 영역 속성 얻기
        regions = measure.regionprops(labels)
        
        # 말풍선 후보 필터링
        bubbles = self.filter_regions(regions, gray)
        
        # 겹치는 말풍선 병합
        merged_bubbles = self.merge_overlapping_bubbles(bubbles)
        
        return merged_bubbles
    
    def visualize_bubbles(self, image, bubbles):
        """감지된 말풍선을 시각화합니다."""
        result = image.copy()
        for i, bubble in enumerate(bubbles):
            minr, minc, maxr, maxc = bubble
            cv2.rectangle(result, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            cv2.putText(result, str(i), (minc, minr-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return result 