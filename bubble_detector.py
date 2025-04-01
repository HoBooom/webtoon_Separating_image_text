import cv2
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu

class BubbleDetector:
    def __init__(self):
        self.min_area = 1000  # 말풍선으로 인식할 최소 영역 크기
        self.max_area = 150000  # 말풍선으로 인식할 최대 영역 크기
        self.white_threshold = 200  # 말풍선 내부 흰색 영역 임계값
        self.aspect_ratio_threshold = 5.0  # 가로세로 비율 임계값
    
    def detect_bubbles(self, image):
        """이미지에서 말풍선 영역을 감지합니다."""
        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 이진화
        try:
            thresh = threshold_otsu(gray)
            binary = gray > thresh
            binary = binary.astype(np.uint8) * 255
        except:
            # Otsu 이진화가 실패하면 일반적인 임계값 사용
            _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 연결된 영역 찾기
        labels = measure.label(binary)
        
        # 영역 속성 얻기
        regions = measure.regionprops(labels)
        
        # 말풍선 후보들 저장
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
                
                # 가로세로 비율이 임계값 미만이고 내부가 주로 흰색인 영역 선택
                if aspect_ratio < self.aspect_ratio_threshold:
                    roi = gray[minr:maxr, minc:maxc]
                    if np.mean(roi) > self.white_threshold:
                        bubbles.append(region.bbox)
        
        return bubbles
    
    def visualize_bubbles(self, image, bubbles):
        """감지된 말풍선을 시각화합니다."""
        result = image.copy()
        for bubble in bubbles:
            minr, minc, maxr, maxc = bubble
            cv2.rectangle(result, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
        return result 