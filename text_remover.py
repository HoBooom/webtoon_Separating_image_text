import cv2
import numpy as np

class TextRemover:
    def __init__(self):
        """텍스트 제거 모듈 초기화"""
        self.inpaint_radius = 3
        self.margin = 5  # 텍스트 주변 여백
    
    def remove_text(self, image, bubble):
        """말풍선 내의 텍스트를 제거합니다.
        
        Args:
            image: 원본 이미지
            bubble: 말풍선 바운딩 박스 (min_row, min_col, max_row, max_col)
            
        Returns:
            텍스트가 제거된 이미지
        """
        minr, minc, maxr, maxc = bubble
        result = image.copy()
        
        # 말풍선 영역 추출
        roi = image[minr:maxr, minc:maxc]
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 이진화 (텍스트는 주로 어두운 부분)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 텍스트 영역 확장 (여백 추가)
        binary = cv2.dilate(binary, kernel, iterations=self.margin)
        
        # 인페인팅 마스크 생성
        mask = np.zeros_like(image)
        mask[minr:maxr, minc:maxc] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 인페인팅 (텍스트 제거)
        result = cv2.inpaint(result, mask[:,:,0], self.inpaint_radius, cv2.INPAINT_TELEA)
        
        return result
    
    def remove_all_texts(self, image, bubbles):
        """모든 말풍선의 텍스트를 제거합니다.
        
        Args:
            image: 원본 이미지
            bubbles: 말풍선 바운딩 박스 목록
            
        Returns:
            모든 텍스트가 제거된 이미지
        """
        result = image.copy()
        
        for bubble in bubbles:
            result = self.remove_text(result, bubble)
            
        return result 