import cv2
import pytesseract
import numpy as np

class TextExtractor:
    def __init__(self, tesseract_cmd=None, lang='kor+eng'):
        """
        OCR을 위한 초기화
        
        Args:
            tesseract_cmd: Tesseract 실행 파일 경로 (Windows에서 필요)
            lang: 인식할 언어 (기본값: 한국어+영어)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang
    
    def extract_text(self, image, bubble):
        """말풍선 영역에서 텍스트를 추출합니다.
        
        Args:
            image: 원본 이미지
            bubble: 말풍선 바운딩 박스 (min_row, min_col, max_row, max_col)
            
        Returns:
            추출된 텍스트
        """
        minr, minc, maxr, maxc = bubble
        
        # 말풍선 영역 추출
        roi = image[minr:maxr, minc:maxc]
        
        # 전처리: 이미지 향상
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 이진화
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR 실행
        custom_config = f'--oem 3 --psm 6 -l {self.lang}'
        text = pytesseract.image_to_string(binary, config=custom_config)
        
        return text.strip()
    
    def extract_all_texts(self, image, bubbles):
        """이미지의 모든 말풍선에서 텍스트를 추출합니다.
        
        Args:
            image: 원본 이미지
            bubbles: 말풍선 바운딩 박스 목록
            
        Returns:
            말풍선과 텍스트 쌍의 딕셔너리
        """
        results = {}
        
        for i, bubble in enumerate(bubbles):
            text = self.extract_text(image, bubble)
            results[i] = {
                'bbox': bubble,
                'text': text
            }
            
        return results 