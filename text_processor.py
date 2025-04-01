import cv2
import numpy as np
import pytesseract

class TextProcessor:
    def __init__(self, tesseract_cmd=None, lang='kor+eng'):
        """
        텍스트 처리 모듈 초기화
        
        Args:
            tesseract_cmd: Tesseract 실행 파일 경로 (Windows에서 필요)
            lang: 인식할 언어 (기본값: 한국어+영어)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang
        self.inpaint_radius = 5
    
    def extract_text(self, image, region):
        """감지된 영역에서 텍스트를 추출합니다.
        
        Args:
            image: 원본 이미지
            region: 텍스트 영역 바운딩 박스 (min_row, min_col, max_row, max_col)
            
        Returns:
            추출된 텍스트
        """
        minr, minc, maxr, maxc = region
        
        # 텍스트 영역 추출
        roi = image[minr:maxr, minc:maxc]
        
        # 전처리: 이미지 향상
        if len(roi.shape) == 3:  # 컬러 이미지인 경우
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:  # 이미 그레이스케일인 경우
            gray = roi
        
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
    
    def remove_text(self, image, region):
        """감지된 영역에서 텍스트를 제거합니다.
        
        Args:
            image: 원본 이미지
            region: 텍스트 영역 바운딩 박스 (min_row, min_col, max_row, max_col)
            
        Returns:
            텍스트가 제거된 이미지
        """
        minr, minc, maxr, maxc = region
        result = image.copy()
        
        # 텍스트 영역 추출
        roi = image[minr:maxr, minc:maxc]
        
        # 그레이스케일 변환
        if len(roi.shape) == 3:  # 컬러 이미지인 경우
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:  # 이미 그레이스케일인 경우
            gray = roi
        
        # 이진화 (텍스트는 주로 어두운 부분)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 인페인팅 마스크 생성
        mask = np.zeros_like(image)
        if len(mask.shape) == 3:  # 컬러 이미지인 경우
            mask[minr:maxr, minc:maxc] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            mask_channel = mask[:,:,0]
        else:  # 그레이스케일 이미지인 경우
            mask[minr:maxr, minc:maxc] = binary
            mask_channel = mask
        
        # 인페인팅 (텍스트 제거)
        result = cv2.inpaint(result, mask_channel, self.inpaint_radius, cv2.INPAINT_TELEA)
        
        return result
    
    def process_all_regions(self, image, regions):
        """모든 텍스트 영역을 처리합니다.
        
        Args:
            image: 원본 이미지
            regions: 텍스트 영역 바운딩 박스 목록
            
        Returns:
            텍스트 추출 결과와 텍스트가 제거된 이미지
        """
        results = {}
        clean_image = image.copy()
        
        for i, region in enumerate(regions):
            # 텍스트 추출
            text = self.extract_text(image, region)
            
            # 결과 저장
            results[i] = {
                'bbox': region,
                'text': text
            }
            
            # 텍스트 제거
            clean_image = self.remove_text(clean_image, region)
        
        return results, clean_image 