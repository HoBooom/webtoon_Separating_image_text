import cv2
import numpy as np
from paddleocr import PaddleOCR

class TextProcessor:
    def __init__(self, tesseract_cmd=None, lang='korean'):
        """
        텍스트 처리 모듈 초기화
        
        Args:
            tesseract_cmd: 사용하지 않음 (Tesseract 호환성 유지)
            lang: 인식할 언어 (기본값: 한국어)
        """
        # 언어 매핑 (Tesseract 언어 코드 -> PaddleOCR 언어 코드)
        lang_map = {
            'kor': 'korean',
            'kor,en': 'korean',
            'eng': 'en',
            'en': 'en',
            'jpn': 'japan',
        }
        
        # 언어 코드 변환
        paddle_lang = lang_map.get(lang, lang)
        
        # PaddleOCR 초기화
        self.ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
        self.inpaint_radius = 5
    
    def extract_text(self, image, region):
        """감지된 영역에서 텍스트를 추출합니다.
        
        Args:
            image: 원본 이미지
            region: 텍스트 영역 바운딩 박스 (min_row, min_col, max_row, max_col)
            
        Returns:
            추출된 텍스트 또는 빈 문자열
        """
        minr, minc, maxr, maxc = region
        
        # 텍스트 영역 추출
        roi = image[minr:maxr, minc:maxc]
        
        # ROI 크기가 너무 작으면 빈 문자열 반환
        if roi.shape[0] < 5 or roi.shape[1] < 5:
            return ""
        
        # PaddleOCR로 텍스트 인식
        result = self.ocr.ocr(roi, cls=True)
        
        # 결과 추출
        text = ""
        if result and result[0]:
            for line in result[0]:
                if line[1][0]:  # 텍스트와 신뢰도의 튜플
                    text += line[1][0] + " "
        
        return text.strip()
    
    def get_background_color(self, image, region, margin=5):
        """텍스트 영역 주변의 배경색을 추출합니다.
        
        Args:
            image: 원본 이미지
            region: 텍스트 영역 바운딩 박스 (min_row, min_col, max_row, max_col)
            margin: 영역 주변에서 배경색을 샘플링할 여백 픽셀 수
            
        Returns:
            배경색 (B, G, R)
        """
        minr, minc, maxr, maxc = region
        h, w = image.shape[:2]
        
        # 영역 주변의 픽셀 수집
        border_pixels = []
        
        # 위쪽 테두리
        top_minr = max(0, minr - margin)
        top_maxr = max(0, minr)
        if top_minr < top_maxr:
            top_border = image[top_minr:top_maxr, max(0, minc-margin):min(w, maxc+margin)]
            border_pixels.extend(top_border.reshape(-1, 3))
        
        # 아래쪽 테두리
        bottom_minr = min(h, maxr)
        bottom_maxr = min(h, maxr + margin)
        if bottom_minr < bottom_maxr:
            bottom_border = image[bottom_minr:bottom_maxr, max(0, minc-margin):min(w, maxc+margin)]
            border_pixels.extend(bottom_border.reshape(-1, 3))
        
        # 왼쪽 테두리
        left_minc = max(0, minc - margin)
        left_maxc = max(0, minc)
        if left_minc < left_maxc:
            left_border = image[max(0, minr-margin):min(h, maxr+margin), left_minc:left_maxc]
            border_pixels.extend(left_border.reshape(-1, 3))
        
        # 오른쪽 테두리
        right_minc = min(w, maxc)
        right_maxc = min(w, maxc + margin)
        if right_minc < right_maxc:
            right_border = image[max(0, minr-margin):min(h, maxr+margin), right_minc:right_maxc]
            border_pixels.extend(right_border.reshape(-1, 3))
        
        # 배경색 계산 (중간값 사용 - 더 견고함)
        if border_pixels:
            border_pixels = np.array(border_pixels)
            bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
            return tuple(map(int, bg_color))
        else:
            # 기본 흰색 반환
            return (255, 255, 255)
    
    def remove_text(self, image, region):
        """감지된 영역에서 텍스트를 제거하고 배경색으로 완전히 덮습니다.
        
        Args:
            image: 원본 이미지
            region: 텍스트 영역 바운딩 박스 (min_row, min_col, max_row, max_col)
            
        Returns:
            텍스트가 제거된 이미지
        """
        minr, minc, maxr, maxc = region
        result = image.copy()
        
        # 주변 배경색 가져오기
        bg_color = self.get_background_color(image, region)
        
        # 텍스트 영역을 배경색으로 채우기
        cv2.rectangle(result, (minc, minr), (maxc, maxr), bg_color, -1)
        
        # 경계를 부드럽게 하기 위한 가우시안 블러 적용
        # 작은 영역에만 블러 적용
        blur_margin = 3
        blur_minr = max(0, minr - blur_margin)
        blur_maxr = min(image.shape[0], maxr + blur_margin)
        blur_minc = max(0, minc - blur_margin)
        blur_maxc = min(image.shape[1], maxc + blur_margin)
        
        # 블러 영역이 충분히 크면 적용
        if (blur_maxr - blur_minr > 2*blur_margin) and (blur_maxc - blur_minc > 2*blur_margin):
            blur_region = result[blur_minr:blur_maxr, blur_minc:blur_maxc]
            result[blur_minr:blur_maxr, blur_minc:blur_maxc] = cv2.GaussianBlur(blur_region, (5, 5), 0)
        
        return result
    
    def process_all_regions(self, image, regions):
        """모든 텍스트 영역을 처리하고, 텍스트가 인식되지 않은 영역은 제외합니다.
        
        Args:
            image: 원본 이미지
            regions: 텍스트 영역 바운딩 박스 목록
            
        Returns:
            텍스트 추출 결과와 텍스트가 제거된 이미지
        """
        results = {}
        clean_image = image.copy()
        valid_count = 0
        
        for i, region in enumerate(regions):
            # 텍스트 추출
            text = self.extract_text(image, region)
            
            # 텍스트가 있는 경우에만 처리
            if text.strip():
                # 결과 저장
                results[valid_count] = {
                    'bbox': region,
                    'text': text
                }
                
                # 텍스트 제거
                clean_image = self.remove_text(clean_image, region)
                valid_count += 1
        
        return results, clean_image 