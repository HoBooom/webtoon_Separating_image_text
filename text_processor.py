import cv2
import numpy as np
import re
from paddleocr import PaddleOCR

class TextProcessor:
    def __init__(self, tesseract_cmd=None, lang='korean', confidence_threshold=0.4):
        """
        텍스트 처리 모듈 초기화
        
        Args:
            tesseract_cmd: 사용하지 않음 (Tesseract 호환성 유지)
            lang: 인식할 언어 (기본값: 한국어)
            confidence_threshold: 텍스트 감지 신뢰도 임계값 (0.0 ~ 1.0)
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
        
        # PaddleOCR 초기화 (더 높은 det_db_thresh로 텍스트 감지 정확도 향상)
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang=paddle_lang, 
            show_log=False,
            rec_batch_num=1,  # 배치 크기 조정
            rec_algorithm='SVTR_LCNet',  # 인식 알고리즘 변경 (더 정확한 버전)
            det_db_thresh=0.3,  # 텍스트 감지 임계값 조정
            det_db_box_thresh=0.5,  # 텍스트 박스 감지 임계값 조정
        )
        self.inpaint_radius = 5
        self.confidence_threshold = confidence_threshold
        
        # 만화/웹툰 특화 텍스트 후처리를 위한 정규식 패턴
        self.common_patterns = {
            r'[oO0]{3,}': '...',  # 연속된 o나 0을 말줄임표로 변환
            r'\.{2,}': '...',  # 연속된 점을 말줄임표로 정규화
            r'。{2,}': '...',  # 연속된 동그라미 마침표를 말줄임표로 변환
            r'\!{2,}': '!!',  # 연속된 느낌표 정규화
            r'\?{2,}': '??',  # 연속된 물음표 정규화
        }
    
    def enhance_for_text(self, image):
        """텍스트 가시성 향상을 위한 이미지 처리"""
        # 이미지 복사
        enhanced = image.copy()
        
        # 그레이스케일 변환
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced
        
        # 선명도 향상 (Unsharp Masking)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(sharpened)
        
        # 컬러 이미지로 변환 (필요한 경우)
        if len(image.shape) == 3:
            result = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2BGR)
        else:
            result = contrast_enhanced
            
        return result
    
    def preprocess_for_ocr(self, image):
        """OCR 성능 향상을 위한 이미지 전처리 (개선된 버전)
        
        Args:
            image: 원본 이미지
            
        Returns:
            전처리된 이미지
        """
        # 이미지 복사
        processed = image.copy()
        
        # 이미지 크기 확인 및 필요시 확대
        h, w = processed.shape[:2]
        min_size = 100  # 최소 크기 (픽셀)
        
        # 이미지가 너무 작으면 확대
        scale_factor = 1.0
        if h < min_size or w < min_size:
            scale_factor = max(min_size / h, min_size / w)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            processed = cv2.resize(processed, new_size, interpolation=cv2.INTER_CUBIC)
        
        # 그레이스케일 변환
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
            
        # 선명도 향상 (Unsharp Masking)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
        # 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # 적응형 이진화 시도
        binary_adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Otsu 이진화 시도
        _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 두 이진화 결과를 평균하여 사용
        binary = cv2.addWeighted(binary_adaptive, 0.5, binary_otsu, 0.5, 0)
        
        # 모폴로지 연산으로 텍스트 영역 향상
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 컬러 이미지로 변환 (OCR 입력용)
        if len(image.shape) == 3:
            processed_binary = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        else:
            processed_binary = morph
            
        return processed_binary, scale_factor
    
    def postprocess_text(self, text):
        """인식된 텍스트 후처리 (개선된 버전)
        
        Args:
            text: OCR로 인식된 텍스트
            
        Returns:
            후처리된 텍스트
        """
        if not text:
            return ""
        
        # 정규화된 텍스트 생성
        processed_text = text
        
        # 만화/웹툰 특화 패턴 처리
        for pattern, replacement in self.common_patterns.items():
            processed_text = re.sub(pattern, replacement, processed_text)
            
        # 특수문자 정리 (필요시 추가)
        # 한글, 영문, 숫자, 기본 구두점은 유지
        processed_text = re.sub(r'[^\w\s\.\,\!\?\(\)\[\]\{\}\:\;\-\'\"\—\…]', '', processed_text)
        
        # 불필요한 공백 정리
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        return processed_text
    
    def extract_text(self, image, region):
        """감지된 영역에서 텍스트를 추출합니다. (개선된 버전)
        
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
        
        # 텍스트 영역에 약간의 패딩 추가 (컨텍스트 확보)
        h, w = image.shape[:2]
        pad = 3
        padded_roi = image[
            max(0, minr-pad):min(h, maxr+pad),
            max(0, minc-pad):min(w, maxc+pad)
        ]
        
        # 세 가지 버전의 이미지 준비
        # 1. 원본 이미지
        original_roi = padded_roi.copy()
        
        # 2. 선명도가 향상된 이미지
        enhanced_roi = self.enhance_for_text(padded_roi)
            
        # 3. OCR용으로 전처리된 이미지
        processed_roi, scale_factor = self.preprocess_for_ocr(padded_roi)
        
        # 세 가지 이미지에서 OCR 시도
        all_texts = []
        confidences = []
        
        # 이미지 버전들로 OCR 시도
        for idx, img in enumerate([original_roi, enhanced_roi, processed_roi]):
            result = self.ocr.ocr(img, cls=True)
            
            if result and result[0]:
                for line in result[0]:
                    if line[1][0] and line[1][1] >= self.confidence_threshold:
                        all_texts.append(line[1][0])
                        confidences.append(line[1][1])
            
            # 텍스트가 발견되었다면 추가 처리 중단
            if all_texts:
                break
        
        # 결과가 없으면 빈 문자열 반환
        if not all_texts:
            return ""
        
        # 중복 제거 및 신뢰도 기반 정렬
        unique_texts = []
        for text, conf in sorted(zip(all_texts, confidences), key=lambda x: x[1], reverse=True):
            if text not in unique_texts:
                unique_texts.append(text)
        
        # 결과 조합
        combined_text = " ".join(unique_texts)
        
        # 텍스트 후처리
        processed_text = self.postprocess_text(combined_text)
        
        return processed_text.strip()
    
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