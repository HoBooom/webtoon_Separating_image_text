import os
import cv2
import numpy as np
import io
import requests
import time
from PIL import Image

class AzureTextProcessor:
    def __init__(self, api_key, endpoint, lang='ko'):
        """
        Azure Computer Vision OCR 기반 텍스트 처리 모듈 초기화
        
        Args:
            api_key: Azure Computer Vision API 키
            endpoint: Azure Computer Vision API 엔드포인트
            lang: 인식할 언어 (기본값: 한국어)
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.lang = lang
        self.ocr_url = f"{endpoint}/vision/v3.2/read/analyze"
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key,
            'Content-Type': 'application/octet-stream'
        }
        self.inpaint_radius = 5

    def extract_text_from_region(self, image, region):
        """
        이미지의 특정 영역에서 텍스트를 추출합니다.
        
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
        
        # OpenCV 이미지를 PIL 이미지로 변환
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        
        # 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        # Azure OCR API 호출
        try:
            response = requests.post(
                self.ocr_url,
                headers=self.headers,
                params={'language': self.lang, 'readingOrder': 'natural'},
                data=img_bytes
            )
            response.raise_for_status()
            
            # 작업 상태 폴링을 위한 URL 받기
            operation_url = response.headers["Operation-Location"]
            
            # 분석 결과 기다리기
            analysis = {}
            poll = True
            while poll:
                response_final = requests.get(
                    operation_url,
                    headers={"Ocp-Apim-Subscription-Key": self.api_key}
                )
                analysis = response_final.json()
                
                if "status" in analysis and analysis["status"] == "succeeded":
                    poll = False
                elif "status" in analysis and analysis["status"] == "failed":
                    return ""
                else:
                    time.sleep(1)
            
            # 텍스트 추출
            text = ""
            if "analyzeResult" in analysis and "readResults" in analysis["analyzeResult"]:
                for read_result in analysis["analyzeResult"]["readResults"]:
                    for line in read_result["lines"]:
                        text += line["text"] + " "
            
            return text.strip()
            
        except Exception as e:
            print(f"Error in Azure OCR: {e}")
            return ""

    def get_background_color(self, image, region, margin=5):
        """
        텍스트 영역 주변의 배경색을 추출합니다.
        
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
        """
        감지된 영역에서 텍스트를 제거하고 배경색으로 완전히 덮습니다.
        
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

    def process_image(self, image, regions):
        """
        이미지의 모든 텍스트 영역을 처리합니다.
        
        Args:
            image: 원본 이미지
            regions: 텍스트 영역 바운딩 박스 목록
            
        Returns:
            텍스트 추출 결과와 텍스트가 제거된 이미지
        """
        results = {}
        clean_image = image.copy()
        valid_count = 0
        
        # 모든 영역 처리
        for i, region in enumerate(regions):
            # 텍스트 추출
            text = self.extract_text_from_region(image, region)
            
            # 텍스트가 있는 경우에만 처리
            if text.strip():
                # 결과 저장
                results[valid_count] = {
                    'bbox': list(region),  # 튜플을 리스트로 변환
                    'text': text,
                    'confidence': 0.9  # Azure OCR은 개별 단어에 대한 신뢰도를 제공하지만 이 구현에서는 단순화
                }
                
                # 텍스트 제거
                clean_image = self.remove_text(clean_image, region)
                valid_count += 1
        
        return results, clean_image
    
    def visualize_text_regions(self, image, text_results):
        """
        감지된 텍스트 영역을 시각화합니다.
        
        Args:
            image: 원본 이미지
            text_results: 텍스트 결과 사전 (Azure OCR 결과)
            
        Returns:
            시각화된 이미지
        """
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for idx, (key, item) in enumerate(text_results.items()):
            # Azure OCR 결과에서 바운딩 박스 추출
            if isinstance(item, dict) and 'bbox' in item:
                bbox = item['bbox']
                text = item.get('text', '')
                confidence = item.get('confidence', 0)
                
                if len(bbox) == 4:  # [minr, minc, maxr, maxc] 형식
                    minr, minc, maxr, maxc = bbox
                    # 사각형 그리기
                    cv2.rectangle(result, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
                    # 텍스트 및 인덱스 표시
                    cv2.putText(result, f"{idx}: {text[:20]} ({confidence:.2f})", 
                               (minc, minr-10), font, 0.5, (0, 255, 0), 1)
        
        return result
    
    def remove_all_text(self, image, text_results):
        """
        이미지에서 모든 텍스트 영역을 제거합니다.
        
        Args:
            image: 원본 이미지
            text_results: 텍스트 결과 사전 (Azure OCR 결과)
            
        Returns:
            텍스트가 제거된 이미지
        """
        clean_image = image.copy()
        
        for _, item in text_results.items():
            if isinstance(item, dict) and 'bbox' in item:
                bbox = item['bbox']
                if len(bbox) == 4:  # [minr, minc, maxr, maxc] 형식
                    clean_image = self.remove_text(clean_image, bbox)
        
        return clean_image
        
    def process_entire_image(self, image_path):
        """
        Azure OCR API를 사용하여 이미지 전체를 처리합니다.
        이 방법은 텍스트 감지와 인식을 모두 Azure에 의존합니다.
        
        Args:
            image_path: 이미지 경로
            
        Returns:
            텍스트 추출 결과
        """
        # 이미지 파일을 바이트로 읽기
        with open(image_path, "rb") as image_file:
            img_bytes = image_file.read()
        
        # Azure OCR API 호출
        try:
            response = requests.post(
                self.ocr_url,
                headers=self.headers,
                params={'language': self.lang, 'readingOrder': 'natural'},
                data=img_bytes
            )
            response.raise_for_status()
            
            # 작업 상태 폴링을 위한 URL 받기
            operation_url = response.headers["Operation-Location"]
            
            # 분석 결과 기다리기
            analysis = {}
            poll = True
            while poll:
                response_final = requests.get(
                    operation_url,
                    headers={"Ocp-Apim-Subscription-Key": self.api_key}
                )
                analysis = response_final.json()
                
                if "status" in analysis and analysis["status"] == "succeeded":
                    poll = False
                elif "status" in analysis and analysis["status"] == "failed":
                    return {}
                else:
                    time.sleep(1)
            
            # 결과 파싱 및 반환
            results = {}
            count = 0
            
            if "analyzeResult" in analysis and "readResults" in analysis["analyzeResult"]:
                for read_result in analysis["analyzeResult"]["readResults"]:
                    for line in read_result["lines"]:
                        # 바운딩 박스 변환
                        bbox_points = line.get("boundingBox", [])
                        if len(bbox_points) == 8:  # x1,y1,x2,y2,x3,y3,x4,y4 형태
                            # y 최소/최대, x 최소/최대로 변환
                            x_values = [bbox_points[i] for i in range(0, 8, 2)]
                            y_values = [bbox_points[i] for i in range(1, 8, 2)]
                            minr, maxr = min(y_values), max(y_values)
                            minc, maxc = min(x_values), max(x_values)
                            bbox = [int(minr), int(minc), int(maxr), int(maxc)]
                            
                            results[count] = {
                                'bbox': bbox,
                                'text': line["text"],
                                'confidence': line.get("confidence", 0.9)
                            }
                            count += 1
            
            return results
            
        except Exception as e:
            print(f"Error in Azure OCR: {e}")
            return {} 