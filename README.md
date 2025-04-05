# 웹툰 이미지 텍스트 감지 및 제거 프로그램

이 프로그램은 웹툰 이미지에서 텍스트 영역을 직접 감지하고, PaddleOCR을 통해 텍스트를 추출한 후, 원본 이미지에서 텍스트를 제거합니다.

## 기능
- 웹툰 이미지에서 텍스트 영역 직접 감지
- PaddleOCR을 통한 정확한 텍스트 추출
- 주변 배경색을 사용하여 자연스러운 텍스트 제거 
- 빈 텍스트 영역 자동 필터링 (텍스트가 인식되지 않은 영역 제외)
- 텍스트 영역 시각화

## 설치 방법
1. PaddlePaddle 및 PaddleOCR 설치:
   ```
   pip install paddlepaddle paddleocr
   ```

2. 필요한 Python 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

## 사용 방법
```bash
python direct_text_main.py --input [이미지 경로] --output_dir [결과 저장 디렉토리] --visualize
```

### 고급 옵션
```bash
python direct_text_main.py --input [이미지 경로] --output_dir [결과 저장 디렉토리] --lang [언어코드] --visualize
```

## 지원 언어
- 한국어: `korean` (기본값)
- 영어: `en`
- 일본어: `japan`
- 중국어 간체: `ch`
- 중국어 번체: `chinese_cht`
- 기타 언어는 PaddleOCR 문서 참조

## 처리 과정
1. **텍스트 영역 감지**: 이미지에서 텍스트가 있는 영역을 자동으로 감지합니다.
2. **OCR 처리**: 감지된 각 영역에서 PaddleOCR을 사용하여 텍스트를 추출합니다.
3. **텍스트 필터링**: 텍스트가 인식되지 않은 영역은 처리 대상에서 제외됩니다.
4. **텍스트 제거**: 주변 배경색을 분석하여 텍스트 영역을 자연스럽게 채웁니다.
5. **경계 부드럽게 처리**: 가우시안 블러로 제거된 영역의 경계를 부드럽게 처리합니다.

## 파일 설명
- `direct_text_main.py`: 메인 실행 파일
- `text_detector.py`: 텍스트 영역 감지 모듈
- `text_processor.py`: 텍스트 추출 및 제거 모듈
- `direct_text_test.py`: 테스트 및 디버깅용 스크립트

## 출력 파일
- `*_clean.jpg`: 텍스트가 제거된 깨끗한 이미지
- `*_texts.json`: 추출된 텍스트 정보 (위치 및 내용)
- `*_text_regions.jpg`: 감지된 텍스트 영역 시각화 (--visualize 옵션 사용 시) 