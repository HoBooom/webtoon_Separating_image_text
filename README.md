# 웹툰 말풍선 OCR 및 텍스트 제거 프로그램

이 프로그램은 웹툰 이미지에서 말풍선 영역을 감지하고, OCR을 통해 텍스트를 추출한 후, 원본 이미지에서 말풍선 내 텍스트를 제거합니다.

## 기능
- 웹툰 이미지에서 말풍선 영역 감지
- OCR을 통한 말풍선 내 텍스트 추출
- 원본 이미지에서 말풍선 내 텍스트 제거

## 설치 방법
1. Tesseract OCR 설치:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

2. 한국어 언어 팩 설치:
   - Tesseract 언어 데이터 다운로드: https://github.com/tesseract-ocr/tessdata/blob/master/kor.traineddata
   - 다운로드한 kor.traineddata 파일을 Tesseract의 tessdata 디렉토리에 복사

3. 필요한 Python 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

## 사용 방법
```python
python main.py --input [이미지 경로] --output [결과 저장 경로]
``` 