# 웹툰 말풍선 텍스트 추출 시스템

이 프로젝트는 웹툰 이미지에서 말풍선을 감지하고, 말풍선 내의 텍스트만 인식하여 추출하는 시스템입니다.

## 주요 기능

1. Hugging Face의 YOLOv8 기반 말풍선 감지 모델을 사용하여 웹툰 이미지에서 말풍선 영역을 감지
2. Azure Computer Vision OCR을 사용하여 텍스트 인식
3. 말풍선 내에 위치한 텍스트만 필터링하여 처리
4. 말풍선 내 텍스트가 제거된 클린 이미지 생성
5. 결과 시각화 (말풍선 및 텍스트 영역 표시)

## 설치 방법

### 1. 필요 라이브러리 설치

```bash
pip install ultralytics opencv-python pillow numpy torch azure-cognitiveservices-vision-computervision requests
```

### 2. Azure Computer Vision API 설정

- [Azure 포털](https://portal.azure.com/)에서 Computer Vision 리소스 생성
- API 키 및 엔드포인트 정보 확인

### 3. Hugging Face 모델 사용 설정

- Hugging Face 모델은 코드 실행 시 자동으로 다운로드됩니다.
- 오프라인 사용을 위해서는 모델을 로컬에 다운로드할 수 있습니다.

## 사용 방법

### 예제 코드

```python
from speech_bubble_ocr import SpeechBubbleOCR

# 초기화
processor = SpeechBubbleOCR(
    model_path="kitsumed/yolov8m_seg-speech-bubble",  # Hugging Face 모델
    azure_api_key="your_azure_api_key",
    azure_endpoint="your_azure_endpoint",
    lang="ko"  # 한국어
)

# 이미지 처리
image_path = "path_to_webtoon_image.jpg"
text_results, clean_image = processor.process_image(image_path)

# 결과 시각화
result_image = processor.visualize_results(image_path, "result.jpg")

# 결과 출력
print(f"말풍선 내 텍스트 개수: {len(text_results)}")
for idx, (key, item) in enumerate(text_results.items()):
    print(f"텍스트 {idx}: {item['text']}")
```

### 명령줄 도구 사용_ 실행

```bash
python example_usage.py --image path_to_image.jpg --api_key your_azure_api_key --endpoint your_azure_endpoint
```
### 글자 박스 병합_주 실행 내요
```bash
python example_usage.py --image test_images/sample.jpg --api_key your_key --endpoint your_endpoint --merge --merge_any_overlap
```
### 병합 사용 x
```bash
python example_usage.py --image test_images/sample.jpg --api_key your_key --endpoint your_endpoint --merge --no-merge_any_overlap
```

### for fine_tuning the detection parameters
```bash
python tune_detection.py --image path_to_image.jpg --api_key your_azure_api_key --endpoint your_azure_endpoint
```

### for batch processing
```bash
python batch_process.py --input_dir your_images_folder --api_key your_azure_api_key --endpoint your_azure_endpoint
```


#### 추가 옵션

- `--model`: 사용할 Hugging Face 말풍선 감지 모델 (기본값: kitsumed/yolov8m_seg-speech-bubble)
- `--lang`: OCR 언어 (기본값: ko)
- `--output_dir`: 결과 저장 디렉토리 (기본값: output)
- `--overlap`: 텍스트가 말풍선 내에 있다고 판단할 최소 중첩 비율 (0.0-1.0, 기본값: 0.5)
- `--confidence`: 말풍선 감지 신뢰도 임계값 (0.0-1.0, 기본값: 0.5)

## 출력 결과

1. 시각화된 이미지: 말풍선(파란색)과 인식된 텍스트 영역(녹색) 표시
2. 텍스트가 삭제된 클린 이미지: 말풍선 내 텍스트만 제거된 이미지
3. 텍스트 파일: 추출된 텍스트 내용

## 주의사항

1. 말풍선 감지 품질은 사용하는 모델과 웹툰 스타일에 따라 달라질 수 있습니다.
2. Azure API 사용량에 따라 비용이 발생할 수 있습니다.
3. 이미지 해상도가 높을수록 처리 시간이 길어질 수 있습니다.

## 참고 자료

- [YOLOv8 Speech Bubble Detection Model](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- [Azure Computer Vision OCR](https://docs.microsoft.com/azure/cognitive-services/computer-vision/overview-ocr) 