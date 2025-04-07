# 웹툰 텍스트 추출 및 제거 프로그램 (Azure OCR 기반)

이 프로그램은 웹툰 이미지에서 텍스트를 감지하고 추출한 후, 텍스트를 제거하여 깨끗한 이미지를 생성합니다. Microsoft Azure Computer Vision OCR API를 사용하여 텍스트를 인식합니다.

## 주요 기능

1. Azure OCR API를 사용한 텍스트 감지 및 인식
2. 인식된 텍스트 JSON 파일로 저장
3. 텍스트가 제거된 깨끗한 이미지 생성
4. 선택적으로 로컬 텍스트 감지 알고리즘 사용 가능

## 설치 방법

1. 이 저장소를 클론합니다:
   ```
   git clone <repository_url>
   cd <repository_directory>
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. Azure 계정 및 Computer Vision API 설정:
   - [Azure Portal](https://portal.azure.com/)에서 Computer Vision 리소스를 생성합니다.
   - API 키와 엔드포인트를 확인하고 저장합니다.

## 사용 방법

기본 사용법 (Azure OCR로 텍스트 감지):
```
python direct_text_main.py --input <이미지_경로> --api_key <API_키> --endpoint <엔드포인트>
```

로컬 텍스트 감지 알고리즘 사용:
```
python direct_text_main.py --input <이미지_경로> --api_key <API_키> --endpoint <엔드포인트> --use_local_detection
```

전체 옵션:
```
python direct_text_main.py --input <이미지_경로> --output_dir <결과_저장_디렉토리> --api_key <API_키> --endpoint <엔드포인트> --lang <언어_코드> --visualize [--use_local_detection]
```

### 매개변수 설명

- `--input`: 처리할 이미지 파일 경로 (필수)
- `--output_dir`: 결과 저장 디렉토리 (기본값: 'output')
- `--api_key`: Azure Computer Vision API 키 (필수)
- `--endpoint`: Azure Computer Vision API 엔드포인트 (필수)
- `--lang`: OCR 언어 코드 (기본값: 'ko', 가능 값: 'ko', 'en', 'ja' 등)
- `--visualize`: 텍스트 감지 결과 시각화 (선택사항)
- `--use_local_detection`: 로컬 텍스트 감지 알고리즘 사용 (기본값: Azure OCR 사용)

### 예시

Azure OCR로 텍스트 감지 및 인식:
```
python direct_text_main.py --input samples/webtoon.jpg --api_key your_api_key --endpoint https://your-resource.cognitiveservices.azure.com/ --lang ko --visualize
```

로컬 텍스트 감지 + Azure OCR 인식:
```
python direct_text_main.py --input samples/webtoon.jpg --api_key your_api_key --endpoint https://your-resource.cognitiveservices.azure.com/ --lang ko --visualize --use_local_detection
```

## 두 가지 감지 방식 비교

1. **Azure OCR 방식 (기본)**: 
   - 텍스트 감지와 인식을 모두 Azure API로 처리
   - 장점: 더 정확한 텍스트 감지, 정교한 인식
   - 단점: API 호출 비용 발생, 인터넷 연결 필요

2. **로컬 감지 + Azure OCR 방식**:
   - 텍스트 영역 감지는 로컬에서, 인식만 Azure로 처리
   - 장점: API 호출 횟수 감소, 일부 오프라인 작업 가능
   - 단점: 텍스트 감지 정확도가 상대적으로 낮을 수 있음

## 출력 결과

프로그램은 다음과 같은 결과물을 생성합니다:

1. 텍스트 영역이 표시된 시각화 이미지 (--visualize 옵션 사용 시)
2. 추출된 텍스트 정보가 담긴 JSON 파일
3. 텍스트가 제거된 깨끗한 이미지

출력 파일 이름에는 타임스탬프가 포함됩니다.

## 구현 세부 사항

- `AzureTextProcessor`: Azure Computer Vision API를 사용하여 텍스트를 감지, 인식, 제거하는 기능 제공
- `TextDetector`: (선택사항) 로컬에서 이미지 처리 기술을 사용하여 텍스트 영역을 감지하는 기능 제공

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 