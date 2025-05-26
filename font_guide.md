# 웹툰 텍스트 오버레이 폰트 통일 가이드

## 개요
`overlay_text_unified.py` 스크립트를 사용하여 모든 말풍선에 동일한 폰트와 크기를 적용할 수 있습니다.

## 사용법

### 1. 기본 사용 (시스템 기본 폰트)
```bash
python overlay_text_unified.py your_image_text.json
```

### 2. 폰트 크기 지정
```bash
python overlay_text_unified.py your_image_text.json --font-size 20
```

### 3. 웹 폰트 사용 (Google Fonts)
```bash
# 예시: Noto Sans KR 폰트 사용
python overlay_text_unified.py your_image_text.json --web-font "https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap" --font-family "'Noto Sans KR', sans-serif"

# 예시: 웹툰체 (무료 웹툰 폰트)
python overlay_text_unified.py your_image_text.json --web-font "https://fonts.googleapis.com/css2?family=Gugi&display=swap" --font-family "'Gugi', cursive"
```

### 4. 로컬 폰트 파일 사용
```bash
python overlay_text_unified.py your_image_text.json --local-font "./fonts/MyCustomFont.ttf" --font-family "CustomWebFont"
```

### 5. 완전 고정 폰트 크기 (웹 폰트 + 고정 크기)
```bash
# 모든 말풍선에 정확히 동일한 크기 적용 (스마트 조정 비활성화)
python3 overlay_text_unified.py your_image_text.json \
  --font-size 18 \
  --web-font "https://fonts.googleapis.com/css2?family=Gugi&display=swap" \
  --font-family "'Gugi', cursive" \
  --fixed-size

# 로컬 폰트로 완전 고정 크기
python3 overlay_text_unified.py your_image_text.json \
  --font-size 20 \
  --local-font "./fonts/NanumBarunGothic.ttf" \
  --fixed-size
```

## 추천 웹툰 폰트

### 1. 무료 웹 폰트 (Google Fonts)

#### 한글 폰트
- **Noto Sans KR**: 깔끔한 산세리프
  ```
  --web-font "https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap"
  --font-family "'Noto Sans KR', sans-serif"
  ```

- **Gugi**: 웹툰스러운 둥근 폰트
  ```
  --web-font "https://fonts.googleapis.com/css2?family=Gugi&display=swap"
  --font-family "'Gugi', cursive"
  ```

- **Jua**: 귀여운 웹툰체
  ```
  --web-font "https://fonts.googleapis.com/css2?family=Jua&display=swap"
  --font-family "'Jua', sans-serif"
  ```

- **Gamja Flower**: 손글씨 느낌
  ```
  --web-font "https://fonts.googleapis.com/css2?family=Gamja+Flower&display=swap"
  --font-family "'Gamja Flower', cursive"
  ```

#### 영문 폰트 (웹툰에 어울리는)
- **Comic Neue**: 만화체
  ```
  --web-font "https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap"
  --font-family "'Comic Neue', cursive"
  ```

### 2. 무료 로컬 폰트 다운로드

#### 네이버 나눔폰트
- 다운로드: https://hangeul.naver.com/font
- 추천: `NanumBarunGothic.ttf` (웹툰에 적합한 가독성)

#### 우아한형제들 폰트
- **배민 주아체**: https://www.woowahan.com/fonts
- **배민 한나체**: 웹툰에 매우 어울림

#### 카카오 폰트
- **카카오체**: https://brunch.co.kr/@designcompass/97

#### 넥슨 폰트
- **넥슨 Lv.1 고딕**: https://font.nexon.com/
- **넥슨 Lv.2 고딕**: 더 두껍고 강조된 느낌

### 3. 로컬 폰트 설치 방법

1. 폰트 파일 다운로드 (.ttf, .otf 파일)
2. 프로젝트 폴더에 `fonts` 디렉토리 생성
3. 폰트 파일을 `fonts` 폴더에 복사
4. 스크립트 실행 시 경로 지정

```bash
mkdir fonts
# 폰트 파일을 fonts 폴더에 복사
python overlay_text_unified.py your_image_text.json --local-font "./fonts/NanumBarunGothic.ttf"
```

## 실시간 폰트 크기 조정

생성된 HTML 파일에는 우측 상단에 폰트 크기 조정 버튼이 있습니다:
- **작게**: 기본 크기의 80%
- **보통**: 지정한 기본 크기
- **크게**: 기본 크기의 120%
- **매우 크게**: 기본 크기의 150%

## 폰트 효과 추가

현재 적용된 효과:
- `font-weight: bold`: 굵은 글씨로 가독성 향상
- `text-shadow`: 흰색 그림자로 배경과 구분
- 자동 크기 조정: 텍스트 길이와 박스 크기에 따른 스마트 조정

## 예시 명령어 모음

```bash
# 20px 크기로 Noto Sans KR 사용
python overlay_text_unified.py sample_text.json \
  --font-size 20 \
  --web-font "https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap" \
  --font-family "'Noto Sans KR', sans-serif"

# 18px 크기로 로컬 배민 주아체 사용
python overlay_text_unified.py sample_text.json \
  --font-size 18 \
  --local-font "./fonts/BMJua.ttf"

# 기본 시스템 폰트로 24px 크기
python overlay_text_unified.py sample_text.json --font-size 24

# 웹 폰트로 완전 고정 크기 (모든 말풍선이 정확히 동일한 크기)
python overlay_text_unified.py sample_text.json \
  --font-size 18 \
  --web-font "https://fonts.googleapis.com/css2?family=Jua&display=swap" \
  --font-family "'Jua', sans-serif" \
  --fixed-size
```

## 폰트 크기 모드 비교

### 스마트 조정 모드 (기본값)
- 텍스트 길이와 말풍선 크기에 따라 자동으로 크기 조정
- 긴 텍스트나 작은 말풍선은 작은 글씨, 짧은 텍스트나 큰 말풍선은 큰 글씨
- 가독성과 미관을 자동으로 최적화

### 고정 크기 모드 (`--fixed-size` 옵션)
- 모든 말풍선에 정확히 동일한 폰트 크기 적용
- 일관성 최우선, 모든 텍스트가 동일한 크기
- 브랜드 가이드라인이나 특별한 요구사항이 있을 때 유용

## 팁

1. **웹툰 장르에 따른 폰트 선택**:
   - 로맨스: Jua, Gamja Flower (부드러운 느낌)
   - 액션: Noto Sans KR Bold (선명하고 강한 느낌)
   - 개그: Gugi, Comic Neue (경쾌한 느낌)

2. **폰트 크기 권장사항**:
   - 모바일 보기: 14-18px
   - 데스크톱 보기: 16-22px
   - 인쇄용: 20-24px

3. **가독성 향상**:
   - 진한 배경에는 밝은 텍스트
   - 밝은 배경에는 진한 텍스트
   - 텍스트 그림자(text-shadow) 활용

4. **성능 고려사항**:
   - 웹 폰트는 로딩 시간이 있음
   - 로컬 폰트는 빠르지만 파일 공유 시 폰트도 함께 배포 필요 