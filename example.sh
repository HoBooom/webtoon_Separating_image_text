#!/bin/bash

# 예제 이미지에서 말풍선 텍스트 추출 및 제거

# 입력 이미지 경로 (샘플 이미지로 변경하세요)
IMAGE_PATH=".images/sample.jpg"

# Tesseract 실행 파일 경로 (Windows에서 필요한 경우 아래 줄의 주석을 제거하고 경로를 수정하세요)
# TESSERACT_PATH="C:\Program Files\Tesseract-OCR\tesseract.exe"

# 결과 저장 디렉토리
OUTPUT_DIR="./output"

# 실행 명령어
python main.py --input "$IMAGE_PATH" --output_dir "$OUTPUT_DIR" --visualize

# Windows에서 Tesseract 경로가 필요한 경우 아래 줄의 주석을 제거하고 위 줄을 주석 처리하세요
# python main.py --input "$IMAGE_PATH" --output_dir "$OUTPUT_DIR" --tesseract_path "$TESSERACT_PATH" --visualize

echo "처리가 완료되었습니다. 결과는 $OUTPUT_DIR 디렉토리에서 확인할 수 있습니다." 