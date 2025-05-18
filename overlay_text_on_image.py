import json
import os
import argparse
from PIL import Image

def create_overlay_html(json_path, output_html_path):
    """
    JSON 파일을 기반으로 이미지 위에 텍스트를 오버레이하는 HTML 파일을 생성합니다.
    JSON 파일과 동일한 이름 규칙을 따르는 _clean.png 또는 _clean.jpg 이미지를 배경으로 사용합니다.

    JSON 파일은 다음을 포함해야 합니다:
    - "texts": 텍스트 객체 목록, 각 객체는 다음을 포함:
        - "text": 표시할 문자열
        - "bbox": 텍스트 경계 상자의 [y_min, x_min, y_max, x_max] 목록
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: JSON 파일을 찾을 수 없습니다. 경로: {json_path}")
        return
    except json.JSONDecodeError:
        print(f"오류: {json_path}에서 JSON을 디코딩할 수 없습니다.")
        return

    texts = data.get("texts", [])
    json_dir = os.path.dirname(os.path.abspath(json_path))
    json_basename = os.path.basename(json_path)

    if not json_basename.endswith("_text.json"):
        print(f"오류: JSON 파일명이 '_text.json'으로 끝나지 않습니다. 예: 'prefix_text.json'. 현재 파일명: {json_basename}")
        return
        
    # Derive _clean.png / _clean.jpg path
    prefix = json_basename[:-len("_text.json")] # "_text.json" 제거하여 prefix 추출
    
    clean_image_filename_png = f"{prefix}_clean.png"
    clean_image_filename_jpg = f"{prefix}_clean.jpg"
    
    actual_image_path_on_disk_png = os.path.join(json_dir, clean_image_filename_png)
    actual_image_path_on_disk_jpg = os.path.join(json_dir, clean_image_filename_jpg)

    actual_image_path_on_disk = None
    image_path_for_html = None

    if os.path.exists(actual_image_path_on_disk_png):
        actual_image_path_on_disk = actual_image_path_on_disk_png
        image_path_for_html = clean_image_filename_png
    elif os.path.exists(actual_image_path_on_disk_jpg):
        actual_image_path_on_disk = actual_image_path_on_disk_jpg
        image_path_for_html = clean_image_filename_jpg
    
    if not actual_image_path_on_disk:
        print(f"오류: 클린 이미지를 찾을 수 없습니다. 다음 경로에서 확인: ")
        print(f"  - {actual_image_path_on_disk_png}")
        print(f"  - {actual_image_path_on_disk_jpg}")
        return

    try:
        with Image.open(actual_image_path_on_disk) as img:
            img_width, img_height = img.size
    except FileNotFoundError: # 이미 위에서 확인했지만, 안전장치로 둡니다.
        print(f"오류: 이미지 파일을 찾을 수 없습니다. 경로: {actual_image_path_on_disk}")
        return
    except Exception as e:
        print(f"오류: 이미지 파일을 열거나 읽을 수 없습니다. {actual_image_path_on_disk}: {e}")
        return

    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>웹툰 텍스트 오버레이</title>
    <style>
        body {{ margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; background-color: #f0f0f0; }}
        .webtoon-container {{
            position: relative;
            width: {img_width}px;
            height: {img_height}px;
            background-image: url('{image_path_for_html.replace("\\\\", "/")}'); /* URL에는 슬래시(/)를 사용하도록 보장 */
            background-size: contain; /* 이미지 비율 유지하며 컨테이너에 맞춤 */
            background-position: center; /* 이미지를 컨테이너 중앙에 위치 */
            background-repeat: no-repeat;
            border: 1px solid #ccc;
        }}
        .text-overlay {{
            position: absolute;
            background-color: rgba(255, 255, 255, 0.0); /* 투명 배경, 디버깅 시 변경 가능 */
            color: black; /* 기본 텍스트 색상 */
            font-family: 'Malgun Gothic', 'Arial', sans-serif; /* 기본 글꼴 */
            /* font-size: 16px; /* 기본 글꼴 크기, 아래에서 동적으로 설정됨 */
            line-height: 1.2;
            padding: 2px;
            box-sizing: border-box;
            /* border: 1px dashed red; /* 텍스트 상자 경계 디버깅용 */ */
            white-space: pre-wrap; /* 텍스트의 공백 및 줄바꿈 유지 */
            overflow: hidden; /* bbox를 벗어나는 텍스트 자르기 */
            display: flex; 
            align-items: center; 
            justify-content: center; 
            text-align: center; 
        }}
    </style>
</head>
<body>
    <div class="webtoon-container">
"""

    for text_info in texts:
        text_content = text_info.get("text", "")
        bbox = text_info.get("bbox")

        if not bbox or len(bbox) != 4:
            print(f"경고: 유효하지 않거나 누락된 bbox로 인해 텍스트 항목을 건너뛰었습니다: {text_info}")
            continue

        y_min, x_min, y_max, x_max = bbox
        
        top = y_min
        left = x_min
        width = x_max - x_min
        height = y_max - y_min

        if width <= 0 or height <= 0:
            print(f"경고: 너비 또는 높이가 0 이하인 bbox를 건너뛰었습니다: {bbox}")
            continue
            
        # 글꼴 크기 동적 계산 (간단한 휴리스틱)
        # 높이를 기준으로 기본 크기 설정
        dynamic_font_size = int(height * 0.7)
        # 텍스트 길이에 비해 너비가 너무 좁으면 글꼴 크기 줄이기
        # (문자 평균 너비를 글꼴 크기의 0.6배로 가정)
        if len(text_content) > 0 and width < len(text_content) * (dynamic_font_size * 0.6):
            dynamic_font_size = int(width / (len(text_content) * 0.6))

        dynamic_font_size = max(8, dynamic_font_size) # 최소 글꼴 크기 8px
        dynamic_font_size = min(dynamic_font_size, height - 2) # 패딩 고려, 높이를 초과하지 않도록
        dynamic_font_size = max(1, dynamic_font_size) # 0이나 음수가 되지 않도록 최종 확인

        html_content += f"""
        <div class="text-overlay" style="left: {left}px; top: {top}px; width: {width}px; height: {height}px; font-size: {dynamic_font_size}px;">{text_content}</div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML 오버레이를 성공적으로 생성했습니다: {os.path.abspath(output_html_path)}")
    except IOError:
        print(f"오류: HTML 파일을 {output_html_path}에 쓸 수 없습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON에서 텍스트를 가져와 이미지 위에 오버레이하는 HTML 파일을 생성합니다.\\nJSON 파일과 이름이 유사한 _clean.png 또는 _clean.jpg 이미지를 배경으로 사용합니다.")
    parser.add_argument("json_file", help="입력 JSON 파일 경로입니다. (예: ..._text.json)")
    
    args = parser.parse_args()

    json_file_path = args.json_file
    
    json_dir_for_output = os.path.dirname(os.path.abspath(json_file_path))
    json_filename_without_ext_for_output = os.path.splitext(os.path.basename(json_file_path))[0]
    output_html_file_path = os.path.join(json_dir_for_output, f"{json_filename_without_ext_for_output}_overlay.html")
    
    create_overlay_html(json_file_path, output_html_file_path) 