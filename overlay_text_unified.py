import json
import os
import argparse
from PIL import Image

def create_overlay_html(json_path, output_html_path, font_size=16, font_family="default", web_font_url=None, local_font_path=None, fixed_size=False):
    """
    JSON 파일을 기반으로 이미지 위에 텍스트를 오버레이하는 HTML 파일을 생성합니다.
    통일된 폰트 크기와 스타일을 적용할 수 있습니다.

    Args:
        json_path: JSON 파일 경로
        output_html_path: 출력 HTML 파일 경로
        font_size: 통일할 폰트 크기 (기본값: 16)
        font_family: 폰트 패밀리 (기본값: "default")
        web_font_url: 웹 폰트 URL (Google Fonts 등)
        local_font_path: 로컬 폰트 파일 경로
        fixed_size: True시 모든 텍스트에 동일한 크기 적용, False시 스마트 조정 (기본값: False)
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
    prefix = json_basename[:-len("_text.json")]
    
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
    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다. 경로: {actual_image_path_on_disk}")
        return
    except Exception as e:
        print(f"오류: 이미지 파일을 열거나 읽을 수 없습니다. {actual_image_path_on_disk}: {e}")
        return

    # 폰트 설정
    font_imports = ""
    font_face_declaration = ""
    
    if web_font_url:
        # 웹 폰트 사용 (Google Fonts 등)
        font_imports = f'@import url("{web_font_url}");'
        if font_family == "default":
            font_family = "웹툰체"  # 사용자가 지정하지 않으면 기본값
    elif local_font_path:
        # 로컬 폰트 파일 사용
        font_filename = os.path.basename(local_font_path)
        font_face_declaration = f"""
        @font-face {{
            font-family: 'CustomWebFont';
            src: url('{font_filename}') format('truetype');
        }}"""
        if font_family == "default":
            font_family = "CustomWebFont"
    else:
        # 기본 폰트 사용
        if font_family == "default":
            font_family = "'Malgun Gothic', 'Arial', sans-serif"

    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>웹툰 텍스트 오버레이 (통일 폰트)</title>
    <style>
        {font_imports}
        {font_face_declaration}
        
        body {{ 
            margin: 0; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
            background-color: #f0f0f0; 
        }}
        
        .webtoon-container {{
            position: relative;
            width: {img_width}px;
            height: {img_height}px;
            background-image: url('{image_path_for_html.replace("\\\\", "/")}');
            background-size: contain;
            background-position: center;
            background-repeat: no-repeat;
            border: 1px solid #ccc;
        }}
        
        .text-overlay {{
            position: absolute;
            background-color: rgba(255, 255, 255, 0.0);
            color: black;
            font-family: {font_family};
            font-size: {font_size}px;
            font-weight: bold;
            line-height: 1.2;
            padding: 2px;
            box-sizing: border-box;
            white-space: pre-wrap;
            overflow: hidden;
            display: flex; 
            align-items: center; 
            justify-content: center; 
            text-align: center;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        }}
        
        /* 폰트 크기 조정 옵션 클래스들 */
        .font-small {{ font-size: {int(font_size * 0.8)}px; }}
        .font-medium {{ font-size: {font_size}px; }}
        .font-large {{ font-size: {int(font_size * 1.2)}px; }}
        .font-xlarge {{ font-size: {int(font_size * 1.5)}px; }}
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

        # 텍스트 길이와 박스 크기에 따른 자동 폰트 크기 조정 클래스 결정
        text_length = len(text_content)
        box_area = width * height
        
        if fixed_size:
            # 고정 크기 사용 - 모든 텍스트에 동일한 크기 적용
            size_class = "font-medium"
        else:
            # 스마트 크기 조정 - 텍스트 길이와 박스 크기에 따라 자동 조정
            # 긴 텍스트나 작은 박스의 경우 작은 폰트 사용
            if text_length > 20 or box_area < 1000:
                size_class = "font-small"
            elif text_length > 10 or box_area < 2000:
                size_class = "font-medium"
            elif box_area > 5000:
                size_class = "font-large"
            else:
                size_class = "font-medium"

        html_content += f"""
        <div class="text-overlay {size_class}" style="left: {left}px; top: {top}px; width: {width}px; height: {height}px;">{text_content}</div>
"""

    html_content += """
    </div>
    
    <!-- 폰트 조정 컨트롤 -->
    <div style="position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; padding: 10px; border-radius: 5px;">
        <h4 style="margin: 0 0 10px 0;">폰트 크기 조정</h4>
        <button onclick="changeFontSize('font-small')">작게</button>
        <button onclick="changeFontSize('font-medium')">보통</button>
        <button onclick="changeFontSize('font-large')">크게</button>
        <button onclick="changeFontSize('font-xlarge')">매우 크게</button>
    </div>
    
    <script>
        function changeFontSize(sizeClass) {
            const textElements = document.querySelectorAll('.text-overlay');
            textElements.forEach(element => {
                // 기존 크기 클래스 제거
                element.classList.remove('font-small', 'font-medium', 'font-large', 'font-xlarge');
                // 새 크기 클래스 추가
                element.classList.add(sizeClass);
            });
        }
    </script>
</body>
</html>
"""

    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"통일 폰트 HTML 오버레이를 성공적으로 생성했습니다: {os.path.abspath(output_html_path)}")
        
        # 로컬 폰트 파일이 지정된 경우 복사
        if local_font_path and os.path.exists(local_font_path):
            import shutil
            output_dir = os.path.dirname(output_html_path)
            font_filename = os.path.basename(local_font_path)
            font_dest_path = os.path.join(output_dir, font_filename)
            shutil.copy2(local_font_path, font_dest_path)
            print(f"폰트 파일을 복사했습니다: {font_dest_path}")
            
    except IOError:
        print(f"오류: HTML 파일을 {output_html_path}에 쓸 수 없습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON에서 텍스트를 가져와 통일된 폰트로 이미지 위에 오버레이하는 HTML 파일을 생성합니다.")
    parser.add_argument("json_file", help="입력 JSON 파일 경로입니다. (예: ..._text.json)")
    parser.add_argument("--font-size", type=int, default=16, help="기본 폰트 크기 (기본값: 16)")
    parser.add_argument("--font-family", default="default", help="폰트 패밀리 (기본값: 시스템 기본)")
    parser.add_argument("--web-font", help="웹 폰트 URL (예: Google Fonts URL)")
    parser.add_argument("--local-font", help="로컬 폰트 파일 경로 (.ttf, .otf 등)")
    parser.add_argument("--fixed-size", action="store_true", help="모든 텍스트에 동일한 폰트 크기 적용 (스마트 조정 비활성화)")
    
    args = parser.parse_args()

    json_file_path = args.json_file
    
    json_dir_for_output = os.path.dirname(os.path.abspath(json_file_path))
    json_filename_without_ext_for_output = os.path.splitext(os.path.basename(json_file_path))[0]
    output_html_file_path = os.path.join(json_dir_for_output, f"{json_filename_without_ext_for_output}_unified_overlay.html")
    
    create_overlay_html(
        json_file_path, 
        output_html_file_path,
        font_size=args.font_size,
        font_family=args.font_family,
        web_font_url=args.web_font,
        local_font_path=args.local_font,
        fixed_size=args.fixed_size
    )