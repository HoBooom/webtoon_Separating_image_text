import json
import os
import argparse
from PIL import Image

def generate_text_overlay_data(json_ocr_path, output_json_path):
    """
    OCR JSON 파일을 기반으로 프론트엔드에서 사용할 텍스트 오버레이 데이터를 JSON 형식으로 생성합니다.
    데이터에는 원본 이미지 정보, 상대 좌표(%) 및 스타일 정보가 포함됩니다.
    배경 이미지는 OCR JSON 파일명과 유사한 _clean.png 또는 _clean.jpg 파일을 사용합니다.
    """
    try:
        with open(json_ocr_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: OCR JSON 파일을 찾을 수 없습니다. 경로: {json_ocr_path}")
        return None
    except json.JSONDecodeError:
        print(f"오류: {json_ocr_path}에서 JSON을 디코딩할 수 없습니다.")
        return None

    texts_from_ocr = ocr_data.get("texts", [])
    json_dir = os.path.dirname(os.path.abspath(json_ocr_path))
    json_basename = os.path.basename(json_ocr_path)

    if not json_basename.endswith("_text.json"):
        print(f"오류: OCR JSON 파일명이 '_text.json'으로 끝나지 않아야 합니다. 예: 'prefix_text.json'. 현재 파일명: {json_basename}")
        return None
        
    prefix = json_basename[:-len("_text.json")]
    
    clean_image_filename_png = f"{prefix}_clean.png"
    clean_image_filename_jpg = f"{prefix}_clean.jpg"
    
    path_on_disk_png = os.path.join(json_dir, clean_image_filename_png)
    path_on_disk_jpg = os.path.join(json_dir, clean_image_filename_jpg)

    actual_image_path_on_disk = None
    image_filename_for_frontend = None # 프론트엔드가 참조할 파일명 (경로X)

    if os.path.exists(path_on_disk_png):
        actual_image_path_on_disk = path_on_disk_png
        image_filename_for_frontend = clean_image_filename_png
    elif os.path.exists(path_on_disk_jpg):
        actual_image_path_on_disk = path_on_disk_jpg
        image_filename_for_frontend = clean_image_filename_jpg
    
    if not actual_image_path_on_disk:
        print(f"오류: 클린 이미지를 찾을 수 없습니다. 다음 경로에서 확인: ")
        print(f"  - {path_on_disk_png}")
        print(f"  - {path_on_disk_jpg}")
        return None

    try:
        with Image.open(actual_image_path_on_disk) as img:
            original_image_width, original_image_height = img.size
    except Exception as e:
        print(f"오류: 이미지 파일을 열거나 읽을 수 없습니다. {actual_image_path_on_disk}: {e}")
        return None

    output_data = {
        "original_image_width": original_image_width,
        "original_image_height": original_image_height,
        "image_filename": image_filename_for_frontend, # output_dir에 있다고 가정하고 파일명만 전달
        "texts": []
    }

    for idx, text_info in enumerate(texts_from_ocr):
        text_content = text_info.get("text", "")
        bbox_abs = text_info.get("bbox") # [y_min, x_min, y_max, x_max]

        if not bbox_abs or len(bbox_abs) != 4:
            print(f"경고: 유효하지 않거나 누락된 bbox로 인해 텍스트 항목을 건너뛰었습니다: {text_info}")
            continue

        y_min, x_min, y_max, x_max = bbox_abs
        abs_bbox_width = x_max - x_min
        abs_bbox_height = y_max - y_min

        if abs_bbox_width <= 0 or abs_bbox_height <= 0:
            print(f"경고: 너비 또는 높이가 0 이하인 bbox를 건너뛰었습니다: {bbox_abs}")
            continue
        
        # bbox를 % 단위로 변환
        bbox_percent = {
            "left": (x_min / original_image_width) * 100,
            "top": (y_min / original_image_height) * 100,
            "width": (abs_bbox_width / original_image_width) * 100,
            "height": (abs_bbox_height / original_image_height) * 100
        }

        # 동적 폰트 크기 계산 (px 단위, 원본 이미지 기준)
        num_lines = text_content.count('\n') + 1
        line_height_css_multiplier = 1.2
        
        font_size_h_px = int((abs_bbox_height / num_lines) / line_height_css_multiplier) if num_lines > 0 else abs_bbox_height
        
        longest_line_len = 0
        if num_lines > 1:
            for line_text in text_content.split('\n'):
                if len(line_text) > longest_line_len:
                    longest_line_len = len(line_text)
        else:
            longest_line_len = len(text_content)

        font_size_w_px = float('inf')
        if longest_line_len > 0:
            font_size_w_px = int(abs_bbox_width / (longest_line_len * 0.55)) # 0.6 대신 0.55로 약간 조정

        calculated_font_size_px = min(font_size_h_px, font_size_w_px)
        calculated_font_size_px = max(8, calculated_font_size_px) # 원본 기준 최소 8px
        
        max_font_for_bbox_height_px = int((abs_bbox_height / num_lines) / line_height_css_multiplier * 0.95) if num_lines > 0 else int(abs_bbox_height * 0.7)
        calculated_font_size_px = min(calculated_font_size_px, max_font_for_bbox_height_px)
        calculated_font_size_px = max(1, calculated_font_size_px)

        # 폰트 크기를 vw 단위로 변환 (Viewport Width 기준)
        # 1vw = 원본 이미지 너비의 1%
        # font_size_vw = (calculated_font_size_px / original_image_width) * 100
        # 이 값은 프론트엔드에서 viewport 너비에 따라 실제 픽셀 크기로 변환됨
        font_size_vw = round((calculated_font_size_px / original_image_width) * 100, 2) if original_image_width > 0 else 1.0

        output_data["texts"].append({
            "id": idx,
            "content": text_content,
            "bbox_percent": bbox_percent,
            "style": {
                "font_family": "Malgun Gothic, Arial, sans-serif", # 기본값, 추후 API 파라미터로 변경 가능
                "font_size_vw": font_size_vw, 
                "color": "black", # 기본값
                "text_align": "center",
                "line_height": line_height_css_multiplier
            }
        })

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"텍스트 오버레이 데이터를 성공적으로 생성했습니다: {os.path.abspath(output_json_path)}")
        return output_data
    except IOError:
        print(f"오류: JSON 데이터를 {output_json_path}에 쓸 수 없습니다.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR JSON 파일을 기반으로 프론트엔드용 텍스트 오버레이 JSON 데이터를 생성합니다.\n" \
                    "배경 이미지는 OCR JSON 파일명과 유사한 _clean.png 또는 _clean.jpg 파일을 사용합니다."
    )
    parser.add_argument("ocr_json_file", help="입력 OCR JSON 파일 경로입니다. (예: ..._text.json)")
    parser.add_argument("-o", "--output", 
                        help="출력 데이터 JSON 파일 경로입니다. 기본값은 입력 JSON과 이름이 같고 '_overlay_data.json'으로 끝납니다.")

    args = parser.parse_args()

    ocr_json_file_path = args.ocr_json_file
    output_file_path = args.output

    if not output_file_path:
        base_dir = os.path.dirname(os.path.abspath(ocr_json_file_path))
        base_filename = os.path.splitext(os.path.basename(ocr_json_file_path))[0]
        # _text를 _overlay_data로 변경
        if base_filename.endswith("_text"):
            output_base_filename = base_filename[:-len("_text")] + "_overlay_data"
        else:
            output_base_filename = base_filename + "_overlay_data"
        output_file_path = os.path.join(base_dir, f"{output_base_filename}.json")
    
    generate_text_overlay_data(ocr_json_file_path, output_file_path) 