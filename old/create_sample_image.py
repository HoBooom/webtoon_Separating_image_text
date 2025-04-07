import cv2
import numpy as np

def create_speech_bubble_image():
    # 800x600 크기의 흰색 배경 생성
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 배경에 간단한 선 그리기 (캐릭터 대신)
    cv2.line(image, (200, 300), (300, 500), (0, 0, 0), 3)
    cv2.line(image, (300, 500), (400, 300), (0, 0, 0), 3)
    cv2.circle(image, (300, 250), 50, (0, 0, 0), 3)
    
    # 말풍선 그리기
    cv2.ellipse(image, (500, 200), (150, 80), 0, 0, 360, (0, 0, 0), 2)
    
    # 말풍선 안에 한글 텍스트 쓰기
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Hello, World!", (420, 190), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "This is a test", (420, 220), font, 0.8, (0, 0, 0), 2)
    
    # 두 번째 말풍선 그리기
    cv2.ellipse(image, (200, 400), (100, 60), 0, 0, 360, (0, 0, 0), 2)
    cv2.putText(image, "Test bubble", (150, 410), font, 0.7, (0, 0, 0), 2)
    
    # 이미지 저장
    cv2.imwrite("sample.jpg", image)
    print("Sample image created: sample.jpg")

if __name__ == "__main__":
    create_speech_bubble_image() 