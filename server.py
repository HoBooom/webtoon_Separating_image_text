from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import subprocess
import json
from supabase import create_client, Client
from dotenv import load_dotenv
import time
from typing import Optional

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱의 URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase 클라이언트 초기화
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

class ProcessRequest(BaseModel):
    input_dir: str
    episode_title: str
    series_id: int
    episode_number: int

class DeleteSeriesRequest(BaseModel):
    series_id: int

class DeleteEpisodeRequest(BaseModel):
    series_id: int
    episode_id: int

@app.post("/process-images")
async def process_images(request: ProcessRequest):
    try:
        # 입력 디렉토리 확인
        if not os.path.exists(request.input_dir):
            raise HTTPException(status_code=400, detail=f"입력 디렉토리를 찾을 수 없습니다: {request.input_dir}")

        # 현재 시간을 기반으로 한 출력 디렉토리 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"batch_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # batch_process.py 실행
        process = subprocess.run([
            "python3", "batch_process.py",
            "--input_dir", request.input_dir,
            "--output_dir", output_dir,
            "--api_key", os.getenv("AZURE_API_KEY"),
            "--endpoint", os.getenv("AZURE_ENDPOINT"),
            "--visualize"
        ], capture_output=True, text=True)

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"AI 처리 실패: {process.stderr}")

        # 처리된 이미지와 JSON 파일 찾기
        clean_image = None
        json_data = None
        for file in os.listdir(output_dir):
            if file.endswith("_clean.png"):
                clean_image = os.path.join(output_dir, file)
            elif file.endswith("_text.json"):
                json_data = os.path.join(output_dir, file)

        if not clean_image or not json_data:
            raise HTTPException(status_code=500, detail="처리된 파일을 찾을 수 없습니다.")

        # Supabase Storage에 파일 업로드
        with open(clean_image, "rb") as f:
            clean_image_path = f"episodes/{request.series_id}/{request.episode_number}/clean_image.png"
            supabase.storage.from_("webtoon-images").upload(clean_image_path, f)

        with open(json_data, "rb") as f:
            json_data_path = f"episodes/{request.series_id}/{request.episode_number}/text_data.json"
            supabase.storage.from_("webtoon-data").upload(json_data_path, f)

        # 공개 URL 생성
        clean_image_url = supabase.storage.from_("webtoon-images").get_public_url(clean_image_path)
        json_data_url = supabase.storage.from_("webtoon-data").get_public_url(json_data_path)

        # summary.json에서 처리 결과 읽기
        with open(os.path.join(output_dir, "summary.json"), "r") as f:
            summary = json.load(f)

        return {
            "clean_image_url": clean_image_url,
            "json_data_url": json_data_url,
            "total_texts": summary["total_texts_extracted"],
            "processing_time": summary["processing_time_seconds"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_error_message(error):
    """에러 메시지를 문자열로 변환"""
    if isinstance(error, HTTPException):
        return error.detail
    if isinstance(error, dict):
        return str(error.get('message', '알 수 없는 오류가 발생했습니다.'))
    return str(error)

@app.delete("/series/{series_id}")
async def delete_series(series_id: int):
    try:
        # 시리즈 존재 여부 확인
        series = supabase.table("series").select("*").eq("id", series_id).single().execute()
        
        if not series.data:
            raise HTTPException(status_code=404, detail=f"시리즈 ID {series_id}를 찾을 수 없습니다.")

        # 시리즈의 모든 에피소드 조회
        episodes = supabase.table("episodes").select("*").eq("series_id", series_id).execute()
        
        # 각 에피소드의 저장소 파일 삭제
        storage_errors = []
        for episode in episodes.data:
            try:
                # 이미지 파일 삭제
                image_path = f"episodes/{series_id}/{episode['episode_number']}/clean_image.png"
                supabase.storage.from_("webtoon-images").remove([image_path])
                
                # JSON 파일 삭제
                json_path = f"episodes/{series_id}/{episode['episode_number']}/text_data.json"
                supabase.storage.from_("webtoon-data").remove([json_path])
            except Exception as e:
                storage_errors.append(f"에피소드 {episode['id']} 파일 삭제 실패: {format_error_message(e)}")

        # 시리즈의 모든 에피소드 삭제
        try:
            episodes_result = supabase.table("episodes").delete().eq("series_id", series_id).execute()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"에피소드 삭제 중 오류 발생: {format_error_message(e)}")

        # 시리즈 삭제
        try:
            series_result = supabase.table("series").delete().eq("id", series_id).execute()
            if not series_result.data:
                raise HTTPException(status_code=500, detail="시리즈 삭제 중 오류가 발생했습니다.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"시리즈 삭제 중 오류 발생: {format_error_message(e)}")

        response_message = "시리즈가 성공적으로 삭제되었습니다."
        if storage_errors:
            response_message += f" (경고: {'; '.join(storage_errors)})"

        return {"message": response_message}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=format_error_message(e))

@app.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: int, series_id: int):
    try:
        # 에피소드 정보 조회
        episode = supabase.table("episodes").select("*").eq("id", episode_id).single().execute()
        
        if not episode.data:
            raise HTTPException(status_code=404, detail=f"에피소드 ID {episode_id}를 찾을 수 없습니다.")

        if episode.data["series_id"] != series_id:
            raise HTTPException(status_code=400, detail="에피소드가 지정된 시리즈에 속하지 않습니다.")

        storage_errors = []
        # 저장소 파일 삭제
        try:
            # 이미지 파일 삭제
            image_path = f"episodes/{series_id}/{episode.data['episode_number']}/clean_image.png"
            supabase.storage.from_("webtoon-images").remove([image_path])
            
            # JSON 파일 삭제
            json_path = f"episodes/{series_id}/{episode.data['episode_number']}/text_data.json"
            supabase.storage.from_("webtoon-data").remove([json_path])
        except Exception as e:
            storage_errors.append(f"파일 삭제 실패: {format_error_message(e)}")

        # 에피소드 삭제
        try:
            result = supabase.table("episodes").delete().eq("id", episode_id).execute()
            if not result.data:
                raise HTTPException(status_code=500, detail="에피소드 삭제 중 오류가 발생했습니다.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"에피소드 삭제 중 오류 발생: {format_error_message(e)}")

        response_message = "에피소드가 성공적으로 삭제되었습니다."
        if storage_errors:
            response_message += f" (경고: {'; '.join(storage_errors)})"

        return {"message": response_message}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=format_error_message(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 