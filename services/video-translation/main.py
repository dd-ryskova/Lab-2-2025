from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import FileResponse
import subprocess
import tempfile
import requests
import os
from transformers import pipeline
import srt
from datetime import timedelta

app = FastAPI()

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
HF_MODEL_URL = os.environ.get("HUGGINGFACE_MODEL_URL", "")
HF_TR_MODEL_URL = os.environ.get("HUGGINGFACE_TR_MODEL_URL", "")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

@app.post("/extract-audio")
async def extract_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as video_tmp:
        video_path = video_tmp.name
        video_tmp.write(await file.read())

    audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = audio_tmp.name

    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            audio_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device="cpu"
        )

        result = whisper_pipe(
            audio_path,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5
        )

        segments = result["chunks"]

        subs = []
        for i, seg in enumerate(segments, start=1):
            start = timedelta(seconds=seg["timestamp"][0])
            end = timedelta(seconds=seg["timestamp"][1])
            text = seg["text"]

            subs.append(srt.Subtitle(i, start, end, text))

        srt_text = srt.compose(subs)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as srt_tmp:
            srt_path = srt_tmp.name
            srt_tmp.write(srt_text.encode('utf-8'))
        
        return FileResponse(srt_path, filename="subs.srt", media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
    finally:
        os.remove(video_path)
        os.remove(audio_path)


@app.post("/translate")
async def translate(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as video_tmp:
        video_path = video_tmp.name
        ru_text = await file.read()
        video_tmp.write(ru_text)
    

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"Translate next text to Russian language, only translation, dont comment, save format, only replace words:\n{ru_text}"
            }
        ],
        "model": "deepseek-ai/DeepSeek-V3.2:novita",
        "stream": False
    }

    h = HEADERS.copy()
    h["Content_Type"] = "application/json"

    response = requests.post(HF_TR_MODEL_URL, headers=h, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HF API error: {response.text}")

    data = response.json()

    try:
        ru_text = data["choices"][0]["message"]["content"]
        ru_text = ru_text.replace("\\n", "\n")
        ru_text = ru_text.replace("b'", "")
        ru_text = ru_text[0:-1] if ru_text[-1] == "'" else ru_text

        with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as srt_tmp:
            srt_path = srt_tmp.name
            srt_tmp.write(ru_text.encode('utf-8'))

        return FileResponse(srt_path, filename="ru_subs.srt", media_type="text/plain")
    except Exception:
        raise HTTPException(status_code=500, detail=f"Unexpected HF response: {data}")
    finally:
        os.remove(video_path)


@app.post("/burn-subtitles")
async def burn_subtitles(srt_file: UploadFile = File(...), video_file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as video_tmp:
        video_path = video_tmp.name
        video_tmp.write(await video_file.read())
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(srt_file.filename)[1]) as srt_tmp:
        srt_path = srt_tmp.name
        srt_tmp.write(await srt_file.read())

    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_tmp.name
    output_tmp.close()

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vf", f"subtitles={srt_path}:charenc=UTF-8:force_style='FontName=DejaVuSans,Fontsize=12,Outline=1,Shadow=1,MarginV=40'",
            "-c:a", "copy",
            output_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg error:\n{result.stderr}"
            )

        return FileResponse(output_path, filename="result.mp4", media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
    finally:
        os.remove(video_path)


@app.get("/clear")
async def clear():
    subprocess.run("rm -rf /tmp/*", shell=True)