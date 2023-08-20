import whisper
import argparse
import webvtt
import os
import sys
import re
import subprocess

from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip
from dotenv import dotenv_values

values = dotenv_values(".env")

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token=values["HUGGING_FACE"])


def download_video(url: str) -> str:
    with YoutubeDL() as ydl:
        video = ydl.extract_info(url, download=True)
        requested_downloads = video.get("requested_downloads")

        return requested_downloads[0]["filepath"].split("/")[-1]

def get_audio(video_path: str, output_ext="wav") -> None:
    filename, ext = os.path.splitext(video_path)
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")


def transcribe_audio(audio_path: str) -> list:
    model = whisper.load_model("medium.en")
    result = model.transcribe(audio_path, language="english")
    
    return result["segments"]

def write_transcription(text):
    with open("diarization.txt", "w") as f:
        for segment in text:
            f.write(f"{segment['text']}\n")
    print("done!")

def diarization(audio_path: str):
    diarization = pipeline(audio_path)
    
    with open(f"transcript.txt", "w") as f:
        f.write(str(diarization))

def clean_diarization(audio_path: str) -> list:
    dz = open('diarization.txt').read().splitlines()
    dzList = []
    
    for l in dz:
        start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
        dzList.append([int(millisec(start)), int(millisec(end))])

    #print(*dzList[:10], sep='\n')

    transcribe_audio = f"whisper '{audio_path}' --language en --model small"

    subprocess.run(transcribe_audio, shell=True)

    vtt_path = audio_path.replace("wav", "vtt")

    captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read(vtt_path)]

    captions = [[int(caption[0] / 1000), int(caption[1] / 1000), caption[2]] for caption in captions]

    print(*captions, sep='\n')


    
    
def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s

def main(url: str) -> None:
    video_path = download_video(url)
    print(video_path)
    
    audio_path = video_path.replace("webm", "wav")

    get_audio(video_path)

    #diarization(audio_path)

    #clean_diarization(audio_path)

    transcription = transcribe_audio(audio_path)

    write_transcription(transcription)

parser = argparse.ArgumentParser()

parser.add_argument("--url", help = "Youtube video URL")

args = parser.parse_args()

if args.url:
    main(args.url)