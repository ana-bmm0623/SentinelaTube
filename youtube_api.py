# youtube_api.py
import re
from googleapiclient.discovery import build

def extrair_id_video(url: str):
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(?:embed\/)?(?:v\/)?(?:shorts\/)?([\w-]{11})(?:\S+)?"
    match = re.search(regex, url)
    return match.group(1) if match else None

def buscar_comentarios(video_id: str, api_key: str, max_comments: int = 500):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        all_comments = []
        request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100, textFormat='plainText')
        
        while request and len(all_comments) < max_comments:
            response = request.execute()
            for item in response['items']:
                all_comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            request = youtube.commentThreads().list_next(previous_request=request, previous_response=response)
        return all_comments
    except Exception as e:
        raise e