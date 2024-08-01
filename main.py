from facenet_pytorch import MTCNN, InceptionResnetV1 
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from requests import post, get
import os
import base64
import json
import spotipy 
from spotipy.oauth2 import SpotifyOAuth
import random
from database import Database
from facenet_pytorch import MTCNN, InceptionResnetV1 
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import numpy as np
import io
import json
import os


# install pytorch
#       pip install facenet-pytorch
#       pip install mtcnn
#       pip install spotipy --upgrade



load_dotenv()   

mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

def extract_face_and_encoding(image_path):
    image = Image.open(image_path)
    faces, boxes = mtcnn(image)
    if faces is not None and boxes is not None:
        # Calculate areas of the bounding boxes
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        
        # Find the index of the largest face
        largest_face_index = areas.index(max(areas))
        
        # Get the largest face
        largest_face = faces[largest_face_index]
        
        # Convert the face tensor to a PIL image
        largest_face_img = Image.fromarray(largest_face.permute(1, 2, 0).int().numpy())
        
        largest_face = faces[largest_face_index].unsqeeze(0)
        encoding = resnet(largest_face).detach().numpy()
        
    else:
        print("No faces detected")
        return None
    
    return  largest_face_img

#----------------------API----------------------#

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
redirect_uri = os.getenv("REDIRECT_URI")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data ={"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}


# returns the artist
def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"

    query_url = url + query
    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)["artists"]["items"]
    if len(json_result) == 0:
        print("No artist with this name exists")
        return None

    return json_result[0]

# returns the top tracks of the artist
def get_songs_by_artist(token, artist_id): 
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]["items"]
    return json_result
    



token = get_token()

# -----------------------Spotipy below------------------------------#

    
def get_top_artists(sp):
    uris = []
    
    ranges = ["short_term", "medium_term", "long_term"]
    for r in ranges:
        all_data = sp.current_user_top_artists(limit=50, time_range=r)
        top_artists_data = all_data["items"]
        for data in top_artists_data:
            uris.append(data["uri"])

    return uris

def get_songs(sp, top_artists_uri):
    song_uris = []
    for artist in top_artists_uri:
        all_data = sp.artist_top_tracks(artist)
        top_songs_data = all_data["tracks"]
        for data in top_songs_data:
            song_uris.append(data["uri"])

    return song_uris
    pass

# adds playlist directly to user's account
def create_playlist(sp, song_uris, emotion):
    user_all_data = sp.current_user()
    id = user_all_data["id"]

    playlist_all_data = sp.user_playlist_create(id, "Feeling " + emotion)
    playlist_id = playlist_all_data["id"]
    random.shuffle(song_uris)
    sp.user_playlist_add_tracks(id, playlist_id, song_uris[0:10])
    print(song_uris[i]["name"] for i in range(10))
    pass




emotions = { "angry": {
            "min_tempo": 0.7, "max_tempo": 1.0,
            "min_energy": 0.8, "max_energy": 1.0,
            "min_danceability": 0.5, "max_danceability": 0.8,
            "min_acousticness": 0.0, "max_acousticness": 0.3,
            "min_valence": 0.0, "max_valence": 0.3
        },
        "disgust": {
            "min_tempo": 0.4, "max_tempo": 0.6,
            "min_energy": 0.2, "max_energy": 0.5,
            "min_danceability": 0.3, "max_danceability": 0.5,
            "min_acousticness": 0.4, "max_acousticness": 0.7,
            "min_valence": 0.0, "max_valence": 0.2
        },
        "fear": {
            "min_tempo": 0.5, "max_tempo": 0.7,
            "min_energy": 0.3, "max_energy": 0.6,
            "min_danceability": 0.3, "max_danceability": 0.6,
            "min_acousticness": 0.3, "max_acousticness": 0.6,
            "min_valence": 0.0, "max_valence": 0.2
        },
        "happy": {
            "min_tempo": 0.6, "max_tempo": 1.0,
            "min_energy": 0.6, "max_energy": 1.0,
            "min_danceability": 0.7, "max_danceability": 1.0,
            "min_acousticness": 0.0, "max_acousticness": 0.3,
            "min_valence": 0.7, "max_valence": 1.0
        },
        "sad": {
            "min_tempo": 0.2, "max_tempo": 0.5,
            "min_energy": 0.2, "max_energy": 0.5,
            "min_danceability": 0.2, "max_danceability": 0.4,
            "min_acousticness": 0.5, "max_acousticness": 1.0,
            "min_valence": 0.0, "max_valence": 0.3
        },
        "surprise": {
            "min_tempo": 0.6, "max_tempo": 1.0,
            "min_energy": 0.6, "max_energy": 0.9,
            "min_danceability": 0.6, "max_danceability": 0.9,
            "min_acousticness": 0.2, "max_acousticness": 0.5,
            "min_valence": 0.5, "max_valence": 0.8
        }
    }


def choose_songs(sp, emotion, song_uris):
    selected_songs_uris = []
    random.shuffle(song_uris)
    
    tracks_all_data = sp.audio_features(song_uris)
    emotion_criteria = emotions[emotion]
    for track_data in tracks_all_data:
        if track_data is None:
            continue
        try:
            if (emotion_criteria["min_tempo"] <= track_data["tempo"] <= emotion_criteria["max_tempo"] and
                emotion_criteria["min_energy"] <= track_data["energy"] <= emotion_criteria["max_energy"] and
                emotion_criteria["min_danceability"] <= track_data["danceability"] <= emotion_criteria["max_danceability"] and
                emotion_criteria["min_acousticness"] <= track_data["acousticness"] <= emotion_criteria["max_acousticness"] and
                emotion_criteria["min_valence"] <= track_data["valence"] <= emotion_criteria["max_valence"]):
                selected_songs_uris.append(track_data["uri"])
                if len(selected_songs_uris) == 10:
                    break
        except TypeError:
            continue
    
    return selected_songs_uris
    pass

sp_oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )

sp = spotipy.Spotify(auth_manager=sp_oauth)
top_artists = get_top_artists(sp)
songs = get_songs(sp, top_artists)
selected_songs = choose_songs(sp, "happy", songs)
print(selected_songs)
create_playlist(sp, selected_songs, "happy")






#------------------------Emotion-------------------------#

pipe = pipeline("image-classification", model="motheecreator/vit-Facial-Expression-Recognition")
streaming = False
threshold = 0.5
mtcnn = MTCNN(keep_all=True, post_process=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval()




def input(image, thresh=None):
    image = Image.open(image)
    image = image.convert("RGB")
    boxes, probabilities = mtcnn.detect(image)
    draw = ImageDraw.Draw(image)
    font_path = "arial.ttf"  # Path to a .ttf file
    font_size = 30  # Specify the font size
    font = ImageFont.truetype(font_path, font_size)

    max_index = int(list(probabilities).index(max(list(probabilities))))
    
    if boxes is not None:
        box = boxes[max_index]
        cropped_image = image.crop(box)
        image_resized = cropped_image.resize((224, 224))
        prediction = pipe(image_resized)
        if prediction[0]['label'] == "neutral":
            prediction_ = prediction[1]['label']
        else:
            prediction_ = prediction[0]['label']

        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=3)
        left, top, right, bottom = draw.textbbox((box[0], box[1] - 10), text = prediction_, font=font)
        draw.rectangle((left-5, top-5, right+5, bottom+5), fill="white")
        draw.text((box[0], box[1] - 10), text=prediction_, font=font, fill="black")

        return image, prediction_
    
    else:
        return "No Boxes"


