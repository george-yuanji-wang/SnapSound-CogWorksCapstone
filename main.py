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
import urllib.parse
import requests
import secrets
from flask import Flask, render_template, redirect, request, jsonify, session, url_for
import datetime


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

# def create_gradio_interface():
#     with gr.Blocks(theme='Taithrah/Minimal') as demo:
#         with gr.Tabs():
#             with gr.TabItem("Emotion Detect"):
#                 with gr.Row():
#                     input_image = gr.Image(type="filepath", label="Input Image")
#                     output_image = gr.Image(type="pil", label="Processed Image")

#                 prediction = gr.Textbox()
#                 print(prediction)
    
#                 threshold_slider = gr.Slider(minimum=0.1, maximum=1, value=0.35, step=0.05, label="Threshold value")
#                 submit_button_classify = gr.Button("Classify Emotion")
#                 if input_image is not None:
#                     submit_button_classify.click(input, inputs=[input_image, threshold_slider], outputs=[output_image, prediction])
#                 else:
#                     print("This will not be printed since 'value' exists")
#     return demo



# app = Flask(__name__)
# app.secret_key = '84u650p3-273c-9347-1f8503662789'


client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
redirect_uri = os.getenv("REDIRECT_URI")

# AUTH_URL = 'https://accounts.spotify.com/authorize'
# TOKEN_URL = 'https://accounts.spotify.com/token'
# API_BASE_URL = 'https://api.spotify.com/v1/'

# # gradio_interface = create_gradio_interface()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/login')
# def login():
#     scope = 'user-read-private user-read-email'

#     params = {
#         'client_id': client_id,
#         'response_type': 'code',
#         'scope': scope,
#         'redirect_uri': redirect_uri,
#         'show_dialog': True
#     }

#     auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

#     return redirect(auth_url)

# @app.route('/callback')
# def callback():
#     if 'error' in request.args:
#         return jsonify({"error": request.args['error']})

#     if 'code' in request.args:
#         req_body = {
#             'code': request.args['code'],
#             'grant_type': 'authorization_code',
#             'redirect_uri': redirect_uri,
#             'client_id': client_id,
#             'client_secret': client_secret,
#         }

#         response = requests.post(TOKEN_URL, data=req_body)
#         token_info = response.json()

#         session['access_token'] = token_info['access_token']
#         session['refresh_token'] = token_info['refresh_token']
#         session['expires_at'] = datetime.now().timestamp() + token_info['expires_in']

#         return redirect(url_for('gradio_interface'))


# @app.route('/gradio', methods=['GET', 'POST'])
# def gradio_interface():
#     if 'access_token' not in session:
#         return redirect(url_for('login'))
    
#     if request.method == 'GET':
#         return gradio_interface.render()
#     elif request.method == 'POST':
#         return gradio_interface.process_api(request.json)

# @app.route('/refresh-token')
# def refresh_token():
#     if 'refresh_token' not in session:
#         return redirect('/login')
    
#     if datetime.now().timestamp() > session['expires_at']:
#         req_body = {
#             'grant_type': 'refresh_token',
#             'refresh_token': session['refresh_token'],
#             'client_id': client_id,
#             'client_secret': client_secret
#         }

#         response = requests.post(TOKEN_URL, data=req_body)
#         new_token_info = response.json()

#         session['access_token'] = new_token_info['access_token']
#         session['expires_at'] = datetime.now().timestamp() + new_token_info['expires_in']

#         return redirect(url_for('gradio_interface'))


# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8080))
#     app.run(host='0.0.0.0', port=port, debug=True)

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

#--------------------------end auth-----------------------------

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

# returns the top tracks of the artist
def get_songs_by_artist(token, artist_id): 
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result
    
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

def get_songs_by_mood(token, mood, limit=10):
    if mood not in emotions:
        raise ValueError(f"Invalid mood: {mood}")

    url = "https://api.spotify.com/v1/recommendations"
    headers = get_auth_header(token)

    # List of possible seed genres
    all_genres = ["pop", "rock", "hip-hop", "electronic", "indie", "jazz", "classical", "country", "r-n-b", "latin"]
    
    # Randomly select 5 genres
    seed_genres = random.sample(all_genres, 5)

    # Function to get a random value within a range
    def get_random_in_range(min_val, max_val):
        return random.uniform(min_val, max_val)

    params = {
        "limit": limit,
        "seed_genres": ",".join(seed_genres),
        "target_tempo": get_random_in_range(emotions[mood]["min_tempo"], emotions[mood]["max_tempo"]) * 180,  # Convert to BPM
        "target_energy": get_random_in_range(emotions[mood]["min_energy"], emotions[mood]["max_energy"]),
        "target_danceability": get_random_in_range(emotions[mood]["min_danceability"], emotions[mood]["max_danceability"]),
        "target_acousticness": get_random_in_range(emotions[mood]["min_acousticness"], emotions[mood]["max_acousticness"]),
        "target_valence": get_random_in_range(emotions[mood]["min_valence"], emotions[mood]["max_valence"]),
        "min_tempo": emotions[mood]["min_tempo"] * 180,  # Convert to BPM
        "max_tempo": emotions[mood]["max_tempo"] * 180,  # Convert to BPM
        "min_energy": emotions[mood]["min_energy"],
        "max_energy": emotions[mood]["max_energy"],
        "min_danceability": emotions[mood]["min_danceability"],
        "max_danceability": emotions[mood]["max_danceability"],
        "min_acousticness": emotions[mood]["min_acousticness"],
        "max_acousticness": emotions[mood]["max_acousticness"],
        "min_valence": emotions[mood]["min_valence"],
        "max_valence": emotions[mood]["max_valence"]
    }

    response = get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.text}")

    tracks = response.json()["tracks"]
    return [{"name": track["name"], "artist": track["artists"][0]["name"], "id": track["id"]} for track in tracks]


def create_spotify_embeds(songs):
    embed_html = "<div style='display: flex; flex-wrap: wrap; justify-content: space-around;'>"
    for song in songs:
        embed_html += f"""
        <div style='margin: 10px; width: 300px;'>
            <h3>{song['name']} by {song['artist']}</h3>
            <iframe src="https://open.spotify.com/embed/track/{song['id']}" 
                    width="300" 
                    height="80" 
                    frameborder="0" 
                    allowtransparency="true" 
                    allow="encrypted-media">
            </iframe>
        </div>
        """
    embed_html += "</div>"
    return embed_html


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
    font_path = "capstone_ctrlv/arial.ttf"  # Path to a .ttf file
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
        print(image, prediction_)
        return image, prediction_
    
    else:
        return "No Boxes"


def process_image_and_get_playlist(image, threshold):
    # Process the image and get the emotion
    processed_image, emotion = input(image, threshold)
    
    # Get the Spotify access token
    token = get_token()
    
    # Get songs based on the detected emotion
    try:
        songs = get_songs_by_mood(token, emotion.lower(), limit=10)
        
        # Create Spotify embeds
        playlist_embeds = create_spotify_embeds(songs)
        
        # Create a formatted song list
        song_list = "\n".join([f"{i+1}. {song['name']} by {song['artist']}" for i, song in enumerate(songs)])
        
        return processed_image, emotion, song_list, playlist_embeds
    except Exception as e:
        return processed_image, emotion, f"Error: {str(e)}", None



with gr.Blocks(theme='Taithrah/Minimal') as demo:
    with gr.Tabs():
        with gr.TabItem("Emotion Detect"):
            with gr.Row():
                input_image = gr.Image(type="filepath", label="Input Image")
                output_image = gr.Image(type="pil", label="Processed Image")

            prediction = gr.Textbox(label="Detected Emotion")
            threshold_slider = gr.Slider(minimum=0.1, maximum=1, value=0.35, step=0.05, label="Threshold value")
            submit_button_classify = gr.Button("Classify Emotion and Get Playlist")

            song_list = gr.Textbox(label="Song List")
            playlist_embeds = gr.HTML(label="Spotify Playlist")

            submit_button_classify.click(
                process_image_and_get_playlist,
                inputs=[input_image, threshold_slider],
                outputs=[output_image, prediction, song_list, playlist_embeds]
            )

        with gr.TabItem("Mood Detect"):
            with gr.Row():
                input_image = gr.Image(type="filepath", label="Input Image")

            prediction = gr.Textbox(label="Predicted Mood")
            threshold_slider = gr.Slider(minimum=0.1, maximum=1, value=0.35, step=0.05, label="Threshold value")

demo.launch(share=True)

