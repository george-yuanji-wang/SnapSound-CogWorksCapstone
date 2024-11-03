    
def get_top_artists(sp, time_range='short_term', limit=1):
    try:
        results = sp.current_user_top_artists(time_range=time_range, limit=limit)
        artists = results['items']
        
        top_artists = []
        for artist in artists:
            artist_info = {
                'name': artist['name'],
                'popularity': artist['popularity'],
                'genres': artist['genres'],
                'spotify_url': artist['external_urls']['spotify']
            }
            top_artists.append(artist_info)
        
        return top_artists
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

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
    





