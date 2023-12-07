import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

def get_spotify_features(track_url):
    # Credenciais do Spotify
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

    # Configuração das credenciais
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Obtendo o ID da faixa a partir do URL
    track_id = track_url.split("/")[-1].split("?")[0] 

    # Obtendo as características da faixa
    track_info = sp.track(track_id)
    features = sp.audio_features(track_id)[0]

    # Extraindo as características desejadas
    track_features = {
        'acousticness': features['acousticness'],
        'danceability': features['danceability'],
        'duration_ms': track_info['duration_ms'],
        'instrumentalness': features['instrumentalness'],
        'liveness': features['liveness'],
        'loudness': features['loudness'],
        'mode': features['mode'],
        'speechiness': features['speechiness'],
        'tempo': features['tempo'],
        'popularity': track_info['popularity']
    }

    return track_features

