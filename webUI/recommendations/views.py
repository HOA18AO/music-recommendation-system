from django.shortcuts import render
from django.http import JsonResponse
from sklearn.metrics.pairwise import cosine_similarity # tính cosine similarity
import json
import pandas as pd
import numpy as np
import requests
import torchaudio # đọc dữ liệu từ preview_url
from io import BytesIO
import soundfile as sf
import librosa

# chuyển đổi kiểu dữ liệu của mean_mfcc (và/hoặc mfccs)
def parse_mfcc(mfcc_string):
    try:
        return np.array(mfcc_string.strip('[]').split(','), dtype=np.float32)
    except:
        return

def load_audio_with_torchaudio(url, target_sr=16000, duration=30): # faster
    response = requests.get(url)
    if response.status_code == 200:
        audio_data = BytesIO(response.content)
        waveform, sr = torchaudio.load(audio_data)
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        waveform = waveform.mean(dim=0)  # Convert to mono
        # print(waveform)
        max_samples = duration * target_sr # đoạn này dùng để làm gì?
        waveform = waveform[:max_samples]
        # print(waveform)
        return waveform.numpy(), target_sr
    else:
        raise Exception("Failed to download audio.")

def get_top_n_recommendations(target_vector, track_mfcc_dict, track_info_dict, top_n=10):
    """
    Lấy top N bài hát tương tự dựa trên cosine similarity và lọc các bài trùng tên và nghệ sĩ.

    Parameters:
        target_vector (np.array): Vector MFCC của bài hát đầu vào.
        track_mfcc_dict (dict): Dictionary {track_id: mfcc_vector}.
        tracks_df (pd.DataFrame): DataFrame chứa thông tin 'track_id', 'track_name', 'artist_name'.
        top_n (int): Số lượng bài hát cần trả về.

    Returns:
        list: Danh sách các track_id của bài hát lọc trùng.
    """
    # Bước 1: Chuyển dữ liệu và tính cosine similarity
    all_vectors = np.array(list(track_mfcc_dict.values()))
    all_ids = list(track_mfcc_dict.keys())
    target_vector = target_vector.reshape(1, -1)

    similarities = cosine_similarity(target_vector, all_vectors).flatten()

    # Bước 2: Sắp xếp theo độ tương tự, loại bỏ các bài hát có cosine similarity = 1
    similar_indices = similarities.argsort()[::-1] # sắp xếp
    candidate_ids = [all_ids[i] for i in similar_indices if similarities[i] < 1] # bỏ giá trị quá khớp

    # Bước 4: Lọc trùng lặp
    seen = set()
    filtered_tracks = []

    for track_id in candidate_ids:
        # Nếu chưa có bài hát nào trong filtered_tracks, thêm bài hát đầu tiên
        if len(filtered_tracks) == 0:
            filtered_tracks.append(track_id)
        # Nếu bài hát hiện tại khác bài hát trước đó về tên bài hát
        elif track_info_dict[track_id]['track_name'] != track_info_dict[filtered_tracks[-1]]['track_name']:
            filtered_tracks.append(track_id)
        # nếu bài hát hiện tại trùng tên với bài hát trước đó, nhưng khác nghệ sĩ
        elif track_info_dict[track_id]['artist_name'] != track_info_dict[filtered_tracks[-1]]['artist_name']:
            filtered_tracks.appen(track_id)
        # các trường hợp còn lại
        else:
            continue # bỏ qua
        # Nếu đủ top_n thì dừng
        if len(filtered_tracks) == top_n:
            break

    return filtered_tracks


# recommendations\datasets\filtered_tracks_reduced.csv
tracks = pd.read_csv('recommendations/datasets/filtered_tracks_reduced.csv') # dữ liệu của các tracks
df = pd.read_csv('recommendations/datasets/mixed_features.csv') # thông tin đặc trưng của các tracks
# Tạo dictionary chứa track_id, track_name, artist_name
track_info_dict = {
    row['track_id']: {'track_name': row['track_name'], 'artist_name': row['artist_name']} 
    for _, row in tracks.iterrows()
}
track_mfcc_dict = {
    df.at[row.Index, 'track_id']: parse_mfcc(df.at[row.Index, 'feature'])
    for row in df.itertuples()
}

def home(request):
    return render(request, 'home.html')

def recommend(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        song_input = data.get('song_input', '')  # Lấy giá trị song_input từ JSON
        # song_input = request.POST.get('song_input', '')
        # bước 1: tách ra giá trị input_track_id
        # ví dụ: https://open.spotify.com/track/7dqTJ7Ba3VtBtGTXHRK8N8
        input_track_id = song_input.split('/')[-1]
        # print(input_track_id)
        
        # bước 2: dùng Spotify API (requests) để lấy phần preview (url), nếu không có giá trị này, chuyển đến bước cuối và trả về giá trị rỗng
        url = "https://spotify-downloader9.p.rapidapi.com/tracks"
        querystring = {"ids":input_track_id}

        headers = {
            "x-rapidapi-key": "ce37fe09ebmsh3fab47deaca200ep10e229jsne809a290fbbd",
            "x-rapidapi-host": "spotify-downloader9.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)
        full_data = response.json()
        # print(full_data)
        # print(full_data.get('success'))
        if full_data.get('success') and 'preview_url' in full_data['data']['tracks'][0]:
            # print(full_data['data']['tracks'][0]['preview_url'])
            input_preview_url = full_data['data']['tracks'][0]['preview_url']
        else:
            print('no preview')
            # Trường hợp không có preview_url hoặc thất bại khi gọi API
            return JsonResponse({'error': 'No preview URL found or API request failed'}, status=400)
        
        # bước 3: Từ phần preview_url, tách ra giá trị MFCC (n_mfcc = 20), trích xuất đặc trưng trung bình, STD, max, min từ ma trận MFCC
        signal, sr = load_audio_with_torchaudio(input_preview_url)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20, n_fft=4096, hop_length=2048)  # 2D array
        
        # Tính các đặc trưng: mean, std, min, max, range cho từng vector MFCC
        means = np.round(np.mean(mfcc, axis=1), 4)
        stds = np.round(np.std(mfcc, axis=1), 4)
        maxs = np.max(mfcc, axis=1)
        mins = np.min(mfcc, axis=1)
        
        # print(
        #     f'means : {means}\n',
        #     f'stds : {stds}\n',
        #     f'maxs : {maxs}\n',
        #     f'mins : {mins}\n'
        # )
        
        input_feature_vector = np.concatenate([means, stds, maxs, mins])
        
        # bước 4: Tính tương đồng cosine giữa vector đặc trưng thu được với tập dữ liệu trong thư mục '/datasets', lấy ra top 10
        recommendation_ids = get_top_n_recommendations(input_feature_vector, track_mfcc_dict, track_info_dict, top_n=10)
        # print(recommendation_ids)
        
        # bước 5: Tìm các track_id trong bảng tracks (filtered_tracks_reduced.csv)
        # tracks.columns = ['track_name', 'artist_name', 'track_id',...] lấy dữ liệu của các cột này
        # truyền vào biến recommended_tracks
        recommended_tracks = tracks[tracks['track_id'].isin(recommendation_ids)].drop_duplicates(subset='track_id')
        recommended_tracks = recommended_tracks[['track_name', 'artist_name', 'preview_url']]
        
        # print(song_input)
        recommendations = recommended_tracks.to_dict(orient='records')
        # print(recommendations)
        return JsonResponse({'recommendations': recommendations})
    return JsonResponse({'error': 'Invalid request'}, status=400)