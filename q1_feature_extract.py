import os
# ========== 新增：HF国内镜像，解决huggingface网络超时WinError10060 ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import pandas as pd
import subprocess
import librosa
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置区（这里全部改成r原始字符串，避免路径报错）=====================
# ffmpeg路径
FFMPEG_PATH = r'D:\liulanqi\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'
# 视频根目录：子文件夹存放视频
VIDEO_ROOT = r'C:\Users\Qsz53\Desktop\jianmo\fujian1\shipin'
# 标签Excel路径
LABEL_PATH = r'C:\Users\Qsz53\Desktop\jianmo\label.xlsx'
# 输出pkl保存路径
OUTPUT_PKL = r'C:\Users\Qsz53\Desktop\jianmo\feature_result.pkl'

# ===================== 读取标签文件 =====================
df_label = pd.read_excel(LABEL_PATH)
print(f"读取标签表成功，一共 {len(df_label)} 条样本")

# ===================== 加载BERT文本模型（现在走国内镜像hf-mirror） =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()

# ===================== 特征提取函数 =====================
def extract_text_feature(text):
    """文本BERT特征"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=64).to(device)
        out = bert_model(**inputs)
        cls_feat = out.last_hidden_state[:,0,:]
    return cls_feat.cpu().numpy()[0]

def extract_audio_feature(video_path, tmp_wav="tmp.wav"):
    """使用ffmpeg提取音频，librosa提取声学特征"""
    # ffmpeg把视频转临时wav音频
    cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", tmp_wav]
    subprocess.run(cmd, capture_output=True, check=False)
    # librosa读取音频
    y, sr = librosa.load(tmp_wav, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feat = np.mean(mfcc, axis=1)
    # 删除临时音频
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
    return feat

def extract_visual_feature(video_path, sample_frame=5):
    """视觉特征：采样多帧，取平均灰度特征，可后续替换为预训练CNN"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx_list = np.linspace(0, max(0, total_frames-1), sample_frame, dtype=int)
    frame_feats = []
    for fid in frame_idx_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64,64))
        frame_feats.append(gray.flatten())
    cap.release()
    if len(frame_feats)==0:
        return np.zeros(64*64)
    return np.mean(frame_feats, axis=0)

# ===================== 遍历所有视频，提取三模态特征 =====================
result_list = []
for idx, row in df_label.iterrows():
    video_name = row['video_name']
    text_content = row['text']
    emotion_label = row['label']
    # 在子文件夹中搜索视频文件
    video_file = None
    for subdir, _, files in os.walk(VIDEO_ROOT):
        for f in files:
            if f == video_name:
                video_file = os.path.join(subdir, f)
                break
        if video_file is not None:
            break
    if video_file is None:
        print(f"警告：找不到视频 {video_name}，跳过")
        continue
    print(f"正在处理：{video_name}")
    # 三模态提取
    text_feat = extract_text_feature(text_content)
    audio_feat = extract_audio_feature(video_file)
    visual_feat = extract_visual_feature(video_file)
    result_list.append({
        "video_name": video_name,
        "text_feature": text_feat,
        "audio_feature": audio_feat,
        "visual_feature": visual_feat,
        "emotion_label": emotion_label
    })

# ===================== 保存结果到pkl文件 =====================
import pickle
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(result_list, f)
print(f"\n全部处理完成！特征已保存到 {OUTPUT_PKL}")
print(f"成功提取样本数量：{len(result_list)}")
