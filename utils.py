import os
import json
import random

from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)

def json_validate(file_path):
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)

    emotion_list = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
    character_list = list(map(str.lower, json_data['character_list'].keys()))
    scene_list = json_data['scene_list']

    typo = 0
    for scene_num, scene_data in scene_list.items():
        for speaker_data in scene_data:
            speaker = speaker_data['speaker'].lower()
            if speaker == 'narrator':
                if speaker not in character_list:
                    typo += 1
            elif speaker not in character_list:
                emotion = speaker_data['emotion'].lower()
                if (speaker not in character_list) or (emotion not in emotion_list):
                    typo += 1

    return typo != 0

def json_preprocessing(file_path, ref_audio_path = 'ref_audio'):
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)

    character_list = json_data['character_list']
    scene_list = json_data['scene_list']

    male_ref_audio = os.listdir(os.path.join(ref_audio_path, 'male'))
    female_ref_audio = os.listdir(os.path.join(ref_audio_path, 'female'))
    random.shuffle(male_ref_audio)
    random.shuffle(female_ref_audio)

    character_ref_audio_path = {}
    for character_name, gender in character_list.items():
        character_name, gender = list(map(str.lower, [character_name, gender]))
        if character_name == 'narrator':
            character_ref_audio_path[character_name] = os.path.join(ref_audio_path, 'narrator', 'male', 'neutral_1.wav')
        else:
            character_ref_audio_path[character_name] = os.path.join(ref_audio_path, gender, locals()[f'{gender}_ref_audio'].pop())

    for scene_num, scene_data in scene_list.items():
        for speaker_num in range(len(scene_data)):
            scene_list[scene_num][speaker_num]['speaker'] = scene_list[scene_num][speaker_num]['speaker'].lower()
            if scene_list[scene_num][speaker_num]['speaker'] != 'narrator':
                scene_list[scene_num][speaker_num]['emotion'] = scene_list[scene_num][speaker_num]['emotion'].lower()

    return character_list, scene_list, character_ref_audio_path