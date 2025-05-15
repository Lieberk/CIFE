import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class CIFE_Dataset(Dataset):
    def __init__(self, path_vid, config):
        with open(os.path.join(config['data_path'], 'data.json'), 'r') as f:
            self.data_complete = json.load(f)

        self.news_id = []
        with open(os.path.join(config['data_path'], 'data-split/', path_vid), "r") as fr:
            for line in fr.readlines():
                self.news_id.append(line.strip())
        self.data = [item for item in self.data_complete if str(item['video_id']) in self.news_id]
        self.text_fea_path = os.path.join(config['data_path'], 'preprocess/pre_CIFE/bart_text_feature/')
        self.IVOD_data_path = os.path.join(config['data_path'], 'preprocess/pre_CIFE/IVOD_feature/')
        self.IEAM_data_path = os.path.join(config['data_path'], 'preprocess/pre_CIFE/IEAM_feature/')
        self.frame_fea_path = os.path.join(config['data_path'], 'clip_vit_feature/')
        self.model_adj_path = os.path.join(config['data_path'], 'preprocess/pre_CIFE/model_adj/')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = item['video_id']

        title_embed = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['title_embed']
        title_mask = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['title_mask']
        title_embed = torch.FloatTensor(title_embed)
        title_mask = torch.FloatTensor(title_mask)

        audio_transcript_embed = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['audio_transcript_embed']
        audio_transcript_mask = pickle.load(open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['audio_transcript_mask']
        audio_transcript_embed = torch.FloatTensor(audio_transcript_embed)
        audio_transcript_mask = torch.FloatTensor(audio_transcript_mask)

        frames = pickle.load(open(os.path.join(self.frame_fea_path, vid + '.pkl'), 'rb'))
        frames = torch.FloatTensor(frames)

        ivod_feature = pickle.load(open(os.path.join(self.IVOD_data_path, vid + '.pkl'), 'rb'))['ivod_feature']
        ivod_mask = pickle.load(open(os.path.join(self.IVOD_data_path, vid + '.pkl'), 'rb'))['ivod_mask']
        ivod_feature = torch.FloatTensor(ivod_feature)
        ivod_mask = torch.FloatTensor(ivod_mask)

        ieam_feature = pickle.load(open(os.path.join(self.IEAM_data_path, vid + '.pkl'), 'rb'))['ieam_feature']
        ieam_mask = pickle.load(open(os.path.join(self.IEAM_data_path, vid + '.pkl'), 'rb'))['ieam_mask']
        ieam_feature = torch.FloatTensor(ieam_feature)
        ieam_mask = torch.FloatTensor(ieam_mask)

        model_adj = pickle.load(open(os.path.join(self.model_adj_path, vid + '.pkl'), 'rb'))
        model_adj = torch.FloatTensor(model_adj)

        target_ids = pickle.load(
            open(os.path.join(self.text_fea_path, vid + '.pkl'), 'rb'))['target_inputid']
        target_ids = torch.LongTensor(target_ids)

        return {
            'title': title_embed,
            'title_mask': title_mask,
            'audio_transcript': audio_transcript_embed,
            'audio_transcript_mask': audio_transcript_mask,
            'ivod_feature': ivod_feature,
            'ivod_mask': ivod_mask,
            'ieam_feature': ieam_feature,
            'ieam_mask': ieam_mask,
            'target_ids': target_ids,
            'frames': frames,
            'model_adj': model_adj,
        }


def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            video = video[:seq_len]
            mask = np.ones(seq_len)
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.FloatTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def CIFE_collate_fn(batch):
    title = [item['title'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    audio_transcript = [item['audio_transcript'] for item in batch]
    audio_transcript_mask = [item['audio_transcript_mask'] for item in batch]

    ivod_feature = [item['ivod_feature'] for item in batch]
    ivod_mask = [item['ivod_mask'] for item in batch]

    ieam_feature = [item['ieam_feature'] for item in batch]
    ieam_mask = [item['ieam_mask'] for item in batch]

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(55, frames)

    target_ids = [item['target_ids'] for item in batch]
    model_adj = [item['model_adj'] for item in batch]

    return {
        'title': torch.stack(title),
        'title_mask': torch.stack(title_mask),
        'audio_transcript': torch.stack(audio_transcript),
        'audio_transcript_mask': torch.stack(audio_transcript_mask),
        'ivod_feature': torch.stack(ivod_feature),
        'ivod_mask': torch.stack(ivod_mask),
        'ieam_feature': torch.stack(ieam_feature),
        'ieam_mask': torch.stack(ieam_mask),
        'target_ids': torch.stack(target_ids),
        'model_adj': torch.stack(model_adj),
        'frames': frames,
        'frames_masks': frames_masks,
    }