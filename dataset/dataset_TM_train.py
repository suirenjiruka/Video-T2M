import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
import json
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 't2m':
            self.data_root = './HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.key_dir = './test_video'
        elif dataset_name == 'kit':
            self.data_root = './KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            kinematic_chain = paramUtil.kit_kinematic_chain

        split_file = pjoin(self.data_root, 'train.txt')


        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name)) #[frame, vec_dim] ([frame, 263] in HumanML3D)
                # Read text
                with open(pjoin(self.key_dir, name + '.json'))as f:
                    key_points = json.load(f)
                    for i, k in enumerate(key_points):
                        if(np.shape(k) != (17,2)):
                            key_points[i] = np.zeros((17,2))
                    key_points = np.array(key_points)
                    k_shape = key_points.shape
                    key_points = key_points.reshape(k_shape[0], k_shape[1] * k_shape[2]) #[frame, joints* 2 (34)]
                    if(k_shape[0]<180):
                         key_points = np.concatenate([key_points, np.zeros(((180 - k_shape[0]), k_shape[1] * k_shape[2]))], axis=0)
                    else:
                        key_points = key_points[:180]

                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text':[text_dict],
                                                       'key_points': key_points}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text':text_data,
                                       'key_points': key_points}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list, key_point = data['m_token_list'], data['text'], data['key_points']
        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption= text_data['caption']

        
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if m_tokens_len+1 < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        #print(caption, m_tokens.reshape(-1).shape, np.array(key_point).shape,  m_tokens_len)

        return caption, m_tokens.reshape(-1), key_point,  m_tokens_len




def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers = 2) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


