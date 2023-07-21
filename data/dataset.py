import os
import sys
import numpy as np
from torch.utils.data import Dataset
from .preprocess import load_pcd, disturb_data

class MSRAction3D(Dataset):
    def __init__(
        self,
        root,
        frames_per_clip=16,
        frame_interval=1,
        num_points=2048,
        train=True,
    ):
        super(MSRAction3D, self).__init__()

        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if (
                train
                and (int(video_name.split("_")[1].split("s")[1]) <= 8)
            ) or (
                not train
                and (int(video_name.split("_")[1].split("s")[1]) > 8)
            ):
                video = load_pcd(root, video_name[:video_name.rfind('_')])
                self.videos.append(video)
                label = int(video_name.split("_")[0][1:]) - 1
                self.labels.append(label)

                nframes = len(video)
                self.index_map.extend(
                    (index, t)
                    for t in range(
                        nframes
                        - frame_interval * (frames_per_clip - 1)
                    )
                )
                index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        clip = [video[t+i*self.frame_interval] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        if self.train:
            # scale the points
            clip=disturb_data(clip)

        clip = clip / 300

        return clip.astype(np.float32), label

class HOI4D(Dataset):
    def __init__(
        self,
        root,
        frames_per_clip=16,
        frame_interval=1,
        num_points=2048,
        train=True,
    ):
        super(HOI4D, self).__init__()
        #temp_data_list = []
        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if (
                train
                and (video_name.split("_")[0] == 'right')
                and (int(video_name.split("_")[1].split("Y")[1]) <= 20210800001)
                and (int(video_name.split("_")[2][1:])== 1 )
                
            ) or (
                not train
                and (video_name.split("_")[0] == 'right')
                and (int(video_name.split("_")[1].split("Y")[1]) > 20210800001) and (int(video_name.split("_")[1].split("Y")[1]) <= 20210800002)
                and (int(video_name.split("_")[2][1:]) == 1 )
                
            ):
                video = load_pcd(root, video_name[:video_name.rfind('_')])
                self.videos.append(video)
                label = int(video_name.split("_")[7][1:]) #T1---T6
                self.labels.append(label)

                nframes = len(video)
                self.index_map.extend(
                    (index, t)
                    for t in range(
                        nframes
                        - frame_interval * (frames_per_clip - 1)
                    )
                )
                index += 1

        self.frames_per_clip = frames_per_clip
        self.frame_interval = frame_interval
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        clip = [video[t+i*self.frame_interval] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        if self.train:
            # scale the points
            clip=disturb_data(clip)

        # clip = clip / 300

        return clip.astype(np.float32), label
