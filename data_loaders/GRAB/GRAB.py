import torch
import numpy as np
import os
import utils.rotation_conversions as geometry

SAMPLING = "RANDOM_SEGMENT"
DATA_DEBUG = False

class GRAB(torch.utils.data.Dataset):
    def __init__(self, num_frames = 60, datapath='./dataset/GRAB_SAGA', split="train", num_joints = 53, inference = False):
        super().__init__()

        assert split in ['train', 'test', 'val']

        self.num_joints = num_joints
        self.num_frames = num_frames
        self.split = split
        self.inference = inference
        self.data_path = os.path.join(datapath, split)
        self.ds = sorted(os.listdir(self.data_path))
        self.dataname = "GRAB"
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):

        # Loading file
        sequence = torch.from_numpy(np.load(os.path.join(self.data_path, self.ds[idx])))

        # Sampling Policy:
        if SAMPLING == "RANDOM_SEGMENT":
            start_frame = np.random.randint(low = 0, high = sequence.shape[0] - self.num_frames)
        elif SAMPLING == "START":
            start_frame = 0
        elif SAMPLING == "END":
            start_frame = sequence.shape[0] - self.num_frames
        elif SAMPLING == "MID":
            start_frame = min(0, sequence.shape[0]//2 - self.num_frames//2)
        
        # Reshape to (frames, joints, features)
        sequence = sequence[start_frame:start_frame+self.num_frames].reshape(self.num_frames, self.num_joints, 3)

        # Converting to rot6d
        clip = geometry.axis_angle_to_matrix(sequence)
        clip = geometry.matrix_to_rotation_6d(clip)

        # Duplicating and setting translation
        clip[:, -1, :3] = sequence[:, -1]
        clip[:, -1, 3:] = sequence[:, -1]

        # Reshape to (joints, features, frames)
        clip = clip.permute(1, 2, 0)

        # TODO: Reduce the size to [51, 6, 60] by deleting the first and last entry!
        # clip = clip[1:-1]

        # DEBUG
        if DATA_DEBUG:
            print(f'File: {self.ds[idx]}')
            print(f'Start Frame: {start_frame}')

        assert clip.shape[2] == self.num_frames and clip.shape[1] == 6

        if self.inference:
            return clip, self.ds[idx], start_frame
        else:
            return clip, {}


if __name__ == "__main__":
    grab = GRAB(split="val")

    print(len(grab))

    for i in grab:
        print(i[0].shape)