from data_loaders.get_data import get_dataset
from utils.model_util import create_model_and_diffusion
from model.mdm import MDM
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import utils.rotation_conversions as geometry

from argparse import Namespace

# Recreate the namespace object
args = Namespace(
    arch='trans_enc',
    batch_size=64,
    cond_mask_prob=0.0,
    cuda=True,
    data_dir='',
    dataset='GRAB',
    device=0,
    diffusion_steps=1000,
    emb_trans_dec=False,
    eval_batch_size=32,
    eval_during_training=False,
    eval_num_samples=1000,
    eval_rep_times=3,
    eval_split='test',
    lambda_fc=0.0,
    lambda_rcxyz=0.0,
    lambda_vel=0.0,
    latent_dim=512,
    layers=8,
    log_interval=1000,
    lr=0.0001,
    lr_anneal_steps=0,
    noise_schedule='cosine',
    num_frames=60,
    num_steps=600000,
    overwrite=False,
    resume_checkpoint='',
    save_dir='save/test',
    save_interval=50000,
    seed=10,
    sigma_small=True,
    train_platform_type='NoPlatform',
    unconstrained=True,
    weight_decay=0.0,
)

def predict(model_path):
    # load the training and validation datasets
    train_dataset=get_dataset("GRAB", 60, "train")
    validation_dataset=get_dataset("GRAB", 60, "val")

    # create the gaussian diffuion model using the same parameters as in training
    model, diffusion = create_model_and_diffusion(args, DataLoader(train_dataset))
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # get the first sequence from each dataset, and a batch dimension
    train_sample_gt=train_dataset[0][0].unsqueeze(0)
    val_sample_gt=validation_dataset[0][0].unsqueeze(0)

    # TODO: remove this once we train in 53 dims
    if train_sample_gt.size(1)==53:
        train_sample_gt=train_sample_gt[:,1:-1,:,:].clone()
        val_sample_gt=val_sample_gt[:,1:-1,:,:].clone()

    # maximum diffusion
    t=999
    t_tensor=torch.tensor(t).unsqueeze(0)

    # diffuse the sequences
    noise=torch.randn_like(train_sample_gt)
    train_sample_diffused=diffusion.q_sample(train_sample_gt, t=t_tensor, noise=noise)
    val_sample_diffused=diffusion.q_sample(val_sample_gt, t=t_tensor, noise=noise)

    # replace the first and last frames with the ground truth as in training
    train_sample_diffused[:, :, :, 0] = train_sample_gt[:, :, :, 0].clone()
    train_sample_diffused[:, :, :, -1] = train_sample_gt[:, :, :, -1].clone()

    val_sample_diffused[:, :, :, 0] = val_sample_gt[:, :, :, 0].clone()
    val_sample_diffused[:, :, :, -1] = val_sample_gt[:, :, :, -1].clone()
    
    # pure noise for all but the first frame
    random_sample = torch.randn_like(train_sample_gt)
    random_sample[:, :, :, 0] = train_sample_gt[:, :, :, 0].clone()
    random_sample_diffused=random_sample

    skip=3

    for t in tqdm(range(999,0,-skip)):

        t_tensor=torch.tensor(t).unsqueeze(0)

        train=model(train_sample_diffused, t_tensor)
        val=model(val_sample_diffused, t_tensor)
        random_sample=model(random_sample_diffused, t_tensor)

        # replace the first and last frames with the ground truth as in training
        train[:, :, :, 0] = train_sample_gt[:, :, :, 0].clone()
        train[:, :, :, -1] = train_sample_gt[:, :, :, -1].clone()

        val[:, :, :, 0] = val_sample_gt[:, :, :, 0].clone()
        val[:, :, :, -1] = val_sample_gt[:, :, :, -1].clone()
        
        random_sample[:, :, :, 0] = train_sample_gt[:, :, :, 0].clone()

        noise=torch.randn_like(train_sample_gt)

        t_minus = torch.tensor(t-skip).unsqueeze(0)
        train_sample_diffused=diffusion.q_sample(train, t=t_minus, noise=noise)
        val_sample_diffused=diffusion.q_sample(val, t=t_minus, noise=noise)
        random_sample_diffused=diffusion.q_sample(random_sample, t=t_minus, noise=noise)

        if t<=skip:
             # get the output from training
            train_output=train.squeeze(0)
            val_output=val.squeeze(0)
            random_sample_output=random_sample.squeeze(0)
    
    return (train_output, val_output, random_sample_output), (train_dataset[0][1], validation_dataset[0][1]), (train_dataset[0][2], validation_dataset[0][2])
   

if __name__ == "__main__":
    (train, val, rand), (train_name, val_name), (train_start, val_start)=predict("/cluster/courses/digital_humans/datasets/team_1/motion-diffusion-model/save/midterm/model000034451.pt")
    train = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(train.permute(2, 0, 1))).detach().cpu().numpy()
    print(train.shape)
    print(train_name)
    print(train_start)
    np.save(f'./predicts/{train_name[:-8]}_{train_start}.npy', train)
    val = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(val.permute(2, 0, 1))).detach().cpu().numpy()
    print(val.shape)
    print(val_name)
    print(val_start)
    np.save(f'./predicts/{val_name[:-8]}_{val_start}.npy', val)
    rand = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(rand.permute(2, 0, 1))).detach().cpu().numpy()
    print(rand.shape)
    np.save(f'./predicts/rand.npy', rand)