import argparse
import os
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

from config import GlobalConfig
from model_viz import TransFuser
from data_viz import CARLA_Data


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to model ckpt')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--save_path', type=str, default=None, help='path to save visualizations')
parser.add_argument('--total_size', type=int, default=1000, help='total images for which to generate visualizations')
parser.add_argument('--attn_thres', type=int, default=1, help='minimum # tokens of other modality required for global context')

args = parser.parse_args()

# Config
config = GlobalConfig()

if args.save_path is not None and not os.path.isdir(args.save_path):
	os.makedirs(args.save_path, exist_ok=True)

# Data
viz_data = CARLA_Data(root=config.viz_data, config=config)
dataloader_viz = DataLoader(viz_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
model = TransFuser(config, args.device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total parameters: ', params)

model.load_state_dict(torch.load(os.path.join(args.model_path, 'best_model.pth')))
model.eval()

x = [i for i in range(16, 512, 32)]
y = [i for i in range(16, 256, 32)]
patch_centers = []
for i in x:
	for j in y:
		patch_centers.append((i,j))

cnt = 0

# central tokens in both modalities, adjusted for alignment mismatch
central_image_tokens = list(range(16,40))
central_lidar_tokens = list(range(4,64,8))+list(range(6,64,8))+list(range(5,64,8))
global_context = [[], [], [], []]

with torch.no_grad():
	for enum, data in enumerate(tqdm(dataloader_viz)):
		
		if enum*args.batch_size >= args.total_size: # total images for which to generate visualizations
			break
		
		# create batch and move to GPU
		fronts_in = data['fronts']
		lidars_in = data['lidars']
		fronts = []
		bevs = []
		lidars = []
		for i in range(config.seq_len):
			fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
			lidars.append(lidars_in[i].to(args.device, dtype=torch.float32))

		# driving labels
		command = data['command'].to(args.device)
		gt_velocity = data['velocity'].to(args.device, dtype=torch.float32)

		# target point
		target_point = torch.stack(data['target_point'], dim=1).to(args.device, dtype=torch.float32)
		
		pred_wp, attn_map = model(fronts, lidars, target_point, gt_velocity)

        # we use 4 attention heads in the model
		attn_map1 = attn_map[:,0,:,:,:].detach().cpu().numpy()
		attn_map2 = attn_map[:,1,:,:,:].detach().cpu().numpy()
		attn_map3 = attn_map[:,2,:,:,:].detach().cpu().numpy()
		attn_map4 = attn_map[:,3,:,:,:].detach().cpu().numpy()

		curr_cnt = 0
		for idx in range(args.batch_size):
			img = np.transpose(data['fronts'][0][idx].numpy(), (1,2,0))
			lidar_bev = (data['lidar_bevs'][0][idx].squeeze(0).numpy()*255).astype(np.uint8)
			lidar_bev = np.stack([lidar_bev]*3, 2)
			combined_img = np.vstack([img, lidar_bev])

			if args.save_path is not None:
				img_path = os.path.join(args.save_path, str(cnt).zfill(5))
				if not os.path.isdir(img_path):
					os.makedirs(img_path, exist_ok=True)
				Image.fromarray(img).save(os.path.join(img_path, 'input_image.png'))
				Image.fromarray(np.rot90(lidar_bev, 1, (1,0))).save(os.path.join(img_path, 'input_lidar.png')) # adjust for alignment mismatch
				
			cnt += 1

			for head in range(4):
				curr_attn = attn_map4[idx,head]
				for token in range(128):
					attn_vector = curr_attn[token]
					attn_indices = np.argpartition(attn_vector, -5)[-5:]

					if token in central_image_tokens:
						if np.sum(attn_indices>=64) >= args.attn_thres:
							global_context[head].append(1)
						else:
							global_context[head].append(0)

					# if token in central_lidar_tokens:
					# 	if np.sum(attn_indices<64) >= args.attn_thres:
					# 		global_context[head].append(1)
					# 	else:
					# 		global_context[head].append(0)

					if (token<64 and (attn_indices>=64).any()) or (token>=64 and (attn_indices<64).any()):
						
						if args.save_path is not None:
							curr_path = os.path.join(img_path, str(token)+'_'+str(head)+'_'+'_'.join(str(xx) for xx in attn_indices))
							if not os.path.isdir(curr_path):
								os.makedirs(curr_path, exist_ok=True)

						tmp_attn = np.zeros((512, 256, 3)).astype(np.uint8)
						row = patch_centers[token][0]
						col = patch_centers[token][1]
						tmp_attn[row-16:row+16, col-16:col+16, :]=1
						cropped_img = combined_img*tmp_attn
						if args.save_path is not None:
							if token<64:
								Image.fromarray(cropped_img[:256,:,:]).save(os.path.join(curr_path, 'source_token_img.png'))
							else:
								Image.fromarray(np.rot90(cropped_img[256:,:,:], 1, (1,0))).save(os.path.join(curr_path, 'source_token_lidar.png'))

						tmp_attn = np.zeros((512, 256, 3)).astype(np.uint8)
						for attn_token in attn_indices:
							row = patch_centers[attn_token][0]
							col = patch_centers[attn_token][1]
							tmp_attn[row-16:row+16, col-16:col+16, :]=1
						cropped_img = combined_img*tmp_attn
						if args.save_path is not None:
							Image.fromarray(cropped_img[:256,:,:]).save(os.path.join(curr_path, 'attended_token_img.png'))
							Image.fromarray(np.rot90(cropped_img[256:,:,:], 1, (1,0))).save(os.path.join(curr_path, 'attended_token_lidar.png'))

						curr_cnt += 1


global_context = np.array(global_context)
global_context = np.sum(global_context, 0)
global_context = global_context>0

valid_tokens = global_context.sum()
valid_percent = valid_tokens/len(global_context)

print (global_context.sum(), len(global_context), valid_percent)