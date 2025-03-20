import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Rectangle
import os

from ldm.modules.attention import CrossAttention, SpatialTransformer, BasicTransformerBlock, MemoryEfficientCrossAttention
from cldm.warping_cldm_network import CustomBasicTransformerBlock, CustomSpatialTransformer

class CrossAttentionHook(nn.Module):
    def __init__(self, generated_image, per_time_step, per_attention_head, controlnet, new_attention_map):
        super().__init__()
        
        self.generated_image = generated_image # Controlent == True 일때만, generated_image == False 가 가능하다. 
        self.controlnet = controlnet
        
        self.current_time_step = 0
        self.lst_idx = 0
        
        self.per_time_step = per_time_step
        self.time_division = 5
        self.total_timestep = 50
        self.per_attention_head = per_attention_head
        self.new_attention_map = new_attention_map # Controlnet 이랑 같이 써야 한다. 
        self.height = 0
        self.width = 0
        
        self.cal_IMACS_scores = False
        
        if self.per_time_step:
            self.cross_attention_forward_hooks = defaultdict(lambda: defaultdict(int))
        else:
            self.cross_attention_forward_hooks = defaultdict(lambda:0)
        
        
    def clear(self):
        self.cross_attention_forward_hooks.clear()
        self.current_time_step = 0
        self.lst_idx = 0
    
    def cal_IMACS(self, attention_maps, agn_mask, cloth_mask):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # resized_attention_maps
            #   range: [0,1], size: [512, 384], type: numpy
            #   resized_attention_maps[resized_attention_maps >= 0.6] = 1, 즉, 값들이 현재 plot 을 위해 반전되어있다.
            attention_maps = 1. - attention_maps 
            
             # agn_mask is inverting...
            #   range: [0,1], size: [1, 512, 384, 1], type: torch
            if self.controlnet:
                cloth_mask = cloth_mask.permute(0,3,1,2)
                cloth_mask = cloth_mask.squeeze(0).squeeze(0).cpu().numpy()
                inversion_cloth_mask = 1. - cloth_mask
                
                masked_region_score = (np.sum(attention_maps * cloth_mask)) / (np.sum(cloth_mask) + 1e-8)
                
                non_masked_region_score = (np.sum(attention_maps * inversion_cloth_mask)) / (np.sum(inversion_cloth_mask) + 1e-8)
            else:
                agn_mask = agn_mask.permute(0,3,1,2)
                agn_mask = agn_mask.squeeze(0).squeeze(0).cpu().numpy()
                inversion_agn_mask = agn_mask
                agn_mask = 1. - agn_mask
            
                masked_region_score = (np.sum(attention_maps * agn_mask)) / np.sum(agn_mask)
                
                non_masked_region_score = (np.sum(attention_maps * inversion_agn_mask)) / np.sum(inversion_agn_mask)
            
            penalty = 3.
            IMACS = masked_region_score - penalty * non_masked_region_score
            
            return IMACS 
    
    def new_make_images(self, input_image, cloth, attention_maps, save_dir, batch_idx):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # input_image: [batch_size, 3, 512, 384]
            # attention_maps: [64, 48]
            attention_maps, idx = attention_maps[0], attention_maps[1]
            # range: [0, 1]
            attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min() + 1e-8)
          
            latent_attention_maps = attention_maps.cpu().numpy()
            latent_attention_maps[latent_attention_maps <=0.4] = 0
            plt.imshow(latent_attention_maps, cmap='jet')
            attention_map_filename = f"idx-{batch_idx}_attention_map_generated_image-{self.generated_image}_controlNet-{self.controlnet}.png"
            attetion_map_save_pth = f"{save_dir}/{attention_map_filename}"
            plt.axis('off')
            plt.savefig(attetion_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # [512, 384, 3], range: [-1, 1]
            
            # [1, 3, 512, 384] -> [1, 32, 24, 3]
            input_image = F.interpolate(input_image, size=(self.height, self.width), mode='bicubic').clamp_(min=0).permute(0,2,3,1)
            input_image = input_image[0]
            
            cloth_image = cloth[0]
            input_image = np.uint8(((input_image + 1.0) * 127.5).numpy())
            cloth_image = np.uint8(((cloth_image + 1.0) * 127.5).numpy())
            
            # [512, 384, 3] 

            resized_attention_maps = F.interpolate(attention_maps.cpu().unsqueeze(0).unsqueeze(0), size=(512,384), mode='bicubic')
            resized_attention_maps = ((resized_attention_maps - resized_attention_maps.min()) / (resized_attention_maps.max() - resized_attention_maps.min() + 1e-8)).squeeze(0).squeeze(0).numpy()
            
            resized_attention_maps = 1.0 - resized_attention_maps
            resized_attention_maps[resized_attention_maps >= 0.6] = 1
            resized_attention_maps = cv2.applyColorMap(np.uint8(resized_attention_maps*255), cv2.COLORMAP_JET)
            heat_map = cv2.addWeighted(cloth_image, 0.7, resized_attention_maps, 0.5, 0)

            plt.imshow(input_image)
            plt.gca().add_patch(Rectangle((idx[1]-0.5, 
                                           idx[0]-0.5), 1, 1, edgecolor='red', facecolor='none'))  # 선택한 위치 주변에 빨간색 테두리 네모 추가
            plt.axis('off')
            attention_map_filename = f"idx-{batch_idx}_attention_map_generated_image_controlNet-{self.controlnet}.png"
            attetion_map_save_pth = f"{save_dir}/new/{attention_map_filename}"
            os.makedirs(f"{save_dir}/new", exist_ok=True)
            plt.savefig(attetion_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            plt.imshow(heat_map)
            heat_map_filename = f"idx-{batch_idx}_cloth.png"
            heat_map_save_pth = f"{save_dir}/new/{heat_map_filename}"
            plt.axis('off')
            os.makedirs(f"{save_dir}/new", exist_ok=True)
            plt.savefig(heat_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
    
    
    def make_images(self, input_image, cloth, attention_maps, save_dir, batch_idx, agn_mask=None, cloth_mask=None):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # input_image: [batch_size, 3, 512, 384]
            # attention_maps: [64, 48]
            cloth_mask_for_score = cloth_mask
            #range: [0, 1], cloth mask region 에 대해서만 attention score 시각화. 자꾸 가장자리에 attention 하길래 
            if cloth_mask is not None and self.generated_image == False:
                cloth_mask = F.interpolate(cloth_mask.permute(0,3,1,2), size=(32, 24), mode='bicubic').squeeze(0).squeeze(0)
                attention_maps = attention_maps.cpu() * cloth_mask 
            
            attention_maps = (attention_maps - attention_maps.min()) / (attention_maps.max() - attention_maps.min() + 1e-8)
                
            # [512, 384, 3], range: [-1, 1]
            if self.generated_image:
                input_image = input_image[0]
                input_image = input_image.permute(1,2,0)
            else:
                input_image = cloth[0]
            input_image = np.uint8(((input_image + 1.0) * 127.5).numpy())
            
            # [512, 384, 1] 
            if self.generated_image:
                resized_attention_maps = F.interpolate(attention_maps.cpu().unsqueeze(0).unsqueeze(0), size=(512,384), mode='bicubic')
                
            else:
    
                resized_attention_maps = F.interpolate(attention_maps.cpu().unsqueeze(0).unsqueeze(0), size=(512,384), mode='bicubic')
            resized_attention_maps = ((resized_attention_maps - resized_attention_maps.min()) / (resized_attention_maps.max() - resized_attention_maps.min() + 1e-8)).squeeze(0).squeeze(0).numpy()
            
            if self.controlnet:
                resized_attention_maps = 1.0 - resized_attention_maps
                resized_attention_maps[resized_attention_maps >= 0.6] = 1
                
                if self.cal_IMACS_scores:
                    IMACS = self.cal_IMACS(resized_attention_maps, agn_mask, cloth_mask_for_score)
                    return IMACS
                
                resized_attention_maps = cv2.applyColorMap(np.uint8(resized_attention_maps*255), cv2.COLORMAP_JET)
                heat_map = cv2.addWeighted(input_image, 0.7, resized_attention_maps, 0.5, 0)
            
                attention_maps = F.interpolate(attention_maps.cpu().unsqueeze(0).unsqueeze(0), size=(512,384), mode='bicubic')
                attention_maps = attention_maps.squeeze(0).squeeze(0).cpu().numpy()
                attention_maps[attention_maps <=0.4] = 0

            else: 
                resized_attention_maps[resized_attention_maps >= 0.6] = 1
                
                if self.cal_IMACS_scores:
                    IMACS = self.cal_IMACS(resized_attention_maps, agn_mask, cloth_mask_for_score)
                    return IMACS
                resized_attention_maps = cv2.applyColorMap(np.uint8(resized_attention_maps*255), cv2.COLORMAP_JET)
                heat_map = cv2.addWeighted(input_image, 0.7, resized_attention_maps, 0.5, 0)
                attention_maps = 1.0 - attention_maps
                attention_maps = attention_maps.cpu().numpy()
                attention_maps[attention_maps <= 0.4] = 0
            
            plt.imshow(attention_maps, cmap='jet')
            attention_map_filename = f"idx-{batch_idx}_attention_map_generated_image-{self.generated_image}_controlNet-{self.controlnet}.png"
            attetion_map_save_pth = f"{save_dir}/{attention_map_filename}"
            plt.axis('off')
            plt.savefig(attetion_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            plt.imshow(heat_map)
            heat_map_filename = f"idx-{batch_idx}_heatmap_generated_image-{self.generated_image}_controlNet-{self.controlnet}.png"
            heat_map_save_pth = f"{save_dir}/{heat_map_filename}"
            plt.axis('off')
            plt.savefig(heat_map_save_pth, bbox_inches='tight', pad_inches=0)
            plt.close()
    
    def make_attention_maps(self, agn_mask):
        # 하나로 합치기 
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # [8, seq_len, height, width] * 15
            if self.per_time_step:
                attention_maps = {}
                for time_pack in range(self.time_division):
                    attention_maps[time_pack] = []
            else:
                attention_maps = []
        
            for key in self.cross_attention_forward_hooks.keys():
                if self.per_time_step:
                    for time_pack in range(self.time_division):
                        if self.controlnet:
                            if self.cross_attention_forward_hooks[key][time_pack].size(2) == 32:
                                attention_maps[time_pack].append(F.interpolate(self.cross_attention_forward_hooks[key][time_pack], size=(32, 24), mode='bicubic').clamp_(min=0))
                        else:
                            attention_maps[time_pack].append(F.interpolate(self.cross_attention_forward_hooks[key][time_pack], size=(64, 48), mode='bicubic').clamp_(min=0))
                   
                        # controlnet 이 아닌데에 대해서는 attention.py 에서 time_pack 조절을 다시 해야된다. warping_cldm_network.py 참고
                else:
                    if self.controlnet:
                        # [0, 1, 2, 3, 4, 5, 6, 7, 8] key:3, 4, 5 가 32 resol. (4가 StableVITON 으로 학습된 idx)
                        # self.height = 16
                        # self.width = int(self.height * 3 / 4)
                        # self.key_name = [0,1,2] 
                        # if self.cross_attention_forward_hooks[key].size(2) == self.height and key in self.key_name:
                        #    attention_maps.append(F.interpolate(self.cross_attention_forward_hooks[key], size=(self.height, self.width), mode='bicubic').clamp_(min=0))
                        
                        self.height = 32
                        self.width = int(self.height * 3 / 4)
                        self.key_name = 4
                        if self.cross_attention_forward_hooks[key].size(2) == self.height and key == self.key_name:
                           attention_maps.append(F.interpolate(self.cross_attention_forward_hooks[key], size=(self.height, self.width), mode='bicubic').clamp_(min=0))
                        #attention_maps.append(F.interpolate(self.cross_attention_forward_hooks[key], size=(32, 24), mode='bicubic').clamp_(min=0))
                    else:
                        # self.key_name = 13#[0,14] encoder: [0,5], decoder: [6,14]
                        # if key == self.key_name:
                        # self.height = 16
                        # self.width = int(self.height * 3 / 4)
                        # self.key_name = [6,7,8] 
                        
                        # if self.cross_attention_forward_hooks[key].size(2) == self.height and key in self.key_name:
                        #     attention_maps.append(F.interpolate(self.cross_attention_forward_hooks[key], size=(64, 48), mode='bicubic').clamp_(min=0))
                        attention_maps.append(F.interpolate(self.cross_attention_forward_hooks[key], size=(64, 48), mode='bicubic').clamp_(min=0))
            
            if self.new_attention_map:
                # attention_maps: [8, seq_len, height, width] for 32 resolution 
                # agn_mask is inverting...
                agn_mask = 1. - agn_mask
                # (1, 1, 32, 24) -> (1, 1, (32x24))
                agn_mask = F.interpolate(agn_mask.permute(0,3,1,2), size=(self.height, self.width), mode='nearest').flatten(start_dim=2,end_dim=3)
                # ((32x24))
                agn_mask = agn_mask.squeeze(0).squeeze(0).cuda()
                grid = torch.ones(agn_mask.size()).cuda()
                # ((32 x 24))
                true_region = (agn_mask * grid) != 0
                
                new_attention_maps = []
                for idx in range(true_region.size(0)):
                    if true_region[idx]:
                        selected_idx = (idx // self.width, idx % self.width) 
                        # [8, 1, 64, 48] 
                        new_attention_map = [attention_map[:,idx,:,:] for attention_map in attention_maps]
                        new_attention_map = torch.cat(new_attention_map, dim=0).mean(0)
                        # [64, 48] x Number of True
                        new_attention_maps.append((new_attention_map, selected_idx))
                
                return new_attention_maps
                
            if self.per_time_step:
                if self.generated_image:
                    for time_pack in range(self.time_division):
                        attention_maps[time_pack] = torch.cat(attention_maps[time_pack], dim=0)
                        attention_maps[time_pack] = attention_maps[time_pack].mean(0).mean(0)
                else:
                    for time_pack in range(self.time_division):
                        attention_maps[time_pack] = [attention_map.mean(dim=1) for attention_map in attention_maps[time_pack]]
                        attention_maps[time_pack] = torch.cat(attention_maps[time_pack], dim=0)
                        attention_maps[time_pack] = attention_maps[time_pack].mean(0)
            elif self.per_attention_head:
                attention_maps = torch.cat(attention_maps, dim = 1)
                attention_maps = list(torch.chunk(attention_maps, 8, dim=0))
                for i, maps in enumerate(attention_maps):
                    attention_maps[i] = maps.squeeze(0).mean(0)
            else:
                if self.generated_image:
                    # [num, 64, 48]
                    attention_maps = [attention_map.mean(dim=1) for attention_map in attention_maps]
                    attention_maps = torch.cat(attention_maps, dim=0)
                    attention_maps = attention_maps.mean(0)
                else:
                    attention_maps = [attention_map.mean(dim=1) for attention_map in attention_maps]
                    attention_maps = torch.cat(attention_maps, dim=0)
                    attention_maps = attention_maps.mean(0)
            self.clear()
        
        return attention_maps
    
    def cross_attention_hook(self, module, input, output, name):
        # Get heat maps
        # print(f"Input size: {len(input)}")
        # print(f"Output size: {len(output)}")
        # print(f"Module Name: {name}")
        # x: [batch_size(1), 1, H, W]
        # y: [batch_size(1), 1, 768]
        if self.controlnet:
            x, y = input[0], input[1]
            # [num_heads(8), context_seq_len, height, width]
            self.cross_attention_forward_hooks, self.lst_idx, self.current_time_step = module.get_attention_score(x, y, self.cross_attention_forward_hooks, self.lst_idx,
                                                                            self.current_time_step, generated_image=self.generated_image, per_time_step=self.per_time_step)
        else:
            x, y, h = input[0], input[1], input[2]
            # [num_heads(8), context_seq_len(1), height, width]
            self.cross_attention_forward_hooks, self.lst_idx, self.current_time_step = module.get_attention_score(x, y, h, self.cross_attention_forward_hooks, self.lst_idx,
                                                                            self.current_time_step, generated_image=self.generated_image, per_time_step=self.per_time_step)
            
    def take_module(self, model):
        
        for name, module in model.named_modules():
            if self.controlnet: 
                if isinstance(module, CustomSpatialTransformer) and 'warp_flow_blks' in name:
                    module.register_forward_hook(lambda m, inp, out, n=name: self.cross_attention_hook(m, inp, out, n))
            else:
                if isinstance(module, SpatialTransformer) and not 'middle_block' in name and not 'warp_flow_blks' in name: 
                    module.register_forward_hook(lambda m, inp, out, n=name: self.cross_attention_hook(m, inp, out, n))
                
        