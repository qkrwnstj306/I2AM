import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img

from hook import CrossAttentionHook
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.attention import CrossAttention, SpatialTransformer, BasicTransformerBlock, MemoryEfficientCrossAttention
from cldm.warping_cldm_network import CustomBasicTransformerBlock, CustomSpatialTransformer
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./DATA/zalando-hd-resized")
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./samples")

    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    
    parser.add_argument('--only_one_data', type=str2bool, default=False) 
    parser.add_argument('--certain_data_idx', type=str, default="00273_00.jpg") 
    
    # For hook
    """
    R2G를 보려면 generated_image = True | controlnet, new_attention_map = False
    G2R을 보려면 generated_image = False | controlnet = True (G2R SRAM을 보려면 new_attention_map도 True로 수정)
    """
    parser.add_argument('--generated_image', type=str2bool, default=True) # True: R2G, False: G2R 
    parser.add_argument('--per_time_step', type=str2bool, default=False) # if True, TLAM
    parser.add_argument('--per_attention_head', type=str2bool, default=False) # if True, HLAM
    
    parser.add_argument('--controlnet', type=str2bool, default=False) # if you visualize Speicifc-Reference Attribution Maps, set to True
    
    # SRAM을 보려면 per_time_step, per_attention_head는 False로 설정해야 한다.
    parser.add_argument('--new_attention_map', type=str2bool, default=False) # if you visualize Speicifc-Reference Attribution Maps, set to True
    
    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params

    model = create_model(config_path=None, config=config)
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp

    model.load_state_dict(load_cp)
    model = model.cuda()
    model.eval()
    
    #sampler = PLMSSampler(model)
    sampler = DDIMSampler(model)
    
    """ For Debugging"""
    hook = CrossAttentionHook(args.generated_image, args.per_time_step, args.per_attention_head, args.controlnet, args.new_attention_map)
    hook.take_module(sampler.model.model.diffusion_model)
    
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        is_paired=not args.unpair,
        is_test=True,
        is_sorted=True,
        only_one_data = args.only_one_data,
        certain_data_idx = args.certain_data_idx,
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8) 
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    attention_save_dir = os.path.join(save_dir, "attention")
    IMACS = 0.
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(attention_save_dir, exist_ok=True)
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        z, c = model.get_input(batch, params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = model.q_sample(z, ts)     

        samples, _, _ = sampler.sample(
            args.denoise_steps,
            bs,
            shape, 
            c,
            x_T=start_code,
            verbose=False,
            eta=args.eta,
            unconditional_conditioning=uc_full,
        )

        x_samples = model.decode_first_stage(samples)
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            if args.repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
                repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img)

            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])

    
        """ For Debugging"""
        input_image = torch.clamp(x_samples, min=-1.0, max=1.0).cpu()
        cloth = batch["cloth"].cpu()
        agn_mask = batch["agn_mask"].cpu()
        cloth_mask = batch["cloth_mask"].cpu()
        
        attention_maps = hook.make_attention_maps(agn_mask)
        
        if hook.controlnet and hook.new_attention_map:
            for idx in range(len(attention_maps)):
                hook.new_make_images(input_image, cloth, attention_maps[idx], attention_save_dir, f"{batch_idx}-{idx}")
            
        else:
            if hook.per_time_step:
                for idx, key in enumerate(attention_maps.keys(),1):
                    hook.make_images(input_image, cloth, attention_maps[key], attention_save_dir, f"{batch_idx}-time-{idx}", cloth_mask=cloth_mask) 
            elif hook.per_attention_head:
                for idx, attention_map in enumerate(attention_maps,1):
                    hook.make_images(input_image, cloth, attention_map, attention_save_dir, f"{batch_idx}-head-{idx}", cloth_mask=cloth_mask)
                
            else:
                if hook.cal_IMACS_scores:
                    IMACS += hook.make_images(input_image, cloth, attention_maps, attention_save_dir, f"{batch_idx}", agn_mask, cloth_mask)    
                else:
                    hook.make_images(input_image, cloth, attention_maps, attention_save_dir, f"{batch_idx}", agn_mask, cloth_mask)
        
        print(IMACS)
        
    print(f"TOTAL SCORES: {IMACS/len(dataloader)}")
      
if __name__ == "__main__":
    args = build_args()
    main(args)
