import os
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


def run_lama_inpainter(img, mask, lama_model, device="cuda"):
    with torch.no_grad():
        batch = {
            'image': default_collate([img/255.0]).permute(0,3,1,2).float().to("cuda"),
            'mask': default_collate([mask]).permute(0,3,1,2).float().to("cuda"),
            'unpad_to_size': default_collate((img.shape[0], img.shape[1]))}
        batch = lama_model(batch)   
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res
