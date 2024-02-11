import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
import os
sys.path.append(os.getcwd())
from config.eval.cifar10 import get_config as get_eval_config
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import lib.utils.utils as utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils
from PIL import Image

save_samples_path = 'path/to/tauLDR_samples'

eval_cfg = get_eval_config()
train_cfg = bookkeeping.load_ml_collections(Path(eval_cfg.train_config_path))

for item in eval_cfg.train_config_overrides:
    utils.set_in_nested_dict(train_cfg, item[0], item[1])

S = train_cfg.data.S
device = torch.device(eval_cfg.device)

model = model_utils.create_model(train_cfg, device)

loaded_state = torch.load(Path(eval_cfg.checkpoint_path),
    map_location=device)

modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
model.load_state_dict(modified_model_state)

model.eval()

def imgtrans(x):
    x = np.transpose(x, (1,2,0))
    return x

total_samples = 0
batch = 50
sampler = sampling_utils.get_sampler(eval_cfg)
while True:
    print(total_samples)
    samples, _, _ = sampler.sample(model, batch, 1)
    samples = samples.reshape(batch, 3, 32, 32)
    samples_uint8 = samples.astype(np.uint8)
    for i in range(samples.shape[0]):
        path_to_save = save_samples_path + f'/{total_samples + i}.png'
        img = Image.fromarray(imgtrans(samples_uint8[i]))
        img.save(path_to_save)


    total_samples += batch
    if total_samples >= 50000:
        break
