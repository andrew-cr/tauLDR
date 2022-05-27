import torch
import torch.nn.functional as F
import math 

# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb