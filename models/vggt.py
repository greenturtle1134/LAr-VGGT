import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from models.tokenizer import Tokenizer
from models.transformer import Aggregator
from heads.camera_head import CameraHead

def unflatten_tokens(counts, x):
    """
    Unflattening method used for unflattening both all_coords and all_patches
    :param counts: NxS indicating the number of tokens for each event, view pair
    :param x: TxC to unflatten, where T = sum(counts)
    :returns: NxSxPxC result, where P = max(counts)
    """
    N, S = counts.shape
    P = torch.max(counts).item()
    _, C = x.shape

    flat_counts = counts.reshape(-1)
    
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=counts.device),
                        torch.cumsum(flat_counts, dim=0)[:-1]])

    res = torch.zeros((N * S, P, C), dtype=x.dtype, device=x.device)

    for k, count in enumerate(flat_counts):
        if count > 0:
            count = count.item()
            start = starts[k].item()
            res[k, :count] = x[start:start + count]

    return res.reshape(N, S, P, C)
    
class VGGT(nn.Module):

    def __init__(self):
        super().__init__()

        embed_dim = 256  # This is actually calculated from the output dimension of the tokenizer
        
        self.tokenizer = Tokenizer()
        self.aggregator = Aggregator(embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2*embed_dim)
        pass

    def forward(self, patch_counts, all_coords, all_patches):
        all_tokens = self.tokenizer.forward(all_patches)

        pos = unflatten_tokens(patch_counts, all_coords)
        tokens = unflatten_tokens(patch_counts, all_tokens)

        aggregated_tokens_list, patch_start_idx = self.aggregator.forward(tokens, pos)

        predictions = dict()

        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
            predictions["pose_enc_list"] = pose_enc_list

        return predictions, aggregated_tokens_list, patch_start_idx