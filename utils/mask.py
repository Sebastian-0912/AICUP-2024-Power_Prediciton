import torch
def generate_tgt_mask(size):
    """
    Generate a square mask for the sequence to prevent the model from seeing future positions.
    """
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask