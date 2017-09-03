import torch

def numpy2torch(data):
    return torch.from_numpy(data)

def to_cuda(data):
    if torch.cuda.is_available():
        return data.cuda()

def to_np(data):
    return data.data.cpu().numpy()
