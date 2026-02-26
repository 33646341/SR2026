import numpy as np
import torch
import torch.utils.data
try:
    import piq
except ImportError:
    piq = None
from scipy.stats import entropy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output.item()


def mse(input, target):
    with torch.no_grad():
        loss = nn.MSELoss()
        output = loss(input, target)
    return output.item()


def psnr(input, target):
    with torch.no_grad():
        # input and target are in [-1, 1], convert to [0, 1]
        input = (input + 1.) / 2.
        target = (target + 1.) / 2.
        mse = torch.mean((input - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(input, target, window_size=11, size_average=True):
    # input and target are in [-1, 1], convert to [0, 1]
    input = (input + 1.) / 2.
    target = (target + 1.) / 2.
    
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous().type_as(input))
        return window

    channel = input.size(1)
    window = create_window(window_size, channel)

    mu1 = F.conv2d(input, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(input*input, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target*target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(input*target, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class AdvancedMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {}

    def _get_model(self, name):
        if piq is None:
            raise ImportError("piq is not installed. Please run `uv pip install piq`")
            
        if name not in self.models:
            if name == 'lpips':
                # LPIPS: lower is better
                self.models[name] = piq.LPIPS(replace_pooling=True, reduction='none').to(self.device)
            elif name == 'dists':
                # DISTS: lower is better
                self.models[name] = piq.DISTS(reduction='none').to(self.device)
            elif name == 'clipiqa':
                # CLIPIQA: higher is better
                self.models[name] = piq.CLIPIQA(data_range=1.0).to(self.device)
        return self.models[name]

    def compute(self, input, target=None, metric='psnr'):
        """
        Compute metric.
        input, target: tensor in range [-1, 1] (will be converted to [0, 1])
        """
        # Convert [-1, 1] to [0, 1]
        input = (input + 1.) / 2.
        input = torch.clamp(input, 0, 1).to(self.device)
        
        if target is not None:
            target = (target + 1.) / 2.
            target = torch.clamp(target, 0, 1).to(self.device)

        # Check channel count
        C = input.shape[1]

        if metric == 'fsim':
            if target is None:
                raise ValueError("FSIM requires target")
            
            if C == 1:
                return piq.fsim(input, target, data_range=1.0, reduction='mean', chromatic=False).item()
            elif C == 3:
                return piq.fsim(input, target, data_range=1.0, reduction='mean', chromatic=True).item()
            else:
                # Multichannel: average over channels (treated as grayscale)
                scores = []
                for c in range(C):
                    inp_c = input[:, c:c+1, :, :]
                    tgt_c = target[:, c:c+1, :, :]
                    score = piq.fsim(inp_c, tgt_c, data_range=1.0, reduction='mean', chromatic=False)
                    scores.append(score)
                return torch.stack(scores).mean().item()
            
        elif metric == 'lpips':
            model = self._get_model('lpips')
            with torch.no_grad():
                if C == 3:
                    return model(input, target).mean().item()
                elif C == 1:
                    return model(input.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1)).mean().item()
                else:
                    scores = []
                    for c in range(C):
                        inp_c = input[:, c:c+1, :, :].repeat(1, 3, 1, 1)
                        tgt_c = target[:, c:c+1, :, :].repeat(1, 3, 1, 1)
                        scores.append(model(inp_c, tgt_c).mean())
                    return torch.stack(scores).mean().item()
                
        elif metric == 'dists':
            model = self._get_model('dists')
            with torch.no_grad():
                if C == 3:
                    return model(input, target).mean().item()
                elif C == 1:
                    return model(input.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1)).mean().item()
                else:
                    scores = []
                    for c in range(C):
                        inp_c = input[:, c:c+1, :, :].repeat(1, 3, 1, 1)
                        tgt_c = target[:, c:c+1, :, :].repeat(1, 3, 1, 1)
                        scores.append(model(inp_c, tgt_c).mean())
                    return torch.stack(scores).mean().item()
                
        elif metric == 'clipiqa':
            model = self._get_model('clipiqa')
            with torch.no_grad():
                if C == 3:
                    return model(input).mean().item()
                elif C == 1:
                    return model(input.repeat(1, 3, 1, 1)).mean().item()
                else:
                    scores = []
                    for c in range(C):
                        inp_c = input[:, c:c+1, :, :].repeat(1, 3, 1, 1)
                        scores.append(model(inp_c).mean())
                    return torch.stack(scores).mean().item()
        
        elif metric == 'psnr':
             return piq.psnr(input, target, data_range=1.0, reduction='mean').item()
             
        elif metric == 'ssim':
             return piq.ssim(input, target, data_range=1.0, reduction='mean').item()
             
        else:
            raise NotImplementedError(f"Metric {metric} not supported")


# Global evaluator instance
# We delay initialization to avoid CUDA errors if device is not ready or available
_evaluator = None

def _get_evaluator():
    global _evaluator
    if _evaluator is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _evaluator = AdvancedMetrics(device=device)
    return _evaluator

def fsim(input, target):
    """Feature Similarity Index Measure"""
    return _get_evaluator().compute(input, target, 'fsim')

def lpips(input, target):
    """Learned Perceptual Image Patch Similarity"""
    return _get_evaluator().compute(input, target, 'lpips')

def dists(input, target):
    """Deep Image Structure and Texture Similarity"""
    return _get_evaluator().compute(input, target, 'dists')

def clipiqa(input, target=None):
    """CLIP-based Image Quality Assessment (No-Reference)"""
    return _get_evaluator().compute(input, None, 'clipiqa')
