import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from segmentation_models_pytorch.utils import base

def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)
class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss

class custom_loss(base.Loss):
    def __init__(self, batch_size):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()
        self.contrast = ContrastiveLoss(batch_size)
        self.psnr = PeakSignalNoiseRatio()
        self.L1 = nn.L1Loss()
    def forward(self, y_pr, y_gt, ft1=None, ft2=None):
        x=self.mse(y_pr, y_gt)
        y=1-self.ssim(y_pr, y_gt)
        z=self.contrast(ft1, ft2)
        p=(1 - self.psnr(y_pr, y_gt)/40)
#         l = self.L1(y_pr, y_gt)
#         print(x,y,z)
#         return x
#         return x+y/10+z/100
        return x+p/10+z/10+y/10
    
class custom_lossv(base.Loss):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mse = nn.MSELoss()
        # self.contrast = ContrastiveLoss(batch_size)
        self.psnr = PeakSignalNoiseRatio()
        self.L1 = nn.L1Loss()
    def forward(self, y_pr, y_gt, ft1=None, ft2=None):
        x=self.mse(y_pr, y_gt)
        y=1-self.ssim(y_pr, y_gt)
        # z=self.contrast(ft1, ft2)
        p=(1 - self.psnr(y_pr, y_gt)/40)
        return x+p/10+y/10
        