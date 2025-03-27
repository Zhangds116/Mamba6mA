import torch
from torch import nn
import torch.nn.functional as F


class MultiRepresentationFusion(nn.Module):
    def __init__(self):
        super(MultiRepresentationFusion, self).__init__()
        self.Wa = nn.Parameter(torch.randn(1, 1, 64).cuda(), requires_grad=True)
        self.Wb = nn.Parameter(torch.randn(1, 1, 64).cuda(), requires_grad=True)
        self.Wc = nn.Parameter(torch.randn(1, 1, 64).cuda(), requires_grad=True)

    def forward(self, representationY3, representationY5, representationY7):

        Ya = torch.sigmoid(self.Wa * representationY3)
        Yb = torch.sigmoid(self.Wb * representationY5)
        Yc = torch.sigmoid(self.Wc * representationY7)


        Y_combined = F.softmax(torch.cat([Ya, Yb, Yc], dim=-1), dim=-1)


        Y_combined = Y_combined.view(Y_combined.size(0), Y_combined.size(1), 3, -1)


        representation = (Y_combined[:, :, 0] * representationY3) + \
                         (Y_combined[:, :, 1] * representationY5) + \
                         (Y_combined[:, :, 2] * representationY7)
        return representation

