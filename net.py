import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class DeepFloorModel(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = self._initializeVGG(pretrained, freeze)

        # Room Boundary Prediction
        room_boundary_dims = [512, 256, 128, 64, 32, 3]
        self.room_boundary_upsample = nn.ModuleList([nn.ConvTranspose2d(
            room_boundary_dims[i], room_boundary_dims[i+1], kernel_size=4, stride=2, padding=1) for i in range(len(room_boundary_dims)-2)])

        self.room_boundary_concat_convs = nn.ModuleList([nn.Conv2d(
            room_boundary_dims[i], room_boundary_dims[i+1], kernel_size=3, stride=1, padding=1) for i in range(len(room_boundary_dims)-1)])

        self.room_boundary_convs = nn.ModuleList([nn.Conv2d(
            room_boundary_dims[i], room_boundary_dims[i], kernel_size=3, stride=1, padding=1) for i in range(1, len(room_boundary_dims)-1)])

        # Room Type Prediction
        room_type_dims = [512, 256, 128, 64, 32]
        self.room_type_upsample = nn.ModuleList([nn.ConvTranspose2d(
            room_type_dims[i], room_type_dims[i+1], kernel_size=4, stride=2, padding=1) for i in range(len(room_type_dims)-1)])

        self.room_type_concat_convs = nn.ModuleList([nn.Conv2d(
            room_type_dims[i], room_type_dims[i+1], kernel_size=3, stride=1, padding=1) for i in range(len(room_type_dims)-1)])

        self.room_type_convs = nn.ModuleList([nn.Conv2d(
            room_type_dims[i], room_type_dims[i], kernel_size=3, stride=1, padding=1) for i in range(1, len(room_type_dims))])

        # Spatial contextual module
        spatial_context_dims = [256, 128, 64, 32]

        # Spatial contextual module Room Boundary part
        self.room_bndr_context_frs_layers = nn.ModuleList(nn.Conv2d(
            spatial_context_dims[i], spatial_context_dims[i], kernel_size=3, stride=1, padding=1) for i in range(len(spatial_context_dims)))

        self.room_bndr_context_snd_layers = nn.ModuleList(nn.Conv2d(
            spatial_context_dims[i], spatial_context_dims[i], kernel_size=3, stride=1, padding=1) for i in range(len(spatial_context_dims)))

        self.room_bndr_context_reduce = nn.ModuleList(nn.Conv2d(
            spatial_context_dims[i], 1, kernel_size=1, stride=1) for i in range(len(spatial_context_dims)))

        # Spatial contextual module Room Type part
        self.room_type_context_frs_layers = nn.ModuleList(nn.Conv2d(
            spatial_context_dims[i], spatial_context_dims[i], kernel_size=3, stride=1, padding=1) for i in range(len(spatial_context_dims)))

        self.room_type_context_reduce = nn.ModuleList(nn.Conv2d(
            spatial_context_dims[i], 1, kernel_size=1, stride=1) for i in range(len(spatial_context_dims)))

        # Spatial contextual module Direction-awake kernels part
        direction_awake_dims = [9, 17, 33, 65]
        # horzontal
        self.horzontal_layers = nn.ModuleList(self._diraction_awake_layer(
            [1, 1, dim, 1]) for dim in direction_awake_dims)

        # vertical
        self.vertical_layers = nn.ModuleList(self._diraction_awake_layer(
            [1, 1, 1, dim]) for dim in direction_awake_dims)

        # diagonal
        self.diagonal_layers = nn.ModuleList(self._diraction_awake_layer(
            [1, 1, dim, dim], diag=True) for dim in direction_awake_dims)

        # diagonal flip
        self.diagonal_flip_layers = nn.ModuleList(self._diraction_awake_layer(
            [1, 1, dim, dim], diag=True, flip=True) for dim in direction_awake_dims)

        # Spatial contextual module last part
        self.expand_context_convs = nn.ModuleList(nn.Conv2d(
            1, spatial_context_dims[i], kernel_size=1, stride=1) for i in range(len(spatial_context_dims)))

        self.reduce_context_convs = nn.ModuleList(nn.Conv2d(
            2*spatial_context_dims[i], spatial_context_dims[i], kernel_size=1, stride=1) for i in range(len(spatial_context_dims)))

        # Last layer
        self.last = nn.Conv2d(spatial_context_dims[-1], 9, kernel_size=1, stride=1)

    def _diraction_awake_layer(self, shape, diag=False, flip=False, trainable=False):
        w = self.constant_kernel(shape, diag, flip, trainable)
        pad = ((np.array(shape[2:])-1) / 2).astype(int)
        conv = nn.Conv2d(1, 1, shape[2:], 1, list(pad), bias=False)
        conv.weight = w
        return conv

    def _initializeVGG(self, pretrained, freeze):
        encmodel = models.vgg16(pretrained=pretrained)
        if freeze:
            for child in encmodel.children():
                for param in child.parameters():
                    param.requires_grad = False
        features = list(encmodel.features)[:31]
        return nn.ModuleList(features)

    def constant_kernel(self, shape, value=1, diag=False, flip=False, trainable=False):
        if not diag:
            k = nn.Parameter(torch.ones(shape)*value, requires_grad=trainable)
        else:
            w = torch.eye(shape[2], shape[3])
            if flip:
                w = torch.reshape(w, (1, shape[2], shape[3]))
                w = w.flip(0, 1)
            w = torch.reshape(w, shape)
            k = nn.Parameter(w, requires_grad=trainable)
        return k

    def non_local_context(self, room_boud_feat, room_type_feat, idx):

        a = nn.ReLU()(self.room_bndr_context_frs_layers[idx](room_boud_feat))
        a = nn.ReLU()(self.room_bndr_context_snd_layers[idx](a))
        a = nn.Sigmoid()(self.room_bndr_context_reduce[idx](a))
        x = nn.ReLU()(self.room_type_context_frs_layers[idx](room_type_feat))
        x = nn.Sigmoid()(self.room_type_context_reduce[idx](x))
        x = a * x

        # direction-aware kernels
        h = self.horzontal_layers[idx](x)
        v = self.vertical_layers[idx](x)
        d1 = self.diagonal_layers[idx](x)
        d2 = self.diagonal_flip_layers[idx](x)

        # double attention
        c1 = a * (h + v + d1 + d2)

        # expand channel
        c1 = self.expand_context_convs[idx](c1)

        # concatenation + upsample
        features = torch.cat((room_type_feat, c1), dim=1)

        out = nn.ReLU()(self.reduce_context_convs[idx](features))
        return out

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 9, 16, 23, 30}:
                results.append(x)
        rbfeatures = []
        for i, rbtran in enumerate(self.room_boundary_upsample):
            x = rbtran(x)+self.room_boundary_concat_convs[i](results[3-i])
            x = nn.ReLU()(self.room_boundary_convs[i](x))
            rbfeatures.append(x)
        logits_cw = F.interpolate(self.room_boundary_concat_convs[-1](x), 512)
        
        x = results[-1]
        for j, rttran in enumerate(self.room_type_upsample):
            x = rttran(x)+self.room_type_concat_convs[j](results[3-j])
            x = nn.ReLU()(self.room_type_convs[j](x))
            x = self.non_local_context(rbfeatures[j], x, j)

        logits_r = F.interpolate(self.last(x), 512)

        return logits_r, logits_cw
