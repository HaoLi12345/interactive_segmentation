import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Union
from .unet import Unet_decoder, Conv, TwoConv
from monai.networks.nets import UNet


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class MaskDecoder3D(nn.Module):
    def __init__(
        self,
        args,
        *,
        transformer_dim: int = 384,
        multiple_outputs: bool = False,
        num_multiple_outputs: int = 3,
    ) -> None:
        super().__init__()
        self.args = args
        self.multiple_outputs = multiple_outputs
        self.num_multiple_outputs = num_multiple_outputs
        # if self.args.use_sam3d_turbo:
        #     self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, 48, 3) for i in range(num_multiple_outputs + 1)])
        # else:
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, 32, 3) for i in range(num_multiple_outputs + 1)])
        self.iou_prediction_head = MLP(transformer_dim, 256, num_multiple_outputs + 1, 3, sigmoid_output=True)

        self.decoder = Unet_decoder(spatial_dims=3, features=(32, 32, 64, 128, transformer_dim, 32))

        if self.args.refine:
            self.refine = Refine(self.args)

        # Multi-scale deep supervision: single-output aux heads at u2 (64^3, 32ch)
        # and u3 (32^3, 64ch). Driven by the first mask token. Used at training
        # only; inference uses the main 128^3 head exclusively.
        if getattr(self.args, "multi_scale_decoder", False):
            self.ms_mlp_64 = MLP(transformer_dim, transformer_dim, 32, 3)
            self.ms_mlp_32 = MLP(transformer_dim, transformer_dim, 64, 3)
        self.last_ms_masks = None

    def forward(
        self,
        prompt_embeddings: torch.Tensor, # prompt_embedding --> [b, self.num_mask_tokens, c]
        image_embeddings, # image_embedding --> [b, c, low_res / 4, low_res / 4, low_res / 4]
        feature_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ms_on = getattr(self.args, "multi_scale_decoder", False)
        if ms_on:
            u1, u2, u3 = self.decoder(image_embeddings, feature_list, return_multiscale=True)
        else:
            u1 = self.decoder(image_embeddings, feature_list)

        masks, iou_pred = self._predict_mask(u1, prompt_embeddings)

        if ms_on:
            mask_token_0 = prompt_embeddings[:, 1, :]  # first mask token drives aux heads
            masks_64 = self._predict_aux_mask(u2, mask_token_0, self.ms_mlp_64)
            masks_32 = self._predict_aux_mask(u3, mask_token_0, self.ms_mlp_32)
            self.last_ms_masks = (masks_64, masks_32)
        else:
            self.last_ms_masks = None

        return masks, iou_pred

    def _predict_aux_mask(self, embedding, mask_token, mlp):
        b, c, x, y, z = embedding.shape
        hyper_in = mlp(mask_token).unsqueeze(1)  # [b, 1, c]
        masks = (hyper_in @ embedding.view(b, c, x * y * z)).view(b, 1, x, y, z)
        return masks


    def _predict_mask(self, upscaled_embedding, prompt_embeddings):
        b, c, x, y, z = upscaled_embedding.shape

        iou_token_out = prompt_embeddings[:, 0, :]
        mask_tokens_out = prompt_embeddings[:, 1: (self.num_multiple_outputs + 1 + 1), :]  # multiple masks + iou

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_multiple_outputs + 1):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        masks = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.multiple_outputs:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred

class Refine_unet(nn.Module):
    def __init__(self):
        super(Refine_unet, self).__init__()
        self.refine = UNet(spatial_dims=3, in_channels=4, out_channels=1,
                           channels=(32, 64, 64), strides=(2, 2), num_res_units=2)
    def forward(self, x):
        return self.refine(x)

class Refine(nn.Module):
    def __init__(self,
                 args,
                 spatial_dims: int = 3,
                 in_channel: int = 4,
                 out_channel: int = 32,
                 act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm: Union[str, tuple] = ("instance", {"affine": True}),
                 bias: bool = True,
                 dropout: Union[float, tuple] = 0.0,
                 ):
        super().__init__()
        self.args = args

        # 3-state click memory adds a contradiction_map channel (image, mask, pos, neg, contra) = 5
        if getattr(args, "use_3state_memory", False):
            in_channel = 5

        self.first_conv = Conv["conv", 3](in_channels=in_channel, out_channels=out_channel, kernel_size=1)

        self.conv1 = TwoConv(spatial_dims, out_channel, out_channel, act, norm, bias, dropout)
        self.conv2 = TwoConv(spatial_dims, out_channel, out_channel, act, norm, bias, dropout)

        self.conv_error_map = Conv["conv", 3](in_channels=out_channel, out_channels=1, kernel_size=1)
        self.conv_correction = Conv["conv", 3](in_channels=out_channel, out_channels=1, kernel_size=1)


    def forward(self, image, mask_best, points, mask):

        x = self._get_refine_input(image, mask_best, points)
        mask = F.interpolate(mask, scale_factor=0.5, mode='trilinear', align_corners=False)

        x = self.first_conv(x)

        residual = x
        x = self.conv1(x)
        x = residual + x

        residual = x
        x = self.conv2(x)
        x = residual + x

        error_map = self.conv_error_map(x)
        correction = self.conv_correction(x)

        outputs = (error_map * correction + mask)

        outputs = F.interpolate(outputs, scale_factor=2, mode='trilinear', align_corners=False)
        error_map = F.interpolate(error_map, scale_factor=2, mode='trilinear', align_corners=False)
        return outputs, error_map

    def _get_refine_input(self, image, mask, points):
        mask = torch.sigmoid(mask)
        mask = (mask > 0.5)

        coors, labels = points[0], points[1]
        positive_map, negative_map = torch.zeros_like(image), torch.zeros_like(image)
        use_3state = getattr(self.args, "use_3state_memory", False)
        contradiction_map = torch.zeros_like(image) if use_3state else None
        # last_label tracks the most recent click label per voxel (-1 unclicked, 0 neg, 1 pos)
        last_label = torch.full(
            (image.size(0), image.size(2), image.size(3), image.size(4)),
            -1, dtype=torch.int8, device=image.device,
        ) if use_3state else None

        for click_iters in range(len(coors)):
            coors_click, labels_click = coors[click_iters], labels[click_iters]
            for batch in range(image.size(0)):
                point_label = labels_click[batch]
                coor = coors_click[batch]

                if use_3state:
                    # Latest-write-wins + contradiction tag (vectorized within iter).
                    if coor.shape[0] > 0:
                        d_idx = coor[:, 0].long()
                        h_idx = coor[:, 1].long()
                        w_idx = coor[:, 2].long()
                        lbl = point_label.long().to(image.device)
                        prev = last_label[batch, d_idx, h_idx, w_idx]
                        contradicted = (prev != -1) & (prev != lbl.to(torch.int8))
                        if contradicted.any():
                            cd, ch, cw = d_idx[contradicted], h_idx[contradicted], w_idx[contradicted]
                            contradiction_map[batch, 0, cd, ch, cw] = 1
                            positive_map[batch, 0, cd, ch, cw] = 0
                            negative_map[batch, 0, cd, ch, cw] = 0
                        is_pos = lbl == 1
                        if is_pos.any():
                            pd, ph, pw = d_idx[is_pos], h_idx[is_pos], w_idx[is_pos]
                            positive_map[batch, 0, pd, ph, pw] = 1
                        is_neg = ~is_pos
                        if is_neg.any():
                            nd, nh, nw = d_idx[is_neg], h_idx[is_neg], w_idx[is_neg]
                            negative_map[batch, 0, nd, nh, nw] = 1
                        last_label[batch, d_idx, h_idx, w_idx] = lbl.to(torch.int8)
                else:
                    # PRISM original: vector ops, but writes pos=neg=1 if same voxel reappears with opposite label
                    negative_mask = point_label == 0
                    positive_mask = point_label != 0
                    if negative_mask.any():
                        negative_indices = coor[negative_mask]
                        for idx in negative_indices:
                            negative_map[batch, 0, idx[0], idx[1], idx[2]] = 1
                    if positive_mask.any():
                        positive_indices = coor[positive_mask]
                        for idx in positive_indices:
                            positive_map[batch, 0, idx[0], idx[1], idx[2]] = 1

        if use_3state:
            stacked = torch.cat([image, mask, positive_map, negative_map, contradiction_map], dim=1)
        else:
            stacked = torch.cat([image, mask, positive_map, negative_map], dim=1)
        refine_input = F.interpolate(stacked, scale_factor=0.5, mode='trilinear')
        return refine_input

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x

