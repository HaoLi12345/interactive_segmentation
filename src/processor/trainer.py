import torch
import torch.nn.functional as F
import random
import numpy as np
from src.utils.util import _bbox_mask
from src.utils import scribble, boundary_selection
from .trainer_basic import Trainer_basic

class Trainer(Trainer_basic):
    def __init__(self, args, logger):
        super().__init__(args, logger)

    def forward(self, sam_model, image, label, iter_nums, train=False, return_each_iter=False):
        if return_each_iter:
            return_mask_total_iter = torch.zeros([iter_nums, 1, image.size(2), image.size(3), image.size(4)])

        image_embedding, feature_list = self.sam.image_encoder(image)
        self.click_points = []
        self.click_labels = []
        return_loss = 0
        prev_masks = torch.zeros_like(label, dtype=torch.float).to(label.device)
        for iter_num in range(iter_nums):
            loss = 0
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

            points_input, labels_input, box_input = self.get_points(prev_masks_sigmoid, label, train_mode=train)
            mask, dice_pred = self.iteration_forward(sam_model, feature_list, image_embedding, prev_masks,
                                                     points=[points_input, labels_input], boxes=box_input)

            decoder_module = sam_model.module.mask_decoder if hasattr(sam_model, "module") else sam_model.mask_decoder
            ms_masks = getattr(decoder_module, "last_ms_masks", None)

            # ========================================================
            if self.args.multiple_outputs:
                dice_pred_best, max_label_index = torch.max(dice_pred, dim=1)
                mask_list = [mask[i, max_label_index[i], :].unsqueeze(0) for i in range(mask.size(0))]
                mask_best = torch.stack(mask_list, dim=0)
            else:
                mask_best = mask

            # ========================================================
            if train:
                if self.args.multiple_outputs:
                    for i in range(mask.size(1)):
                        single_mask, single_dice = mask[:, i, :].unsqueeze(1), dice_pred[:, i]
                        # ms_masks aux loss is added once per iter (only on first hypothesis)
                        ms = ms_masks if i == 0 else None
                        loss += self.calculate_loss(single_mask, prev_masks, single_dice, label, labels_input, iter_num, ms_masks=ms)
                else:
                    loss = self.calculate_loss(mask, prev_masks, dice_pred[:, 0], label, labels_input, iter_num, ms_masks=ms_masks)

                # ========================================================
                if self.args.refine:
                    if self.args.no_detach:
                        mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best,
                                                                              [self.click_points, self.click_labels],
                                                                              mask_best)
                    else:
                        mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best, [self.click_points, self.click_labels], mask_best.detach())
                    print('dice before refine {} and after {}'.format(
                        self.get_dice_score(torch.sigmoid(mask_best), label),
                        self.get_dice_score(torch.sigmoid(mask_refine), label)))

                    # ========================================================
                    loss += self.loss_segmentation(mask_refine, label) * 1

                    mask_best = mask_refine

            # ========================================================
            else:
                if self.args.refine:
                    if self.args.no_detach:
                        mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best,
                                                                              [self.click_points, self.click_labels],
                                                                              mask_best)
                    else:
                        mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best,
                                                                              [self.click_points, self.click_labels],
                                                                              mask_best.detach())
                    if iter_num == iter_nums - 1 or iter_num == 0:
                        self.logger.info('dice before refine {} and after {}, label 0: {}, label 1: {}'.format(
                            self.get_dice_score(torch.sigmoid(mask_best), label), self.get_dice_score(torch.sigmoid(mask_refine), label),
                            str(labels_input.numel() - torch.count_nonzero(labels_input)), str(torch.count_nonzero(labels_input)) ) )
                    mask_best = mask_refine
                loss = self.get_dice_score(torch.sigmoid(mask_best), label)

            return_loss += loss
            prev_masks = mask_best

            if return_each_iter:
                return_mask_total_iter[iter_num, :] = mask_best

        if return_each_iter:
            return return_loss / iter_nums, return_mask_total_iter
        else:
            return return_loss / iter_nums, prev_masks

    def get_points(self, prev_masks, label, train_mode=True):
        mode = 'train' if train_mode else 'validation'

        batch_points, batch_labels = self.get_next_point(prev_masks, label, mode=mode)

        points_co = torch.cat(batch_points, dim=0).to(self.args.device) # b x num_clicks x 3
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device) # b x num_clicks x 1

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_input = points_co
        labels_input = points_la

        bbox_coords = _bbox_mask(label[:, 0, :], mode=mode, dynamic=self.args.dynamic_box).to(self.args.device) if self.args.use_box else None

        return points_input, labels_input, bbox_coords

    def _sample_sparse_K_and_dim(self, mode):
        """Returns (K, dim_idx). K=-1 means dense (no filter). dim_idx in {0,1,2}."""
        if mode != 'train' or not getattr(self.args, "sparse_scribble_train", False):
            return -1, -1
        p_dense = getattr(self.args, "sparse_scribble_dense_prob", 0.2)
        K_max = getattr(self.args, "sparse_scribble_K_max", 5)
        if random.random() < p_dense:
            return -1, -1
        K = random.randint(1, K_max)
        orient_str = getattr(self.args, "sparse_scribble_orientations", "axial,sagittal,coronal")
        orient_names = [s.strip() for s in orient_str.split(",") if s.strip()]
        dim_map = {"axial": 0, "sagittal": 1, "coronal": 2}
        chosen = random.choice(orient_names) if orient_names else "axial"
        return K, dim_map.get(chosen, 0)

    def _filter_scribble_topk_slices(self, coors, error_mask_chw, K, dim_idx):
        """Filter (1, N, 3) scribble coors to top-K-error slices on dim_idx.
        error_mask_chw shape: (C, D, H, W); coors layout: (D, H, W).
        Selects K slices with the highest FN+FP voxel count and keeps voxels on those slices.
        """
        if K < 1 or coors.size(1) == 0:
            return coors
        err = error_mask_chw.float()
        # sum over all spatial dims except dim_idx (within (D,H,W) = dims 1,2,3 of err)
        sum_axes = tuple([0] + [d for d in (1, 2, 3) if d != dim_idx + 1])
        per_slice = err.sum(dim=sum_axes)  # 1D tensor of length = slice count along dim_idx
        K_eff = min(K, int((per_slice > 0).sum().item()) or per_slice.size(0))
        K_eff = max(1, K_eff)
        topk = torch.topk(per_slice, K_eff).indices.to(coors.device)
        coor_dim = coors[0, :, dim_idx]
        mask = torch.isin(coor_dim, topk)
        return coors[:, mask, :]

    def get_next_point(self, prev_seg, label, mode='train'): # prev_seg --> probability
        batch_points = []
        batch_labels = []

        pred_masks = (prev_seg > 0.5)
        true_masks = (label > 0)
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

        to_point_mask = torch.logical_or(fn_masks, fp_masks)

        # Sample K and orientation once per call: per-iter sparsity, all batch items share K.
        sparse_K, sparse_dim = self._sample_sparse_K_and_dim(mode)

        # do_scribble = random.random()
        # sample_method = random.choice(['line', 'center', 'default'])
        sample_method = 'center'
        scribble_types = {
            'line': 'LineScribble',
            'center': 'CenterlineScribble',
            'default': 'ContourScribble'
        }

        def create_scribble_mask(scribble_type, data):
            scribble_object = getattr(scribble, scribble_type)()
            scribble_mask = scribble_object.batch_scribble(data).permute(1, 2, 3, 0)
            return scribble_mask > 0


        points_list = [len(torch.argwhere(to_point_mask[i])) for i in range(to_point_mask.size(0))]
        points_min = min(points_list)
        num_clicks = self.args.num_clicks if mode == 'train' else self.args.num_clicks_validation
        click_size = points_min if num_clicks > points_min else num_clicks
        dynamic_size = random.randint(1, click_size) if self.args.dynamic and mode == 'train' else click_size
        print(f"num_clicks {num_clicks} points_length: {points_min} dynamic_size: {dynamic_size}")

        for i in range(label.shape[0]):
            bp_list, bl_list = [], []
            points = torch.argwhere(to_point_mask[i])

            # Click strategy: 'random' (PRISM default) vs 'entropy' (ours).
            # Entropy picks the top-k voxels in FN/FP region with highest binary
            # entropy of the previous prediction. At iter 0 prev_seg is all zero
            # so entropy is 0 everywhere -> degenerates to argmax order (similar
            # to random). From iter >= 1 it focuses on uncertain frontiers, which
            # is what we want both at train and test time so the prompt
            # distribution seen during training matches the inference policy.
            if getattr(self.args, "click_strategy", "random") == "entropy":
                eps = 1e-6
                p = prev_seg[i].clamp(eps, 1 - eps)
                p_at = p[points[:, 0], points[:, 1], points[:, 2], points[:, 3]]
                entropy = -(p_at * torch.log(p_at) + (1 - p_at) * torch.log(1 - p_at))
                point_index = torch.topk(entropy, dynamic_size, largest=True).indices.cpu().numpy()
            else:
                point_index = np.random.choice(len(points), size=dynamic_size, replace=False)
            points_select = points[point_index] # each row tensor([0, x, y, z]), size --> num_clicks x 4

            for click_index in range(dynamic_size):
                point = points_select[click_index]
                if fn_masks[i, 0, point[1], point[2], point[3]]:
                    is_positive = True
                else:
                    is_positive = False

                bp = point[1:].clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive), ]).reshape(1, 1)
                bp_list.append(bp)
                bl_list.append(bl)

            if self.args.use_scribble:
                fg, bg_orig = fn_masks[i].permute(3, 0, 1, 2).float(), fp_masks[i].permute(3, 0, 1, 2).float()

                # ====== with the purpose of efficiency only for first few epochs ======
                bbx = _bbox_mask(label[i, 0, :].unsqueeze(0))
                diff_ = 15
                i_min, i_max = bbx[:, :, 0], bbx[:, :, 3]
                j_min, j_max = bbx[:, :, 1], bbx[:, :, 4]
                k_min, k_max = bbx[:, :, 2], bbx[:, :, 5]
                if max(0, i_min - diff_) < min(i_max + diff_, 126):
                    i_min, i_max = max(0, i_min - diff_), min(i_max + diff_, 126)
                if max(0, j_min - diff_) < min(j_max + diff_, 126):
                    j_min, j_max = max(0, j_min - diff_), min(j_max + diff_, 126)
                if max(0, k_min - diff_) < min(k_max + diff_, 126):
                    k_min, k_max = max(0, k_min - diff_), min(k_max + diff_, 126)

                bg_mask = torch.zeros_like(bg_orig).permute(1, 2, 3, 0)
                bg_mask[:, i_min:i_max, j_min:j_max, k_min:k_max] = 1
                bg = bg_orig * bg_mask.permute(3, 0, 1, 2)
                print('filter out voxels: {}'.format(torch.count_nonzero(bg_orig) - torch.count_nonzero(bg)))

                scribble_type = scribble_types.get(sample_method, scribble_types['default'])
                scribble_mask_fg = create_scribble_mask(scribble_type, fg)

                limit_num = 500
                fg_coors_full = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)
                # Sparse-K-slice filter: keep voxels on top-K-error slices in chosen orientation.
                if sparse_K > 0 and fg_coors_full.size(1) > 0:
                    fg_coors_full = self._filter_scribble_topk_slices(fg_coors_full, fn_masks[i], sparse_K, sparse_dim)
                if fg_coors_full.size(1) >= limit_num + 50:
                    a = fg_coors_full.size(1) - limit_num
                    random_number = random.randint(0, a)
                    fg_coors = fg_coors_full[:, random_number: random_number + limit_num, :]
                else:
                    fg_coors = fg_coors_full

                fg_coors_label = torch.ones(1, fg_coors.size(1))
                if fg_coors.size(1) > 0:
                    bp_list.append(fg_coors)
                    bl_list.append(fg_coors_label)


                scribble_mask_bg = create_scribble_mask(scribble_type, bg)
                bg_coors_full = torch.argwhere(scribble_mask_bg)[:, 1:].unsqueeze(0)
                if sparse_K > 0 and bg_coors_full.size(1) > 0:
                    bg_coors_full = self._filter_scribble_topk_slices(bg_coors_full, fp_masks[i], sparse_K, sparse_dim)
                if bg_coors_full.size(1) >= limit_num + 50:
                    a = bg_coors_full.size(1) - limit_num
                    random_number = random.randint(0, a)
                    bg_coors = bg_coors_full[:, random_number: random_number + limit_num, :]
                else:
                    bg_coors = bg_coors_full

                bg_coors_label = torch.zeros(1, bg_coors.size(1))
                if bg_coors.size(1) > 0:
                    bp_list.append(bg_coors)
                    bl_list.append(bg_coors_label)

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

        # for scribble
        if self.args.use_scribble:
            smallest_n = min(tensor.size(1) for tensor in batch_labels)
            batch_points = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in batch_points]
            batch_labels = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in batch_labels]

        # # Check the shapes of the adjusted tensors
        # for i, tensor in enumerate(batch_points):
        #     print(f"Tensor {i + 1} shape: {tensor.shape}")

        print('First batch:   fn: {:.4f}, fp: {:.4f}, label 0: {}, label 1: {}'.format(
            torch.count_nonzero(fn_masks[0]) / torch.count_nonzero(true_masks[0]),
            torch.count_nonzero(fp_masks[0]) / torch.count_nonzero(true_masks[0]),
            str(batch_labels[0].numel() - torch.count_nonzero(batch_labels[0])),
            str(torch.count_nonzero(batch_labels[0]))
        )
        )
        print('--- ===================================== ---')
        print('--- above before model, below after model ---')
        print('--- ===================================== ---')
        return batch_points, batch_labels

    def iteration_forward(self, sam_model, features, image_embedding, prev_masks, points=None, boxes=None):
        prev_masks = F.interpolate(prev_masks, scale_factor=0.25)
        features = [features[i].to(self.args.device) for i in range(0, len(features))]

        new_point_embedding, new_image_embedding = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=prev_masks,
            image_embeddings=image_embedding.to(self.args.device)
        )

        mask, dice_pred = sam_model.mask_decoder(
            prompt_embeddings=new_point_embedding,  # (B, 2, 256)
            image_embeddings=new_image_embedding,  # (B, 256, 64, 64)
            feature_list=features,
        )
        return mask, dice_pred









