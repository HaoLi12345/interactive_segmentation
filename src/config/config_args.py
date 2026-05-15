import argparse
import os
import warnings
parser = argparse.ArgumentParser()


# data
parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
parser.add_argument("--save_dir", default="./implementation/", type=str)
parser.add_argument("--data_dir", default="", type=str)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--split", default="train", type=str)
parser.add_argument('--use_small_dataset', action="store_true")
parser.add_argument("--max_cases", default=0, type=int, help="Truncate image_paths to first N for smoke runs (0 = no limit)")


# network
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument("--lr", default=4e-5, type=float)
parser.add_argument("--lr_scheduler", default='linear', type=str, choices=["linear", "exp"])
parser.add_argument('--warm_up', action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--max_epoch", default=200, type=int)
parser.add_argument("--image_size", default=128, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--checkpoint", default="best", type=str)
parser.add_argument("--checkpoint_sam", default="./checkpoint_sam/sam_vit_b_01ec64.pth", type=str,
                    help='path of pretrained SAM')
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--tolerance", default=5, type=int)
parser.add_argument("--boundary_kernel_size", default=5, type=int,
                    help='an integer for kernel size of avepooling layer for boundary generation')
parser.add_argument("--use_pretrain", action="store_true")
parser.add_argument("--pretrain_path", default="", type=str)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume_best", action="store_true")
parser.add_argument("--ddp", action="store_true")
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--accumulation_steps', type=int, default=20)

parser.add_argument('--iter_nums', type=int, default=11)
parser.add_argument('--num_clicks', type=int, default=50)
parser.add_argument('--num_clicks_validation', type=int, default=10)
parser.add_argument('--use_box', action="store_true")
parser.add_argument('--dynamic_box', action="store_true")
parser.add_argument('--use_scribble', action="store_true")
parser.add_argument('--click_strategy', default='random', choices=['random', 'entropy'],
                    help='How to pick the next click voxel within FN/FP regions. random=PRISM default; entropy=top-k by binary entropy of prev prediction.')


parser.add_argument('--num_multiple_outputs', type=int, default=3)
parser.add_argument('--multiple_outputs', action="store_true")
parser.add_argument('--refine', action="store_true")
parser.add_argument('--no_detach', action="store_true")
parser.add_argument('--refine_test', action="store_true")

parser.add_argument('--dynamic', action="store_true")
parser.add_argument('--efficient_scribble', action="store_true")
parser.add_argument("--use_sam3d_turbo", action="store_true")

parser.add_argument('--multi_scale_decoder', action='store_true',
                    help='Add 64^3 + 32^3 deep-supervision aux heads on the mask decoder. Inference unchanged.')
parser.add_argument('--ms_aux_weights', default='0.5,0.25', type=str,
                    help='Comma-separated aux loss weights for (64^3, 32^3) scales. Main 128^3 weight is fixed at 1.0.')

parser.add_argument('--scribble_every_k_slices', default=0, type=int,
                    help='Test-time only: keep scribble voxels only on every k-th axial slice (D axis, dim 0). 0 = no filtering.')

parser.add_argument('--use_3state_memory', action='store_true',
                    help='Refine head: use 3-state click memory (positive/negative/contradicted maps). Adds 1 channel to refine input. '
                         'Without this flag, PRISM original behavior: same voxel can have positive_map=1 AND negative_map=1 simultaneously.')

parser.add_argument('--sparse_scribble_train', action='store_true',
                    help='Train-time only: enable scribble in get_points and apply random-orientation top-K-error slice filter. '
                         'Mutually requires --use_scribble.')
parser.add_argument('--sparse_scribble_dense_prob', default=0.2, type=float,
                    help='Per-iter probability of using full (dense) scribble (K=infinity); otherwise sample K from [1, K_max].')
parser.add_argument('--sparse_scribble_K_max', default=5, type=int,
                    help='When sampling sparse K, K ~ Uniform[1, K_max].')
parser.add_argument('--sparse_scribble_orientations', default='axial,sagittal,coronal', type=str,
                    help='Comma-separated orientations to randomize over: axial=dim0, sagittal=dim1, coronal=dim2.')

parser.add_argument('--test_K_slices', default=0, type=int,
                    help='Test-time top-K-error slice filter (replaces deprecated --scribble_every_k_slices when >0).')
parser.add_argument('--test_slice_orientation', default='axial', type=str, choices=['axial', 'sagittal', 'coronal', 'random'],
                    help='Test-time slice orientation for K-slice filter.')
parser.add_argument('--test_contradiction_rate', default=0.0, type=float,
                    help='Test-time only: probability of flipping a click label per iter (simulates user changing mind).')
parser.add_argument('--force_inter_iter_contradiction', action='store_true',
                    help='[DEPRECATED, use --inter_iter_contradiction_N instead] Test-time: at iter k>=1, replace one click with a previous-iter voxel with flipped label.')
parser.add_argument('--save_per_iter_predictions', action='store_true',
                    help='Test-time: save per-iter mask + (once) image + GT as nii.gz under save_test_dir/per_iter_pred/<data>/<save_name>/<case>/.')
parser.add_argument('--inter_iter_contradiction_N', default=0, type=int,
                    help='Test-time: at iter k>=1, sample N random voxels from a previous iters prompt pool (mostly scribble), '
                         'flip labels, append to current iter. Triggers PRISM refine-head pos_map=neg_map=1 bug at scale.')



# saving
parser.add_argument("--save_predictions", action="store_true")
parser.add_argument("--save_csv", action="store_true")
parser.add_argument("--save_test_dir", default='./', type=str)
parser.add_argument("--save_name", default='testing_only', type=str)






def check_and_setup_parser(args):
    if args.save_name == 'testing_only':
        warnings.warn("[save_name] (--save_name) should be a real name, currently is for testing purpose (--save_name=testing_only)")


    args.save_dir = os.path.join(args.save_dir, args.data, args.save_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
