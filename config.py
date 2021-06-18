import numpy as np
import argparse, os, cv2


# set the path here
dataset_folder = '/opt/dataset'
scene_id = 1 # {1, 2, ..., 10}


parser = argparse.ArgumentParser()
# tree of space partition.
parser.add_argument('--tree_log2_ary', default=4) # 2^n ary tree.
parser.add_argument('--tree_height', default=6) # include root and leaf.
parser.add_argument('--n2d_file', default='depth_adaptive_random_pattern.npy') # depth adaptive 2d pattern for context points initial selection.
parser.add_argument('--n_coord_per_leaf', default=50000) # max number of coordinates each leaf. 10k default.
parser.add_argument('--multi_leaf', default=[2,2]) # top-k leaf gmm union.
# train.
parser.add_argument('--n_pts_per_frame', type=int, default=256) # default 256.
parser.add_argument('--fake_data_proportion', default=0.2) # percentage of simulated outlier.
parser.add_argument('--num_workers', default=0) # dataset loader.
#parser.add_argument('--num_workers', default=4) # dataset loader.
parser.add_argument('--epoch', default=[60, 60, 60, 70, 80], type=int) # epoch for each level.
parser.add_argument('--LR', default=0.001) # learninig rate.
parser.add_argument('--LR_step', default=30) # reduce learninig rate.
# test.
parser.add_argument('--test_seq')
parser.add_argument('--idx_step', default=10, type=int) # 10 to skip per 10 frames for fast evaluation.
parser.add_argument('--g_n_pts_per_frame', type=int, default=2048) # sample points to predict gmm. 1024 default. 
# for dataset rio10
parser.add_argument('--dataset', default='rio10')
parser.add_argument('--image_width', default=540)
parser.add_argument('--image_height', default=960)
parser.add_argument('--depth_format', default='frame-{:06d}.rendered.depth.png')
parser.add_argument('--color_format', default='frame-{:06d}.color.jpg')
parser.add_argument('--pose_format', default='frame-{:06d}.pose.rnd.txt')
if scene_id == 1: # args for rio10 scene01
    parser.add_argument('--scene_name', default='scene01')
    parser.add_argument('--train_seq', default='seq01_01')
    parser.add_argument('--data_root', default='{}/scene01/seq01'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-2.12, 2.34, -2.31, 1.63, -0.17, 2.41])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[2.23, 1.12, 0.56, 0.28, 0.14, 0.07, 0.04], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[756.02630615, 0., 270.41879272], [0., 756.83215332, 492.88940430], [0., 0., 1.]], dtype=np.double))
elif scene_id == 2: # args for rio10 scene02 
    parser.add_argument('--scene_name', default='scene02')
    parser.add_argument('--train_seq', default='seq02_01')
    parser.add_argument('--data_root', default='{}/scene02/seq02'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-3.30, 2.58, -4.81, 0.03, -0.09, 3.33])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[2.94, 1.47, 0.74, 0.37, 0.19, 0.10, 0.05], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[756.02630615, 0., 270.41879272], [0., 756.83215332, 492.88940430], [0., 0., 1.]], dtype=np.double))
elif scene_id == 3: # args for rio10 scene03 
    parser.add_argument('--scene_name', default='scene03')
    parser.add_argument('--train_seq', default='seq03_01')
    parser.add_argument('--data_root', default='{}/scene03/seq03'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-4.09, 2.21, -3.24, 2.79, -0.19, 2.70])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[3.15, 1.58, 0.79, 0.40, 0.20, 0.10, 0.05], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[756.02630615, 0., 270.41879272], [0., 756.83215332, 492.88940430], [0., 0., 1.]], dtype=np.double))
elif scene_id == 4: # args for rio10 scene04
    parser.add_argument('--scene_name', default='scene04')
    parser.add_argument('--train_seq', default='seq04_01')
    parser.add_argument('--data_root', default='{}/scene04/seq04'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-1.65, 3.26, -2.53, 2.25, -0.05, 2.60])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[2.46, 1.23, 0.62, 0.31, 0.16, 0.08, 0.04], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[756.02630615, 0., 270.41879272], [0., 756.83215332, 492.88940430], [0., 0., 1.]], dtype=np.double))
elif scene_id == 5: # args for rio10 scene05
    parser.add_argument('--scene_name', default='scene05')
    parser.add_argument('--train_seq', default='seq05_01')
    parser.add_argument('--scene_bbx', default=np.array([-1.22, 3.37, -2.73, 1.69, -0.17, 3.97])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[2.30, 1.15, 0.58, 0.29, 0.15, 0.08, 0.04], help='in meter') # to query context points.
    parser.add_argument('--data_root', default='{}/scene05/seq05'.format(dataset_folder))
    parser.add_argument('--intrinsics', default=np.array([[756.02630615, 0., 270.41879272], [0., 756.83215332, 492.88940430], [0., 0., 1.]], dtype=np.double))
elif scene_id == 6: # args for rio10 scene06 
    parser.add_argument('--scene_name', default='scene06')
    parser.add_argument('--train_seq', default='seq06_01')
    parser.add_argument('--data_root', default='{}/scene06/seq06'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-0.92, 4.54, -1.00, 4.40, -0.09, 2.53])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[2.73, 1.37, 0.69, 0.35, 0.18, 0.09, 0.05], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[756.02630615, 0., 270.41879272], [0., 756.83215332, 492.88940430], [0., 0., 1.]], dtype=np.double))
elif scene_id == 7: # args for rio10 scene07
    parser.add_argument('--scene_name', default='scene07')
    parser.add_argument('--train_seq', default='seq07_01')
    parser.add_argument('--data_root', default='{}/scene07/seq07'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-3.61, 2.44, -2.64, 3.38, -0.06, 3.35])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[3.03, 1.52, 0.76, 0.38, 0.19, 0.10, 0.05], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[760.42034912, 0., 268.74777222], [0., 761.68334961, 486.77813721], [0., 0., 1.]], dtype=np.double))
elif scene_id == 8: # args for rio10 scene08 
    parser.add_argument('--scene_name', default='scene08')
    parser.add_argument('--train_seq', default='seq08_01')
    parser.add_argument('--data_root', default='{}/scene08/seq08'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-6.84, 2.53, -5.22, 4.24, -0.07, 2.72])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[4.73, 2.37, 1.19, 0.60, 0.30, 0.15, 0.08], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[759.50683594, 0., 268.33862305], [0., 760.64569092, 487.21499634], [0., 0., 1.]], dtype=np.double))
elif scene_id == 9: # args for rio10 scene09 
    parser.add_argument('--scene_name', default='scene09')
    parser.add_argument('--train_seq', default='seq09_01')
    parser.add_argument('--data_root', default='{}/scene09/seq09'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-3.82, 2.13, -4.29, 1.49, -0.18, 2.53])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[2.98, 1.49, 0.75, 0.38, 0.19, 0.10, 0.05], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[758.79235840, 0., 266.20843506], [0., 760.05169678, 485.69390869], [0., 0., 1.]], dtype=np.double))
elif scene_id == 10: # args for rio10 scene10
    parser.add_argument('--scene_name', default='scene10')
    parser.add_argument('--train_seq', default='seq10_01')
    parser.add_argument('--data_root', default='{}/scene10/seq10'.format(dataset_folder))
    parser.add_argument('--scene_bbx', default=np.array([-7.12, -0.33, -2.90, 4.16, -0.11, 4.93])) # pre-computed scene bbx.
    parser.add_argument('--far_radiuses', default=[3.73, 1.87, 0.94, 0.47, 0.24, 0.12, 0.06], help='in meter') # to query context points.
    parser.add_argument('--intrinsics', default=np.array([[756.02630615, 0., 270.41879272], [0., 756.83215332, 492.88940430], [0., 0., 1.]], dtype=np.double))
# identify the experiment.
parser.add_argument('--exp_name', required=True) # the folder of checkpoint.


# check.
opt = parser.parse_args()
n_epoch = opt.epoch
if type(n_epoch) is int:
    n_epoch = [n_epoch] * (opt.tree_height - 1)
while len(n_epoch) < opt.tree_height - 1:
    n_epoch.append(n_epoch[-1])
max_step_per_epoch = 1000 # to add to arg list.
# neighbor: visual receptive field.
opt.n2d_lists = np.load(opt.n2d_file)
opt.n_neighb_pts = opt.n2d_lists.shape[1]
# check if match the level.
last_level_idx = opt.n2d_lists.shape[0] - 1
while opt.n2d_lists.shape[0] < opt.tree_height - 1:
    opt.n2d_lists = np.concatenate((opt.n2d_lists, opt.n2d_lists[last_level_idx][np.newaxis, :]))
opt.rot_vec = np.array([[0.0], [0.0], [0.0]])
opt.rot_mat = cv2.Rodrigues(opt.rot_vec)[0]
opt.query_pt_feat_dim, opt.context_pts_feat_dim = 32, 512
ary = pow(2, opt.tree_log2_ary)


# global setting.
g_bool_load_from_pickle = False # False default. True if pickle already computed.
g_checkpoint_dir = 'experiment/{}/checkpoint'.format(opt.exp_name)
g_pickle_train_prefix = 'preprocessed_data/pickle_{}_{}_{}arys_{}lvs_{}_trans{}'.format(opt.dataset, opt.scene_name, ary, opt.tree_height - 1, opt.train_seq, opt.rot_vec.tolist()).replace(' ', '')
g_pickle_leaf_coords_prefix = 'preprocessed_data/leaf_coords_for_{}_{}_{}arys_{}lvs_{}'.format(opt.dataset, opt.scene_name, ary, opt.tree_height-1, opt.rot_vec.tolist()).replace(' ', '')
g_pickle_leaf_gmm_prefix = 'preprocessed_data/leaf_gmm_for_{}_{}_{}arys_{}lvs_{}'.format(opt.dataset, opt.scene_name, ary, opt.tree_height-1, opt.rot_vec.tolist()).replace(' ', '')
if not os.path.exists(g_checkpoint_dir):
    os.makedirs(g_checkpoint_dir)
if not os.path.exists(g_pickle_train_prefix):
    os.makedirs(g_pickle_train_prefix)
if not os.path.exists(g_pickle_leaf_gmm_prefix):
    os.makedirs(g_pickle_leaf_gmm_prefix)
if not os.path.exists(g_pickle_leaf_coords_prefix):
    os.makedirs(g_pickle_leaf_coords_prefix)

