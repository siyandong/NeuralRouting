import numpy as np
import random, pickle, copy, cv2, os, time
from glob import glob
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import mixture
from tensorboardX import SummaryWriter
from tqdm import tqdm
import scene_partition_tree
import dataset_loader
from config import *
from log import *


# backup source code.
os.system('cp ./*.py ./experiment/{}'.format(opt.exp_name))


if __name__ == '__main__':

    # build scene space partition tree structure.
    nt = scene_partition_tree.NetTree(opt.scene_bbx, opt=opt)
    nt.build_multi_level(nt.opt.tree_height)

    # prepare dataset for neural routing function. 
    train_data_root = os.path.join(opt.data_root, opt.train_seq)
    train_frame_num = len(glob(os.path.join(train_data_root, opt.color_format.format(0).replace('000000', '??????'))))
    train_frame_list = range(train_frame_num)
    if not g_bool_load_from_pickle:
        print('building dataset...')
        the_list = nt.extract_the_list_from_raw_dataset(train_data_root, train_frame_list)

        nt.save_the_list_to_file(the_list, g_pickle_train_prefix)
    else:
        print('loading dataset from {}'.format(g_pickle_train_prefix))
        the_list = nt.load_the_list_from_file(g_pickle_train_prefix)

    # compute gmm in tree leaf nodes.
    if True:
        # save raw coords in leaf nodes.
        print('initializing leaf coords...')
        if not g_bool_load_from_pickle:
            nt.init_leaf_coords(train_data_root, train_frame_list)
            nt.save_leaf_coords_to_file(g_pickle_leaf_coords_prefix)
        else:
            nt.load_leaf_coords_from_file(g_pickle_leaf_coords_prefix)
        # fit gmm in each leaf node.
        print('fitting leaf gmms...')
        node_list = []
        nt.get_node_list(nt.root_node, node_list)
        valid_num = 0
        for index, node in enumerate(tqdm(node_list)):
            # if load gmm from file.
            leaf_gmm_path = '{}/{}_leaf_gmm.pk'.format(g_pickle_leaf_gmm_prefix, node.node_id)
            if os.path.exists(leaf_gmm_path):
                with open(leaf_gmm_path, 'rb') as fp:
                    node.leaf_gmm = pickle.loads(fp.read())
                    valid_num+=1
                    continue
            # fit gmm.
            if True:
                if len(node.leaf_coords) == 0: continue
                for lid in range(len(node.leaf_coords)):
                    coords = node.leaf_coords[lid]
                    if len(coords) < 10:
                        node.leaf_gmm.append(None)
                        continue
                    crd_ary = np.array(coords)
                    # clustering as initialization.
                    bandwidth = estimate_bandwidth(crd_ary[:, 0:3], quantile=0.2, n_samples=500)
                    bandwidth *= 0.5
                    bandwidth = 0.05
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    ms.fit(crd_ary[:, 0:3])
                    labels = ms.labels_
                    cluster_centers = ms.cluster_centers_
                    labels_unique = np.unique(labels)
                    n_cluster = len(labels_unique)
                    # fit gmm based on clusters.
                    gmm = mixture.GaussianMixture(n_components=n_cluster)
                    gmm.fit(crd_ary[:, 0:3])
                    means = gmm.means_
                    covars = gmm.covariances_
                    weights = gmm.weights_
                    # save to the leaf node.
                    node.leaf_gmm.append(scene_partition_tree.GMM(gmm, means, covars, weights))
                # check.
                if not len(node.leaf_coords) == len(node.leaf_gmm):
                    print('error: please clear gmm files and re-fit.')
                    exit()
                # save to file.
                with open(leaf_gmm_path, 'wb') as fp:
                    pickle.dump(node.leaf_gmm, fp)
                    #print('saved leaf gmm to file.')
        print('{} valid pickle files.'.format(valid_num))

    # initialize neural routing function in each level.
    nt.initialize_levels()

    # start training level by level.
    total_step = 0
    toatl_time = 0.0
    writer = SummaryWriter('runs/{}'.format(opt.exp_name))
    for lid in range(opt.tree_height - 1):
    #for lid in range(0, opt.tree_height - 2): # for lv1~4.
    #for lid in range(4, opt.tree_height - 1): # for lv5.
    #for lid in []:                            # for specific levels.

        start_step = 0
        # dataset loader for the level.
        dataset = dataset_loader.LevelDataset_PPF(train_data_root, the_list, neighbor_da2d=opt.n2d_lists[lid], 
            far_radius=opt.far_radiuses[lid]) # specified_node???
        loader = dataset_loader.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True,
            num_workers=opt.num_workers)

        # train the level.
        total_step = start_step
        history_mean_loss, history_median_loss = 999., 999.
        history_mean_acc, history_median_acc = 0., 0.
        for epoch in range(start_step // max_step_per_epoch, n_epoch[lid]): # for each epoch.
            t_end = time.time()
            toatl_time = 0.0
            epoch_losses, epoch_accs = [], []
            for step, batch_sample in enumerate(loader):
                total_step += 1
                t_beg = time.time()
                toatl_time += t_beg - t_end
                # parse data.
                pt_in = batch_sample[0].reshape(-1, 7, 1)
                nb_in = batch_sample[1].reshape(-1, 7, opt.n_neighb_pts)
                route_labs = batch_sample[2].reshape(-1, opt.tree_height - 1)
                col_in = pt_in.cuda()
                batch_size = col_in.shape[0]
                pts_in = nb_in[:, :, :].cuda()
                lab_gt = route_labs.long().cuda() # (batch_size, n_level).
                if batch_size == 0:
                    continue
                train_loss, train_acc = nt.levels[lid].train_nets(col_in, pts_in, lab_gt)
                # save checkpoint.
                if (total_step % (max_step_per_epoch * 10) == 0) and (epoch > n_epoch[lid]/2):
                    file_prefix = '{}/l{}'.format(g_checkpoint_dir, lid)
                    file_suffix = 'step{}'.format(total_step)
                    nt.levels[lid].save_checkpoint(file_prefix, file_suffix)
                    nt.levels[lid].save_optimizer(file_prefix, file_suffix)
                t_end = time.time()
                toatl_time += t_end - t_beg
                # record loss and accuracy.
                epoch_losses.append(float(train_loss))
                epoch_accs.append(float(train_acc * 100))
                if step % 10 == 0:
                    print('mean total time per step {:.2f}s\n'.format(toatl_time / (step+1)))
                    print('level {} total_step {} train_loss {:.4f} train_acc {:.2f}%'.format(
                        lid+1, total_step, train_loss, train_acc * 100))
                    # write to tensorboard.
                    writer.add_scalar('level{} train loss'.format(lid+1), train_loss, total_step)
                    writer.add_scalar('level{} train acc'.format(lid+1), train_acc * 100, total_step)
                    file = open(g_log_file_path, 'a')
                    file.write('total_step {} train_loss {:.4f} train_acc {:.2f}%\n'.format(
                        total_step, train_loss, train_acc * 100))
                    file.close()
                    # write to file.
                    file = open(g_log_file_path, 'a')
                    file.write('level {} step {} train tree timing {:.2f}s\n'.format(lid+1, total_step, t_end - t_beg))
                    file.write('mean total time per step {:.2f}s\n'.format(toatl_time / (step+1)))
                    file.close()
                if step >= max_step_per_epoch: # end the epoch.
                    break

            # record epoch loss and accuracy.
            epoch_mean_loss, epoch_median_loss = np.mean(epoch_losses), np.median(epoch_losses)
            epoch_mean_acc, epoch_median_acc = np.mean(epoch_accs), np.median(epoch_accs)
            writer.add_scalar('level{} epoch train loss'.format(lid+1), epoch_median_loss, epoch)
            writer.add_scalar('level{} epoch train acc'.format(lid+1), epoch_median_acc, epoch)

            # early termination condition.
            if epoch > (n_epoch[lid] / 3 * 2) and epoch_mean_loss >= history_mean_loss and \
                    epoch_median_loss >= history_median_loss and epoch_mean_acc <= history_mean_acc and \
                    epoch_median_acc <= history_median_acc:
                break
            history_mean_loss, history_median_loss, history_mean_acc, epoch_median_acc = epoch_mean_loss, epoch_median_loss, epoch_mean_acc, epoch_median_acc

            # step LR by epoch.
            nt.levels[lid].schedule_nets()


    print('training finished.')
    exit()

