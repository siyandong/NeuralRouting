import numpy as np
import random, pickle, torch, copy, cv2
from glob import glob
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import mixture
from tqdm import tqdm
import scene_partition_tree
import dataset_loader
from config import *


# downsample.
g_n_pts_per_frame = opt.g_n_pts_per_frame
sampled_image_width, sampled_image_height = int(opt.image_width / 4), int(opt.image_height / 4)
# output directory.
g_prefix_predictionsImage = 'gmm_prediction/_{}_{}lvs_top{}_{}_step{}'.format(
    opt.exp_name, opt.tree_height-1, opt.multi_leaf, opt.test_seq, opt.idx_step).replace(' ', '')
g_prefix_predictionsImage = g_prefix_predictionsImage + '_{}pts'.format(g_n_pts_per_frame)
if not os.path.exists(g_prefix_predictionsImage):
    os.makedirs(g_prefix_predictionsImage)
# the data to evaluate.
crt_data_root = os.path.join(opt.data_root, opt.test_seq)
crt_frame_num = len(glob(os.path.join(crt_data_root, opt.color_format.format(0).replace('000000', '??????'))))
crt_frame_list = range(crt_frame_num)


def union_gmm(gmms, confs=None):
    n = len(gmms)
    if confs == None:
        # confs = [1. / n] * n
        confs = [1.] * n
    gmm = scene_partition_tree.GMM(None,
        np.concatenate([gmm.means for gmm in gmms], axis=0),
        np.concatenate([gmm.covars for gmm in gmms], axis=0),
        np.concatenate([(gmm.weights * conf)[:, np.newaxis]
            for gmm, conf in zip(gmms, confs)]).flatten())
    return gmm

def write_predictionsImage(path, predictions):
    file = open(path, 'w')
    count = 0 # num of valid prediction.
    for idx in range(len(predictions)):
        if predictions[idx] is None:
            continue
        count += 1
    file.write('{}\n'.format(count))
    for idx in range(len(predictions)):
        if predictions[idx] is None:
            continue
        # gmm = predictions[idx]
        gmm = predictions[idx][0]
        confidence = predictions[idx][1]
        file.write('{}\n{}\n{}\n'.format(idx, confidence, len(gmm.means))) # idx, confidence, n_cluster.
        for cid in range(len(gmm.means)):
            file.write('{} '.format(gmm.weights[cid])) # weight.
            file.write('0 0 0 {} {} {}\n'.format(gmm.means[cid][0], gmm.means[cid][1], gmm.means[cid][2])) # w/o color, only position.
            inv_corvar = np.linalg.inv(gmm.covars[cid]) # inv_corvar.
            file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                inv_corvar[0, 0], inv_corvar[0, 1], inv_corvar[0, 2],
                inv_corvar[1, 0], inv_corvar[1, 1], inv_corvar[1, 2],
                inv_corvar[2, 0], inv_corvar[2, 1], inv_corvar[2, 2]))
    file.close()


if __name__ == '__main__':

    # build scene space partition tree structure.
    nt = scene_partition_tree.NetTree(opt.scene_bbx, opt=opt)
    nt.build_multi_level(opt.tree_height)

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

    # set neural routing functions: load from trained checkpoint.
    nt.initialize_levels()
    print('routing functions loading checkpoint...')
    for lid in range(opt.tree_height - 1):
        file_prefix = '{}/l{}'.format(g_checkpoint_dir, lid)
        file_suffix = 'step{}'.format(n_epoch[lid] * max_step_per_epoch)
        nt.levels[lid].load_checkpoint(file_prefix, file_suffix)
    
    # dataset.
    the_list = crt_frame_list
    the_list = the_list[::opt.idx_step]
    dataset = dataset_loader.TestDataset_PPF(crt_data_root, the_list, g_n_pts_per_frame, neighbor_da2d=opt.n2d_lists)
    loader = dataset_loader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers)

    # inference.
    print('inferring correspondences...')
    ps_remain, ps_correct = [], []
    inlier_recalls, inlier_precisions, outlier_recalls = [], [], []
    topk_inlier_recalls, topk_inlier_precisions, topk_outlier_recalls = [], [], []
    p_rejections_wrong_inlier, p_rejections_outlier = [], []
    n_metric_list = []
    n_correct_list = []
    for step, batch_sample in enumerate(tqdm(loader)):
        route_labs = None
        pt_in = batch_sample[0][0]
        nb_ms_in = batch_sample[1][0]
        if not type(batch_sample[2]) == type(-1):
            route_labs = batch_sample[2][0].to(torch.int32)
        fid = batch_sample[3][0].int().item()
        rc_list = batch_sample[4][0]
        batch_size = rc_list.shape[0]
        predImg_path = '{}/predictionsImage_{}.txt'.format(g_prefix_predictionsImage, fid)

        # predict route. 
        # to handle large batch size. 2021/03/12. 
        if batch_size <= 2048:
            confidence, routes_pred = nt.inference(pt_in, nb_ms_in, beam=1, multi_leaf=opt.multi_leaf)
        else:
            confidence, routes_pred = nt.inference(
                pt_in[0:2048], 
                nb_ms_in[0:2048], 
                beam=1, multi_leaf=opt.multi_leaf)
            for sid_beg in range(2048, batch_size, 2048):
                sid_end = sid_beg + 2048
                if sid_end > batch_size: sid_end = batch_size
                conf, rout = nt.inference(
                    pt_in[sid_beg:sid_end], 
                    nb_ms_in[sid_beg:sid_end], 
                    beam=1, multi_leaf=opt.multi_leaf)
                confidence = np.concatenate((confidence, conf), axis=0)
                routes_pred = np.concatenate((routes_pred, rout), axis=0)
        # to handle large batch size. 2021/03/12. 

        beam_size = routes_pred.shape[1]

        # save prediction.
        predictions = []
        for i in range(sampled_image_width * sampled_image_height):
            predictions.append(None)
        for sid in range(batch_size): # for each sample.
            valids = [True] * beam_size
            routes = []
            for kid in range(beam_size): routes.append([])
            for kid in range(beam_size):
                for lid in range(opt.tree_height - 1):
                    if not valids[kid]:
                        break
                    if routes_pred[sid][kid][lid] == ary:
                        valids[kid] = False
                        continue
                    routes[kid].append(routes_pred[sid][kid][lid])
            valid = False
            for kid in range(routes_pred.shape[1]):
                if valids[kid]:
                    valid = True
                    break
            if valid:
                # get the last level node.
                gmms = []
                confs = []
                for kid in range(beam_size):
                    if valids[kid]:
                        node = nt.get_node(routes[kid][0:-1])
                        if len(node.leaf_gmm) == 0:
                            #print('no leaf.')
                            continue
                        gmm = node.leaf_gmm[routes[kid][-1]]
                        if gmm is None:
                            #print('no gmm.')
                            continue
                        if len(gmm.means) == 0:
                            #print('no mode.')
                            continue
                        if len(gmm.means) >= 20:
                            #print('invalid gmm.')
                            continue
                        gmms.append(gmm)
                        confs.append(confidence[sid][kid])
                # merge gmm
                if len(gmms) == 0:
                    continue
                gmm = union_gmm(gmms, confs if 1 else None)
                if gmm == None:
                    continue
                # record
                r, c = rc_list[sid].int()
                idx = r * sampled_image_width + c
                predictions[idx] = (gmm, 0.8)
        # write to file
        write_predictionsImage(predImg_path, predictions)
        #print('frame {} saved.'.format(fid))


    print('done.')
    exit()

