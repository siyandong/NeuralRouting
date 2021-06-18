import numpy as np
import random, pickle, copy, time, cv2, os
import torch
import torch.optim as optim
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import coord_generator
from network import SharedFeatureNet, SharedClassifier
from config import *


class GMM(object):

    def __init__(self, g, mea, cov, wei):
        super(GMM, self).__init__()
        self.gmm = g
        self.means = mea
        self.covars = cov
        self.weights = wei


class SceneArytreeNode(object):

    def __init__(self, box_min, box_max, log2_ary, father=-1):
        super(SceneArytreeNode, self).__init__()
        self.log2_ary = log2_ary
        self.ary = pow(2, log2_ary)
        self.box_min = box_min
        self.box_max = box_max
        len_axis = self.box_max - self.box_min
        self.divide_axis = []
        if self.ary == 8:                   # to debug as standard octree.
            self.divide_axis = [0, 1, 2]    # to debug as standard octree.
        else:
            for i in range(log2_ary):
                self.divide_axis.append(len_axis.argmax())
                len_axis[self.divide_axis[i]] /= 2
        self.node_id = [-1]                 # to identify each node.
        self.father = father
        self.children = []
        self.leaf_coords = []               # scene coordinates if contains leaf nodes.
        self.leaf_gmm = []                  # gmms if contains leaf nodes.

    # space partition.
    def build_children(self):
        self.children.clear()
        for i in range(self.ary):
            box_min = self.box_min.copy()
            box_max = self.box_max.copy()
            for j in range(self.log2_ary):
                ax = self.divide_axis[j]
                if ((i >> j) & 1) == 0:
                    box_max[ax] = (box_min[ax] + box_max[ax]) / 2
                else:
                    box_min[ax] = (box_min[ax] + box_max[ax]) / 2
            self.children.append(SceneArytreeNode(box_min, box_max, self.log2_ary, i))
        for i in range(self.ary):
            self.children[i].node_id = copy.deepcopy(self.node_id)
            self.children[i].node_id.append(i)

    # get ground-truth route by coordinate.
    def get_route(self, coord): 
        ret = 0
        box_min = self.box_min.copy()
        box_max = self.box_max.copy()
        for j in range(self.log2_ary):
            ax = self.divide_axis[j]
            mid = (box_min[ax] + box_max[ax]) / 2
            if coord[ax] <= mid:
                box_max[ax] = mid
            else:
                box_min[ax] = mid
                ret += pow(2, j)
        return ret


class Level(object):

    def __init__(self, level_id, tree):
        self.tree = tree
        self.ary = tree.ary
        self.log2_ary = tree.log2_ary
        self.level_id = level_id
        self.rnet_in = level_id + 1
        # shared neural routing function.
        self.cnet = None # feature encoder for query point.
        self.gnet = None # feature encoder for content points.
        self.rnet = None # route classifier with outlier rejection.
        self.net_optimizer = None
        self.net_scheduler = None
        self.initialize_nets()
        self.node_list = [] # record nodes in this level.

    def initialize_nets(self, cnet=None, gnet=None, rnet=None, LR=opt.LR):
        if cnet == None:
            self.cnet = SharedFeatureNet(opt.query_pt_feat_dim, n_group=1)
        else: self.cnet = cnet
        if gnet == None:
            self.gnet = SharedFeatureNet(opt.context_pts_feat_dim, n_group=1)
        else: self.gnet = gnet
        if rnet == None:
            self.rnet = SharedClassifier(self.rnet_in, opt.query_pt_feat_dim+opt.context_pts_feat_dim, self.ary+1, n_group=1)
        else: self.rnet = rnet
        self.cnet.cuda()
        self.gnet.cuda()
        self.rnet.cuda()
        self.net_optimizer = optim.Adam( # train end-to-end.
            itertools.chain(self.cnet.parameters(), self.gnet.parameters(), self.rnet.parameters()), 
            lr=LR, betas=(0.9, 0.999)) 
        self.net_scheduler = optim.lr_scheduler.StepLR(self.net_optimizer, step_size=opt.LR_step, gamma=0.5)

    def train_nets(self, col_in, pts_in, lab_in): 
        self.cnet.train()
        self.gnet.train()
        self.rnet.train()
        self.net_optimizer.zero_grad()
        # input.
        batch_size = col_in.shape[0]
        x_param = torch.Tensor(batch_size, opt.tree_height - 1).fill_(-1)
        for sid in range(batch_size):
            for lid in range(self.level_id):
                x_param[sid, lid + 1] = lab_in[sid, lid]
        # label.
        lab_gt = lab_in[:, self.level_id]
        if not self.level_id == 0: # simulate dynamic change to create negative sample (input, label).
            n_neg, count = int(opt.fake_data_proportion * batch_size), 0
            if not (n_neg==0):
                for sid in range(batch_size - 1):
                    if not torch.equal(x_param[sid], x_param[sid + 1]):
                        x_param[sid] = x_param[sid + 1]
                        lab_gt[sid] = self.ary
                        count += 1
                        if count >= n_neg:
                            break
        # train.
        x_param = x_param.cuda()
        pred = self.rnet(torch.cat((self.cnet(col_in), self.gnet(pts_in)), 1), x_param, b_level1=self.level_id == 0)
        loss = F.nll_loss(pred, lab_gt)
        loss.backward()
        self.net_optimizer.step()
        result = (pred.data.max(1)[1] - lab_gt).cpu()
        acc = float((result == 0).numpy().sum()) / col_in.shape[0]
        return loss, acc

    def schedule_nets(self):
        self.net_scheduler.step()
        return

    def test_nets(self, col_in, pts_in, param_in, mode='eval'): # output (batch_size, n_class).
        self.cnet.eval()
        self.gnet.eval()
        self.rnet.eval()
        x_param = param_in
        x_param = x_param.cuda()
        batch_size = col_in.shape[0]
        # split the batch if GPU memory is limited.
        max_batch = 2048
        if batch_size <= max_batch: 
            return self.rnet(torch.cat((self.cnet(col_in), self.gnet(pts_in)), 1), x_param, b_level1=self.level_id == 0).data
        pred = torch.Tensor(batch_size, ary + 1).cuda()
        for i in range((batch_size - 1) // max_batch + 1):
            s = slice(i * max_batch, (i + 1) * max_batch)
            pred[s] = self.rnet(torch.cat((self.cnet(col_in[s]), self.gnet(pts_in[s])), 1), x_param[s], b_level1=self.level_id == 0)
        return pred.data

    def save_checkpoint(self, file_prefix, file_suffix):
        torch.save(self.cnet.state_dict(), '{}_cnet_{}'.format(file_prefix, file_suffix))
        torch.save(self.gnet.state_dict(), '{}_gnet_{}'.format(file_prefix, file_suffix))
        torch.save(self.rnet.state_dict(), '{}_rnet_{}'.format(file_prefix, file_suffix))
        return

    def load_checkpoint(self, file_prefix, file_suffix):
        rnet_dict = torch.load('{}_rnet_{}'.format(file_prefix, file_suffix))
        self.rnet_in = rnet_dict['weight_learner_norm_weight1_1.weight'].shape[1]
        self.initialize_nets()
        self.cnet.load_state_dict(torch.load('{}_cnet_{}'.format(file_prefix, file_suffix)))
        self.gnet.load_state_dict(torch.load('{}_gnet_{}'.format(file_prefix, file_suffix)))
        self.rnet.load_state_dict(rnet_dict)
        return

    def save_optimizer(self, file_prefix, file_suffix):
        torch.save(self.net_optimizer.state_dict(), '{}_optimizer_{}'.format(file_prefix, file_suffix))
        torch.save(self.net_scheduler.state_dict(), '{}_scheduler_{}'.format(file_prefix, file_suffix))
        return

    def load_optimizer(self, file_prefix, file_suffix):
        self.net_optimizer.load_state_dict(torch.load('{}_optimizer_{}'.format(file_prefix, file_suffix)))
        self.net_scheduler.load_state_dict(torch.load('{}_scheduler_{}'.format(file_prefix, file_suffix)))
        return


class NetTree(object):

    def __init__(self, scene_box, opt):
        super(NetTree, self).__init__()
        self.root_node = SceneArytreeNode(
            box_min=scene_box[0::2],
            box_max=scene_box[1::2],
            log2_ary=opt.tree_log2_ary)
        self.opt = opt
        self.log2_ary = opt.tree_log2_ary
        self.ary = pow(2, opt.tree_log2_ary)
        self.levels = []

    def initialize_levels(self):
        for lid in range(opt.tree_height - 1):
            self.levels.append(Level(lid, self))
        return

    def train_level(self, level_id, col_in, pts_in, lab_gt):
        self.levels[level_id].train_nets(col_in, pts_in, lab_gt)
        return

    # coord trans if apply rotation augmentation.
    def coord_trans(self, coord): # (, , 3).
        points = coord.reshape((-1, 3))
        points[:, 0:3] = (np.mat(self.opt.rot_mat) * np.mat(points[:, 0:3].T)).T.getA()
        new_coord = points.reshape(coord.shape)
        return new_coord

    def build_multi_level_children(self, node, tree_height):
        if tree_height <= 2: return
        node.build_children()
        for ch in node.children:
            self.build_multi_level_children(ch, tree_height - 1)

    def build_multi_level(self, tree_height=7): # n>2.
        self.build_multi_level_children(self.root_node, tree_height)

    def get_node(self, route_list): # route_list [0, 0, 0] for node_id [-1, 0, 0, 0].
        crt_node = self.root_node
        for i in route_list:
            crt_node = crt_node.children[i]
        return crt_node

    def get_node_list(self, node, node_list):
        node_list.append(node)
        if len(node.children) == 0:
            return
        for cid in range(self.ary):
            self.get_node_list(node.children[cid], node_list)
        return

    def get_route(self, p3d, level=None):
        route_list = []
        crt_node = self.root_node
        for lid in range(999):
            route = crt_node.get_route(p3d)
            route_list.append(route)
            if len(crt_node.children) == 0:
                break
            crt_node = crt_node.children[route]
        if not level == None:
            return route_list[0:level]
        return route_list

    # format raw data for dataset loader.
    def extract_the_list_from_raw_dataset(self, path, fid_list, sample_step=4):
        t_beg = time.time()
        the_list = []
        for fid in tqdm(fid_list):
            rc_route_list = []
            path_coord = '{}/{:06d}_coord.npy'.format(path, fid)
            if os.path.exists(path_coord):
                coord = self.coord_trans(np.load(path_coord))
            else:
                path_depth = '{}/{}'.format(path, opt.depth_format.format(fid))
                path_pose = '{}/{}'.format(path, opt.pose_format.format(fid))
                depth = cv2.imread(path_depth, cv2.IMREAD_UNCHANGED) / 1000.0
                pose = np.loadtxt(path_pose)
                cg = coord_generator.CoordGenerator(opt.intrinsics, opt.image_width, opt.image_height)
                coord, _ = cg.depth_pose_2coord(depth, pose)
                coord = self.coord_trans(coord)
            r_ids, c_ids = list(range(0, coord.shape[0], 4)), list(range(0, coord.shape[1], 4))
            coord = coord[r_ids, :, :][:, c_ids, :]
            # for r in range(coord.shape[0]):
            #     for c in range(coord.shape[1]):
            for r in range(0, coord.shape[0], sample_step):     # sampling.
                for c in range(0, coord.shape[1], sample_step): # sampling.
                    if (coord[r, c] == np.array([0., 0., 0.])).all():
                        continue
                    route_list = self.get_route(coord[r, c])
                    rc_route_list.append((r, c, route_list))
            the_list.append((fid, rc_route_list))
        t_end = time.time()
        print('total time {:.2f}s.'.format(t_end - t_beg))
        return the_list

    def save_the_list_to_file(self, the_list, path_prefix):
        with open('{}/the_list.pk'.format(path_prefix), 'wb') as fp:
            pickle.dump(the_list, fp)
        return

    def load_the_list_from_file(self, path_prefix):
        path = '{}/the_list.pk'.format(path_prefix)
        if not os.path.exists(path):
            print('no pickle file exists.')
            return None
        with open(path, 'rb') as fp:
            the_list = pickle.loads(fp.read())
        return the_list

    def init_leaf_coords(self, path, fid_list, frame_sample_step=10, pixel_sample_step=4):
        t_beg = time.time()
        if len(fid_list) > 8000:
            frame_sample_step = 15
            print('#frame>8000, increase sample step.')
        for idx in tqdm(range(0, len(fid_list), frame_sample_step)):
            fid = fid_list[idx]
            path_coord = '{}/{:06d}_coord.npy'.format(path, fid)
            if os.path.exists(path_coord):
                coord = self.coord_trans(np.load(path_coord))
            else:
                path_depth = '{}/{}'.format(path, opt.depth_format.format(fid))
                path_pose = '{}/{}'.format(path, opt.pose_format.format(fid))
                depth = cv2.imread(path_depth, cv2.IMREAD_UNCHANGED) / 1000.0
                pose = np.loadtxt(path_pose)
                cg = coord_generator.CoordGenerator(opt.intrinsics, opt.image_width, opt.image_height)
                coord, _ = cg.depth_pose_2coord(depth, pose)
                coord = self.coord_trans(coord)
            # for r in range(coord.shape[0]):
            #     for c in range(coord.shape[1]):
            for r in range(0, coord.shape[0], pixel_sample_step):       # sampling.
                for c in range(0, coord.shape[1], pixel_sample_step):   # sampling.
                    if (coord[r, c] == np.array([0., 0., 0.])).all():
                        continue
                    route_list = self.get_route(coord[r, c])
                    node = self.get_node(route_list[0:-1])
                    if len(node.leaf_coords) == 0:
                        node.leaf_coords = []
                        for i in range(self.ary):
                            node.leaf_coords.append([])
                    node.leaf_coords[route_list[-1]].append(coord[r, c])
            #print('frame {} done.'.format(fid))
        t_end = time.time()
        print('total time {:.2f}s to init leaf coords.'.format(t_end - t_beg))
        return

    def save_leaf_coords_to_file(self, path_prefix):
        node_list = []
        self.get_node_list(self.root_node, node_list)
        for node in tqdm(node_list):
            if len(node.leaf_coords) == 0:
                continue
            # sampling.
            for lid in range(len(node.leaf_coords)):
                if len(node.leaf_coords[lid]) < opt.n_coord_per_leaf:
                    continue
                # print('{} coords'.format(len(node.leaf_coords[lid])))
                np.random.shuffle(node.leaf_coords[lid])
                node.leaf_coords[lid] = node.leaf_coords[lid][0:opt.n_coord_per_leaf]
                # print('{} coords'.format(len(node.leaf_coords[lid])))
            with open('{}/{}_leaf_coords.pk'.format(path_prefix, node.node_id), 'wb') as fp:
                pickle.dump(node.leaf_coords, fp)
        return

    def load_leaf_coords_from_file(self, path_prefix):
        node_list = []
        self.get_node_list(self.root_node, node_list)
        for node in tqdm(node_list):
            path = '{}/{}_leaf_coords.pk'.format(path_prefix, node.node_id)
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as fp:
                node.leaf_coords = pickle.loads(fp.read())
        return

    def inference(self, pt_in, nb_ms_in, beam=1, multi_leaf=[]): # beam <= ary
        batch_size = pt_in.shape[0]
        x_param = torch.Tensor(batch_size * beam, opt.tree_height).fill_(-1)
        confidence = torch.zeros(batch_size * beam)
        confidence[::beam] = 1
        pt_in = pt_in.repeat_interleave(beam, 0)
        nb_ms_in = nb_ms_in.repeat_interleave(beam, 0)
        for lid in range(opt.tree_height - 1):
            col_in = pt_in.cuda()
            pts_in = nb_ms_in[:, lid, :, :].cuda()
            #t0 = time.time()
            pred = self.levels[lid].test_nets(col_in, pts_in, x_param).cpu()
            if beam > 1:
                conf, key = pred.softmax(dim=1).topk(beam, dim=1)
                conf *= confidence.reshape(-1, 1)
                ind = conf.reshape(batch_size, -1).topk(beam, dim=1)[1]
                x_param = x_param.reshape(batch_size, beam, -1)
                confidence = confidence.reshape(batch_size, -1)
                key = key.reshape(batch_size, -1)
                conf = conf.reshape(batch_size, -1)
                for i in range(batch_size):
                    x_param[i] = x_param[i].repeat_interleave(beam, 0)[ind[i]]
                    x_param[i, :, lid + 1] = key[i][ind[i]]
                    confidence[i] *= conf[i][ind[i]]
                x_param = x_param.reshape(batch_size * beam, -1)
                confidence = confidence.reshape(-1)
            elif lid + len(multi_leaf) >= opt.tree_height - 1:
                topk = multi_leaf[lid - opt.tree_height + 1]
                pt_in = pt_in.repeat_interleave(topk, 0)
                nb_ms_in = nb_ms_in.repeat_interleave(topk, 0)
                x_param = x_param.repeat_interleave(topk, 0)
                confidence = confidence.repeat_interleave(topk, 0)
                value, key = pred.topk(topk, dim=1)
                key = key.flatten()
                conf = value.softmax(dim=1).flatten()
                x_param[:, lid + 1] = key
                confidence *= conf
            else:
                topk = 1
                value, key = pred.topk(topk, dim=1)
                key = key.flatten()
                conf = value.softmax(dim=1).flatten()
                x_param[:, lid + 1] = key
                confidence *= conf
        return confidence.reshape(batch_size, -1).numpy(), \
               x_param[:, 1:].reshape(batch_size, -1, opt.tree_height - 1).to(torch.int)


# if __name__ == '__main__':
#     print('done.')

