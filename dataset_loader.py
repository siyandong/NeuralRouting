import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
from PIL import Image
import os, cv2, copy, time
from config import *


# args.
image_height, image_width = opt.image_height, opt.image_width
intrinsics = opt.intrinsics
close_radius, far_radiuses = 0, opt.far_radiuses
n_neighb_pts = opt.n_neighb_pts


def isSon(son, fa):
    for i in range(len(fa)):
        if son[i] != fa[i]:
            return False
    return True


# todo: to be migrated...
def depth2local(depth):  # depth: float32, meter.
    cx, cy, fx, fy = intrinsics[0, 2], intrinsics[1, 2], intrinsics[0, 0], intrinsics[1, 1]
    u_base = np.tile(np.arange(image_width), (image_height, 1))
    v_base = np.tile(np.arange(image_height)[:, np.newaxis], (1, image_width))
    X = (u_base - cx) * depth / fx
    Y = (v_base - cy) * depth / fy
    coord_local = np.stack((X, Y, depth), axis=2)
    return coord_local
def partial_pts(pts_all_in, p, r_min, r_max):  # pts_all_in.shape (#points, #channel)
    pts_all = copy.deepcopy(pts_all_in)
    p_mat = p[np.newaxis, 0:3].repeat(pts_all.shape[0], axis=0)
    norms = np.linalg.norm((p_mat - pts_all[:, 0:3]), axis=1)
    return pts_all[np.logical_and(norms >= r_min, norms <= r_max)]
def sample_pts(pts_in, num):  # pts_in.shape (#points, #channel)
    pts = copy.deepcopy(pts_in)
    while pts.shape[0] < num:
        pts = np.concatenate((pts, pts), axis=0)
    rand_ids = np.arange(pts.shape[0])
    np.random.shuffle(rand_ids)
    return pts[rand_ids[0:num], :]
def sample_pts_rc(pts_in, rcs_in, num):  # pts_in.shape (#points, #channel)
    pts = copy.deepcopy(pts_in)
    rcs = copy.deepcopy(rcs_in)
    while pts.shape[0] < num:
        pts = np.concatenate((pts, pts), axis=0)
    rand_ids = np.arange(pts.shape[0])
    np.random.shuffle(rand_ids)
    return pts[rand_ids[0:num], :], rcs_in[rand_ids[0:num], :]
def sample_pts9d_r3d(pts_in, num, radius):  # pts_in.shape (#points, #channel)
    pts = copy.deepcopy(pts_in)
    thresh = 500
    # remove background by 3d radius
    xyz = pts[:, 0:3]
    pts = pts[np.linalg.norm(xyz, axis=1) <= radius]
    # print('pt num after r3d {}'.format(pts.shape[0]))
    if pts.shape[0] < thresh:  # avoid infinite loop.
        return None
    while pts.shape[0] < num:
        pts = np.concatenate((pts, pts), axis=0)
    rand_ids = np.arange(pts.shape[0])
    np.random.shuffle(rand_ids)
    return pts[rand_ids[0:num], :]
def shift_pts(pts_in, cen):  # pts_in.shape (#points, #channel)
    pts = copy.deepcopy(pts_in)
    cen_mat = cen[np.newaxis, :].repeat(pts.shape[0], axis=0)
    pts[:, 0:3] = pts[:, 0:3] - cen_mat
    return pts
def shift_pts6d(pts_in, cen):  # pts_in.shape (#points, #channel)
    pts = copy.deepcopy(pts_in)
    cen_mat = cen[np.newaxis, :].repeat(pts.shape[0], axis=0)
    pts[:, :] = pts[:, :] - cen_mat
    return pts
def shift_pts9d(pts_in, cen):  # pts_in.shape (#points, #channel)
    cpt = copy.deepcopy(cen)
    cpt[3:6] = np.zeros(3)  # remove shift of normal
    pts = copy.deepcopy(pts_in)
    cpt_mat = cpt[np.newaxis, :].repeat(pts.shape[0], axis=0)
    pts[:, :] = pts[:, :] - cpt_mat
    return pts


def make_ppf(pts9d, cen9d):  # (N,9), (9,)
    # prepare
    n_pts = pts9d.shape[0]
    d = pts9d[:, 0:3]
    n2 = pts9d[:, 3:6]
    n1 = np.repeat(cen9d[3:6].reshape(1, 3), n_pts, axis=0)
    # ppf
    dim1 = np.linalg.norm(d, axis=1).reshape(n_pts, 1)
    d = d / (dim1.reshape(n_pts, 1))
    dim2 = np.sum(n1 * d, axis=1).reshape(n_pts, 1)
    dim3 = np.sum(n2 * d, axis=1).reshape(n_pts, 1)
    dim4 = np.sum(n1 * n2, axis=1).reshape(n_pts, 1)
    ppf = np.concatenate((dim1, dim2, dim3, dim4), axis=1)
    ppf7d = np.concatenate((ppf, pts9d[:, 6:9]), axis=1)
    return ppf7d

def compute_points_normal(pts):
    raw_shape = pts.shape
    normal = np.zeros((raw_shape))  # (r,c,3)
    t0 = time.time()
    for r in range(2, raw_shape[0] - 2):
        for c in range(2, raw_shape[1] - 2):
            pts_local = pts[r - 2:r + 3, c - 2:c + 3, :]  # (5,5,3)
            pts_local = pts_local.reshape(-1, 3)  # (N,3)
            pts_local = pts_local[np.linalg.norm(pts_local - pts[r, c, :], axis=1) < 0.1]  # remove outliers.
            if pts_local.shape[0] < 4:
                continue
            pts_local = pts_local - np.mean(pts_local, axis=0)
            C = pts_local.T @ pts_local / pts_local.shape[0]
            e, v = np.linalg.eig(C)
            d = v[:, np.where(e == np.min(e))[0][0]]
            n = d / np.linalg.norm(d)
            if np.dot(n, np.array([0, 0, 1])) > 0:
                n = -n
            normal[r, c, :] = n
    t1 = time.time()
    print('preprocess data: compute normal cost {:.2f}s'.format(t1 - t0))
    return normal


# for depth adaptive 2d
def partial_pts_2d(pts_rc, cen_rc, list_drdc):
    result = None
    r_max, c_max = int(image_height / 4 - 1), int(image_width / 4 - 1)
    mat_drdc = (np.array(list_drdc) / 4).astype(int)
    mat_cen_rc = np.array(cen_rc)
    mat_targ_rc = cen_rc + mat_drdc
    mat_targ_rc[mat_targ_rc < 0] = 0
    targ_r = mat_targ_rc[:, 0]
    targ_r[targ_r > r_max] = r_max
    targ_c = mat_targ_rc[:, 1]
    targ_c[targ_c > c_max] = c_max
    result = pts_rc[targ_r, targ_c]
    return copy.deepcopy(result)


# for depth adaptive 2d
def partial_pts_2d_rc(pts_rc, cen_rc, list_drdc):
    result = None
    r_max, c_max = int(image_height / 4 - 1), int(image_width / 4 - 1)
    mat_drdc = (np.array(list_drdc) / 4).astype(int)
    mat_cen_rc = np.array(cen_rc)
    mat_targ_rc = cen_rc + mat_drdc
    mat_targ_rc[mat_targ_rc < 0] = 0
    targ_r = mat_targ_rc[:, 0]
    targ_r[targ_r > r_max] = r_max
    targ_c = mat_targ_rc[:, 1]
    targ_c[targ_c > c_max] = c_max
    result = pts_rc[targ_r, targ_c]
    return copy.deepcopy(result), copy.deepcopy(
        np.concatenate((targ_r.reshape(targ_r.shape[0], 1), targ_c.reshape(targ_c.shape[0], 1)), axis=1))


# for depth adaptive 2d with dynamics label
def partial_pts_2d_with_label(pts_rc, cen_rc, list_drdc, mask):  # mask: 0 for static pixel, 255 for dynamic pixel.
    result = None
    r_max, c_max = int(image_height / 4 - 1), int(image_width / 4 - 1)
    mat_drdc = (np.array(list_drdc) / 4).astype(int)
    mat_cen_rc = np.array(cen_rc)
    mat_targ_rc = cen_rc + mat_drdc
    mat_targ_rc[mat_targ_rc < 0] = 0
    targ_r = mat_targ_rc[:, 0]
    targ_r[targ_r > r_max] = r_max
    targ_c = mat_targ_rc[:, 1]
    targ_c[targ_c > c_max] = c_max
    m1 = np.zeros((mask.shape[0], mask.shape[1]))
    m1[mask == 0] = 1
    m2 = np.zeros((mask.shape[0], mask.shape[1]))
    m2[targ_r, targ_c] = 1
    m3 = np.logical_and(m1, m2)
    result = pts_rc[m3]
    return copy.deepcopy(result)


class LevelDataset_PPF(Dataset):

    def __init__(self, data_dir, the_list, n_pts_per_frame=opt.n_pts_per_frame, neighbor_da2d=None, far_radius=None,
                 enable_color_aug=True, specified_node=None):
        super().__init__()
        self.data_dir, self.the_list = data_dir, the_list
        self.n_pts_per_frame = n_pts_per_frame
        self.neighbor_da2d = neighbor_da2d  # (n_pts, dim_pt).
        self.far_radius = far_radius  # scalar.
        self.enable_color_aug = enable_color_aug
        self.specified_node = specified_node

    def __len__(self):
        return len(self.the_list)

    def __getitem__(self, idx):
        fid, rc_route = self.the_list[idx]
        # load 
        depth = cv2.imread('{}/{}'.format(self.data_dir, opt.depth_format.format(fid)), cv2.IMREAD_UNCHANGED) / 1000.0
        color = cv2.imread('{}/{}'.format(self.data_dir, opt.color_format.format(fid)), cv2.IMREAD_UNCHANGED)[:, :, 0:3]
        # color jitter
        if self.enable_color_aug:
            img = Image.fromarray(color)
            if np.random.rand() < 0.5:
                img = tfs.ColorJitter(brightness=1.)(img)
            if np.random.rand() < 0.5:
                img = tfs.ColorJitter(contrast=1.)(img)
            if np.random.rand() < 0.5:
                img = tfs.ColorJitter(saturation=1.)(img)
            color = np.array(img)
        if np.max(color) > 1:
            color = color / 255. - 0.5
        local = depth2local(depth)
        r_ids, c_ids = list(range(0, image_height, 4)), list(range(0, image_width, 4))
        depth, color, local = depth[r_ids, :][:, c_ids], color[r_ids, :, :][:, c_ids, :], local[r_ids, :, :][:, c_ids, :]
        # normal by 3d neighbor plane fitting.
        normal_path = '{}/frame-{:06d}.scaled.normal.npy'.format(self.data_dir, fid)
        if os.path.exists(normal_path):
            # print('fid {}'.format(fid)) # to debug rio10 scene09 10
            # normal = np.load(normal_path)
            if os.path.getsize(normal_path) > 1:
                normal = np.load(normal_path, encoding='bytes', allow_pickle=True)
            else:
                normal = compute_points_normal(local)
                np.save(normal_path, normal)
        else:
            normal = compute_points_normal(local)
            np.save(normal_path, normal)
        lclnmlclr = np.concatenate((np.concatenate((local, normal), axis=2), color), axis=2)
        # build a patch
        rand_ids = np.arange(len(rc_route))
        np.random.shuffle(rand_ids)
        selected_ids = rand_ids[0:self.n_pts_per_frame * 2]  # more candidates
        pt_in = torch.zeros((self.n_pts_per_frame, 7, 1))
        nb_in = torch.zeros((self.n_pts_per_frame, 7, opt.n_neighb_pts))
        route_labs = torch.zeros((self.n_pts_per_frame, opt.tree_height - 1)).fill_(ary)
        rc_list = []
        # da2d+3d neighbor
        if not self.neighbor_da2d is None:
            sid = 0
            for tmp_idx in range(len(selected_ids)):
                r, c = rc_route[selected_ids[tmp_idx]][0], rc_route[selected_ids[tmp_idx]][1]
                if np.isnan(lclnmlclr[r, c, 3]):
                    continue
                if self.specified_node:
                    if not isSon(rc_route[selected_ids[tmp_idx]][2], self.specified_node):
                        continue
                route_labs[sid] = torch.Tensor(rc_route[selected_ids[tmp_idx]][2])
                rc_list.append([r, c])
                pt_in[sid] = torch.Tensor(
                    np.concatenate((np.array([[0.], [0.], [0.], [0.]]), color[r, c, 0:3][:, np.newaxis]), axis=0))
                da2d_list = (np.array(self.neighbor_da2d) / depth[r, c]).astype(int)
                # ppf
                pts9d = shift_pts9d(sample_pts(partial_pts_2d(lclnmlclr, (r, c), da2d_list), opt.n_neighb_pts),
                                    lclnmlclr[r, c, :])
                cen9d = copy.deepcopy(lclnmlclr[r, c, :])
                cen9d[0:3] = np.zeros(3)
                ppf7d = make_ppf(pts9d, cen9d)  # (N,9), (9,)
                ppf7d[np.isnan(ppf7d)] = 0.0
                nb_in[sid] = torch.Tensor(ppf7d).transpose(1, 0)
                # remove background by 3d radius
                xyz = pts9d[:, 0:3]
                ids_out_of_bound = np.linalg.norm(xyz, axis=1) > self.far_radius
                nb_in[sid, :, ids_out_of_bound] = 0.
                # count
                sid += 1
                if sid >= self.n_pts_per_frame:
                    break
        pt_in = pt_in[:sid]
        nb_in = nb_in[:sid]
        route_labs = route_labs[:sid]
        return pt_in, nb_in, route_labs, fid, torch.Tensor(np.array(rc_list))


class TestDataset_PPF(Dataset):

    def __init__(self, data_dir, the_list, n_pts_per_frame=opt.n_pts_per_frame, neighbor_da2d=None):
        super().__init__()
        self.data_dir, self.the_list = data_dir, the_list
        self.n_pts_per_frame = n_pts_per_frame
        self.neighbor_da2d = neighbor_da2d  # list of (n_pts, dim_pt)

    def __len__(self):
        return len(self.the_list)

    def __getitem__(self, idx):
        fid = self.the_list[idx]
        # load 
        depth = cv2.imread('{}/{}'.format(self.data_dir, opt.depth_format.format(fid)), cv2.IMREAD_UNCHANGED) / 1000.0
        color = cv2.imread('{}/{}'.format(self.data_dir, opt.color_format.format(fid)), cv2.IMREAD_UNCHANGED)[:, :, 0:3]
        if np.max(color) > 1:
            color = color / 255. - 0.5
        local = depth2local(depth)
        r_ids, c_ids = list(range(0, image_height, 4)), list(range(0, image_width, 4))
        depth, color, local = depth[r_ids, :][:, c_ids], color[r_ids, :, :][:, c_ids, :], local[r_ids, :, :][:, c_ids,
                                                                                          :]
        # normal by 3d neighbor plane fitting.
        normal_path = '{}/frame-{:06d}.scaled.normal.npy'.format(self.data_dir, fid)
        if os.path.exists(normal_path):
            # normal = np.load(normal_path)
            if os.path.getsize(normal_path) > 1:
                normal = np.load(normal_path, encoding='bytes', allow_pickle=True)
            else:
                normal = compute_points_normal(local)
                np.save(normal_path, normal)
        else:
            normal = compute_points_normal(local)
            np.save(normal_path, normal)
        lclnmlclr = np.concatenate((np.concatenate((local, normal), axis=2), color), axis=2)
        # build a patch
        pt_in = torch.zeros((self.n_pts_per_frame, 7, 1))
        nb_ms_in = torch.zeros((self.n_pts_per_frame, opt.tree_height - 1, 7, opt.n_neighb_pts))
        route_labs = torch.zeros((self.n_pts_per_frame, opt.tree_height - 1))
        r_max, c_max = int(image_height / 4 - 1), int(image_width / 4 - 1)
        rc_list = []
        # da2d+3d neighbor
        if not self.neighbor_da2d is None:
            sid, count_crt, count_max = 0, 0, 9999
            mask = np.zeros((r_max, c_max))
            while len(rc_list) < self.n_pts_per_frame:
                # avoid infinite loop
                count_crt += 1
                if count_crt > count_max:
                    break
                r, c = np.random.randint(0, r_max), np.random.randint(0, c_max)
                if depth[r, c] == 0. or mask[r, c] == 1.:
                    continue
                if np.isnan(lclnmlclr[r, c, 3]):
                    continue
                mask[r, c] = 1.
                rc_list.append([r, c])
                pt_in[sid] = torch.Tensor(
                    np.concatenate((np.array([[0.], [0.], [0.], [0.]]), color[r, c, 0:3][:, np.newaxis]), axis=0))
                for lid in range(opt.tree_height - 1):
                    da2d_list = (np.array(self.neighbor_da2d[lid]) / depth[r, c]).astype(int)
                    # ppf
                    pts9d = shift_pts9d(
                        sample_pts(partial_pts_2d(lclnmlclr, (r, c), da2d_list), opt.n_neighb_pts),
                        lclnmlclr[r, c, :])
                    cen9d = copy.deepcopy(lclnmlclr[r, c, :])
                    cen9d[0:3] = np.zeros(3)
                    ppf7d = make_ppf(pts9d, cen9d)  # (N,9), (9,)
                    ppf7d[np.isnan(ppf7d)] = 0.0
                    nb_ms_in[sid, lid, :, :] = torch.Tensor(ppf7d).transpose(1, 0)
                    # remove background by 3d radius
                    xyz = pts9d[:, 0:3]
                    ids_out_of_bound = np.linalg.norm(xyz, axis=1) > opt.far_radiuses[lid]
                    nb_ms_in[sid, lid, :, ids_out_of_bound] = 0.
                # count
                sid += 1
        return pt_in, nb_ms_in, -1, fid, torch.Tensor(np.array(rc_list))


# # debug
# if __name__ == '__main__':
# 	print('done.')

