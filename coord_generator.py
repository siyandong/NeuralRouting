import numpy as np
import cv2


class CoordGenerator(object):

    def __init__(self, intrin, img_w, img_h):
        super(CoordGenerator, self).__init__()
        self.intrinsics = intrin
        self.image_width = img_w
        self.image_height = img_h

    def pixel2local(self, depth): # depth: float32, meter.  
        cx, cy, fx, fy = self.intrinsics[0, 2], self.intrinsics[1, 2], self.intrinsics[0, 0], self.intrinsics[1, 1]
        u_base = np.tile(np.arange(self.image_width), (self.image_height, 1))
        v_base = np.tile(np.arange(self.image_height)[:, np.newaxis], (1, self.image_width))
        X = (u_base - cx) * depth / fx
        Y = (v_base - cy) * depth / fy
        coord_camera = np.stack((X, Y, depth), axis=2)
        points_local = coord_camera.reshape((-1, 3), order='F') # (N, 3).
        return points_local

    def local2world(self, points_local, pose):
        points_local_homo = np.concatenate((points_local, np.ones((points_local.shape[0], 1), dtype=np.float32)), axis=1) # N*4.
        points_world_homo = np.matmul(pose, points_local_homo.T).T # (4*4 * 4*N).T = N*4.
        points_world = np.divide(points_world_homo, points_world_homo[:, [-1]])[:, :-1]
        return points_world

    def depth_pose_2coord(self, depth, pose):
        points_local = self.pixel2local(depth)
        points_world = self.local2world(points_local, pose)
        points_world[(points_local == [0, 0, 0]).all(axis=1)] = 0 # useless?
        img_coord = points_world.reshape((self.image_height, self.image_width, 3), order='F')
        vis_coord = (img_coord - img_coord.min()) / (img_coord.max() - img_coord.min()) * 255
        return img_coord, vis_coord


# if __name__ == '__main__':
#     print('done.')

