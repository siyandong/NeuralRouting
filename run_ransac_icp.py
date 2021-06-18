import argparse
import os


# args. 
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--scene_id', type=int, default=1)
parser.add_argument('--sequence_id', type=int, default=2)
parser.add_argument('--data_folder_mask', default='/opt/dataset/scene{:02d}/seq{:02d}')
parser.add_argument('--prediction_folder', default='/opt/relocalizer_codes/NeuralRouting/gmm_prediction/_rio10_scene01_5lvs_top[2,2]_seq01_02_step1_2048pts')
opt = parser.parse_args()
gmm_prediction_folder = opt.prediction_folder


# fixed args.
gmm_config_path = '/opt/relocalizer_codes/spaint/dsy.prediction.config.txt'


# main script.
if __name__ == '__main__':

    # write gmm config file.
    gmm_file = open(gmm_config_path, 'w')
    gmm_file.write(gmm_prediction_folder)
    gmm_file.close()

    # set masks.
    train_depth_mask = '{}/seq{:02d}_01/frame-%06d.rendered.depth.png'.format(opt.data_folder_mask, opt.scene_id).format(opt.scene_id, opt.scene_id)
    train_color_mask = '{}/seq{:02d}_01/frame-%06d.color.jpg'.format(opt.data_folder_mask, opt.scene_id).format(opt.scene_id, opt.scene_id)
    train_pose_mask = '{}/seq{:02d}_01/frame-%06d.pose.rnd.txt'.format(opt.data_folder_mask, opt.scene_id).format(opt.scene_id, opt.scene_id)
    test_depth_mask = '{}/seq{:02d}_{:02d}/frame-%06d.rendered.depth.png'.format(opt.data_folder_mask, opt.scene_id, opt.sequence_id).format(opt.scene_id, opt.scene_id)
    test_color_mask = '{}/seq{:02d}_{:02d}/frame-%06d.color.jpg'.format(opt.data_folder_mask, opt.scene_id, opt.sequence_id).format(opt.scene_id, opt.scene_id)

    # run runsac + icp.
    cmd = 'CUDA_VISIBLE_DEVICES={:d} '.format(opt.device) +\
    '/opt/relocalizer_codes/spaint/build/bin/apps/spaintgui/spaintgui ' +\
    '--headless --pipelineType slam ' +\
    '-c /opt/relocalizer_codes/spaint/calib_rio10_scene{:02d}.txt '.format(opt.scene_id) +\
    '-f /opt/relocalizer_codes/spaint/Default_Rank16.ini ' +\
    '-d {} '.format(train_depth_mask) +\
    '-r {} '.format(train_color_mask) +\
    '-p {} '.format(train_pose_mask) +\
    '-t Disk ' +\
    '-d {} '.format(test_depth_mask) +\
    '-r {} '.format(test_color_mask) +\
    '-t ForceFail'
    #print(cmd)
    os.system(cmd) 

