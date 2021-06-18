import argparse
import os


# args. 
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--scene_id', type=int, default=1)
parser.add_argument('--sequence_id', type=int, default=2)
parser.add_argument('--frame_step', type=int, default=10)
parser.add_argument('--frame_begin', type=int, default=0)
parser.add_argument('--frame_end', type=int, default=999999)
parser.add_argument('--frame_mask', default='frame-{:06d}.rendered.depth.png')
parser.add_argument('--data_folder_mask', default='/opt/dataset/scene{:02d}/seq{:02d}/seq{:02d}_{:02d}')
parser.add_argument('--prediction_folder', default='/opt/relocalizer_codes/NeuralRouting/gmm_prediction/_rio10_scene01_5lvs_top[2,2]_seq01_02_step10_2048pts')
opt = parser.parse_args()
gmm_prediction_folder = opt.prediction_folder


# fixed args.
gmm_config_path = '/opt/relocalizer_codes/spaint/dsy.prediction.config.txt'
rgbd_config_path = '/opt/relocalizer_codes/spaint/dsy.rgbd.config.txt'
step_config_path = '/opt/relocalizer_codes/spaint/dsy.step.config.txt'


# main script.
if __name__ == '__main__':

    # write gmm config file.
    gmm_file = open(gmm_config_path, 'w')
    gmm_file.write(gmm_prediction_folder)
    gmm_file.close()

    # write rgbd config file.
    rgbd_file = open(rgbd_config_path, 'w')
    for frame_id in range(opt.frame_begin, opt.frame_end+1, opt.frame_step):
        rgbd_file.write('{}/{}\n'.format(opt.data_folder_mask, opt.frame_mask).format(
            opt.scene_id, opt.scene_id, opt.scene_id, opt.sequence_id, frame_id))
    rgbd_file.close()

    # write step config file.
    step_file = open(step_config_path, 'w')
    step_file.write('{:d}'.format(opt.frame_step))
    step_file.close()

    # run runsac.
    cmd = ('CUDA_VISIBLE_DEVICES={:d} ' 
        + '/opt/relocalizer_codes/spaint/build/bin/apps/relocgui/relocgui ' 
        + '-c /opt/relocalizer_codes/spaint/calib_rio10_scene{:02d}.txt '
        + '--test {}').format(opt.device, opt.scene_id, rgbd_config_path)
    os.system(cmd) 

