import numpy as np
import os


# set the path here
dataset_folder = '/opt/dataset' 


# file format mask (fixed)
mask = '{}/scene{:02d}/seq{:02d}/seq{:02d}_{:02d}/frame-{:06d}.pose.txt'
target_mask = '{}/scene{:02d}/seq{:02d}/seq{:02d}_{:02d}/frame-{:06d}.pose.rnd.txt'

# random transformation (fixed)
trans = []
trans.append(np.array([[-0.207853, 0.97816, 0, 2.18209], [-0.97816, -0.207853, 0, -1.09188], [0, 0, 1, 0.19015], [0, 0, 0, 1]]))        
trans.append(np.array([[-0.036429, 0.999337, 0, 0.175627], [-0.999337, -0.036429, 0, 0.730642], [-0, 0, 1, -0.16], [0, 0, 0, 1]]))      
trans.append(np.array([[-0.957291, 0.289124, 0, -0.0948915], [-0.289124, -0.957291, 0, -0.0569603], [0, 0, 1, 0.28269], [0, 0, 0, 1]])) 
trans.append(np.array([[-0.957292,  0.289125, 0, 0.81302], [-0.289125, -0.957292, 0, -0.875411], [0, 0, 1, -0.30922], [0, 0, 0, 1]]))   
trans.append(np.array([[-0.957292, 0.289125, 0, 0.218485], [-0.289125, -0.957292, 0, 1.24795], [0, 0, 1, 0.0420799], [0, 0, 0, 1]]))    
trans.append(np.array([[-0.673072, -0.739576, 0, 1.28461], [0.739576, -0.673072, 0, 2.10945], [0, 0, 1, -0.0253899], [0, 0, 0, 1]]))    
trans.append(np.array([[0.676818, -0.73615, 0, -0.887091], [0.73615, 0.676818, 0, 1.07194], [0, 0, 1, -0.10717], [0, 0, 0, 1]]))        
trans.append(np.array([[0.676819, -0.73615, 0, 2.02555], [0.73615, 0.676819, 0, -0.970286], [0, 0, 1, 0.26751], [0, 0, 0, 1]]))         
trans.append(np.array([[0.202885, 0.979203, 0, 0.863833], [-0.979203, 0.202885, 0, 2.08022], [0, 0, 1, -0.15873], [0, 0, 0, 1]]))       
trans.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))                                                        


if __name__ == '__main__':
    print('preprocessing data...')
    for scene_id in range(1, 11):
        for seq_id in range(1, 3):
            for fid in range(999999):
                path = mask.format(dataset_folder, scene_id, scene_id, scene_id, seq_id, fid)
                target_path = target_mask.format(dataset_folder, scene_id, scene_id, scene_id, seq_id, fid)
                if not os.path.exists(path):
                    break
                Rt = np.loadtxt(path)
                result = np.dot(np.linalg.inv(trans[scene_id-1]), Rt)
                np.savetxt(target_path, result)
        print('scene{:02d} done.'.format(scene_id))

