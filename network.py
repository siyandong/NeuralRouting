import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# encode visual input.
class SharedFeatureNet(nn.Module):

    def __init__(self, n_channel_out=128, n_group=1):
        super(SharedFeatureNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(7, 64, 1), # for ppf(4d)+rgb(3d) as input.
            nn.GroupNorm(n_group, 64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(n_group, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, n_channel_out, 1)
        )

    def forward(self, x): # x.shape = (BS, #feature, #point).
        x = self.mlp(x)
        x = torch.max(x, 2)[0]
        return x


# shared classifier. 
class SharedClassifier(nn.Module): 

    def __init__(self, n_param_in, n_channel_in, n_channel_out, n_group=1):
        super(SharedClassifier, self).__init__()
        self.n_level = n_param_in
        # HyperNet.
        self.weight_learner_norm_weight1_1 = nn.Linear(n_param_in, 32)
        self.weight_learner_norm_weight1_2 = nn.Linear(32, 2048)
        self.weight_learner_norm_bias1_1 = nn.Linear(n_param_in, 32)
        self.weight_learner_norm_bias1_2 = nn.Linear(32, 2048)
        self.weight_learner_norm_weight2_1 = nn.Linear(n_param_in, 32)
        self.weight_learner_norm_weight2_2 = nn.Linear(32, 1024)
        self.weight_learner_norm_bias2_1 = nn.Linear(n_param_in, 32)
        self.weight_learner_norm_bias2_2 = nn.Linear(32, 1024)
        # BaseNet.
        self.base_net_linear1 = nn.Linear(n_channel_in, 2048)
        self.base_net_linear2 = nn.Linear(2048, 1024)
        self.base_net_linear3 = nn.Linear(1024, n_channel_out)
        self.relu = nn.ReLU(inplace=True)
        self.n_group = n_group
        # other param.
        self.n_channel_out = n_channel_out

    def forward(self, x_feature, x_param, b_level1=False): # x_feature (batch_size, n_channel_in), x_param (batch_size, n_level).
        # trunc x_param.
        x_param = x_param[:, :self.n_level]
        # GN weight and bias.
        base_net_norm_weight1 = self.weight_learner_norm_weight1_2(self.relu(self.weight_learner_norm_weight1_1(x_param)))
        base_net_norm_bias1 = self.weight_learner_norm_bias1_2(self.relu(self.weight_learner_norm_bias1_1(x_param)))
        base_net_norm_weight2 = self.weight_learner_norm_weight2_2(self.relu(self.weight_learner_norm_weight2_1(x_param)))
        base_net_norm_bias2 = self.weight_learner_norm_bias2_2(self.relu(self.weight_learner_norm_bias2_1(x_param)))
        # forward BaseNet.
        if b_level1:
            x = self.base_net_linear1(x_feature)
            x = F.group_norm(x, self.n_group, weight=base_net_norm_weight1[0], bias=base_net_norm_bias1[0])
            x = self.relu(x)
            x = self.base_net_linear2(x)
            x = F.group_norm(x, self.n_group, weight=base_net_norm_weight2[0], bias=base_net_norm_bias2[0])
            x = self.relu(x)
            x = self.base_net_linear3(x)
            x = F.log_softmax(x, dim=1)
        else:
            batch_size = x_feature.shape[0]
            # classify by node. much faster in deep levels.
            x_param_np = x_param.cpu().numpy()
            info2idx_map = dict()
            node_infos = set()
            for sid in range(batch_size):
                info = tuple(x_param_np[sid].tolist())
                if not info in node_infos:
                    info2idx_map[info] = len(node_infos)
                    node_infos.add(info)
            new_batches = []
            for idx in range(len(node_infos)):
                new_batches.append([])
            for sid in range(batch_size):
                info = tuple(x_param_np[sid].tolist())
                new_batches[info2idx_map[info]].append(sid)
            # forward each new batch. 
            x_0 = x_feature
            x_1 = self.base_net_linear1(x_0)
            for sid_list in new_batches: x_1[sid_list,:] = F.group_norm(x_1[sid_list,:], self.n_group, weight=base_net_norm_weight1[sid_list[0]], bias=base_net_norm_bias1[sid_list[0]])
            x_1 = self.relu(x_1)
            x_2 = self.base_net_linear2(x_1)
            for sid_list in new_batches: x_2[sid_list,:] = F.group_norm(x_2[sid_list,:], self.n_group, weight=base_net_norm_weight2[sid_list[0]], bias=base_net_norm_bias2[sid_list[0]])
            x_2 = self.relu(x_2)
            x = self.base_net_linear3(x_2)
            x = F.log_softmax(x, dim=1)
        return x


# if __name__ == '__main__':
#     print('done.')

