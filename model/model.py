import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU
from torch.nn.parameter import Parameter
from scipy.stats import zscore
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class multi_view_decoder(nn.Module):
    def __init__(self, num_view, input_dim_set, feature_dim):
        super(multi_view_decoder, self).__init__()
        self.mv_decoder=torch.nn.ModuleList()
        for i in range(num_view):
            decoder = nn.Sequential(
                nn.Linear(feature_dim, 2000),
                nn.ReLU(),
                nn.Linear(2000, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, input_dim_set[i])
            )
            self.mv_decoder.append(decoder)

    def forward(self, x_set):
        x_new = []
        num=0
        for x in x_set:
            x_new.append(self.mv_decoder[num](x))
            num+=1
        return x_new


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


# top 1 hard routing
def topk(t, k):
    values, index = t.topk(k=k, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]


def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]


class Router(nn.Module):
    def __init__(self, input_dim, out_dim, device, strategy='topk', noise=None):
        super(Router, self).__init__()

        self.cls = nn.Linear(input_dim, out_dim, bias=False)
        self.out_dim = out_dim
        self.strategy = strategy
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        self.cls.weight = torch.nn.Parameter(self.cls.weight * 0)

    def forward(self, x):

        x = self.cls(x)
        # x = torch.sum(x, dim=1)
        if self.strategy == 'topk':
            self.noise = torch.rand(x.shape[0], x.shape[1]).to(self.device)

        output = x + self.noise
        return output

class first_pre(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(first_pre, self).__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
        )

    def forward(self, x):
        return self.pre(x)

class multi_view_pre(nn.Module):
    def __init__(self, num_view, input_dim_set, feature_dim):
        super(multi_view_pre, self).__init__()
        self.mv_pre=torch.nn.ModuleList()
        for i in range(num_view):
            self.mv_pre.append(nn.Sequential(first_pre(input_dim_set[i], feature_dim)))

    def forward(self, x_set, num):
        x_new = []
        num=0
        for x in x_set:
            x_new.append(self.mv_pre[num](x))
            num+=1
        return x_new

class MoE(nn.Module):
    
    def __init__(self, n_label, out_dim, cluster_dim, expert_num, k, device, strategy='topk', nonlinear=True, noise=None):
        super(MoE, self).__init__()
        self.router = Router(out_dim, expert_num, device, strategy=strategy, noise=noise)
        self.models = nn.ModuleList()
        self.k = k
        for i in range(expert_num):
            
            self.models.append(Encoder(out_dim, cluster_dim))
        self.strategy = strategy
        self.expert_num = expert_num
        self.mmd_loss = MMD_loss()
        self.device = device

    def forward(self, x):
        select = self.router(x)

        # top k or choose k according to probability
        if self.strategy == 'topk':
            gate, index = topk(select, self.k)
        else:
            gate, index = choosek(select)


        mask = F.one_hot(index, self.expert_num).float()


        density = mask.mean(dim=-2)
        density_proxy = select.mean(dim=-2)
        balance_loss = (density_proxy * density).mean() * float(self.expert_num ** 2)

        mask_count = mask.sum(dim=-2, keepdim=True)
        mask_flat = mask.sum(dim=-1)

        if self.k==1:
            combine_tensor = (gate[..., None] * mask_flat[..., None] * mask).unsqueeze(dim=1).permute(0, 2,
                                                                                 1)  # [batch_size, expert_num, capacity]
        else:
            combine_tensor = (gate[..., None] * mask_flat[..., None] * mask).permute(0, 2, 1)
                  

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        select0 = dispatch_tensor.squeeze(-1)
        expert_inputs = torch.einsum('sm,sec->ecm', x, dispatch_tensor)  
        output = []
        middle_set = []
        for i in range(self.expert_num):
            expert_output = self.models[i](expert_inputs[i])

            output.append(expert_output)


        output = torch.stack(output)  
        middle_set = torch.einsum('sec,ecm->scm', combine_tensor, output).permute(1, 0, 2)
        output = torch.einsum('sec,ecm->sm', combine_tensor, output)
        dist_loss = 0
        idx_set = [i for i in range(middle_set.shape[1])]
        sample_num = int(np.percentile(idx_set,90))
        for i in range(middle_set.shape[0]):
            for j in range(i+1, middle_set.shape[0]):
                sample_1_idx = random.sample(list(idx_set), sample_num)
                sample_2_idx = random.sample(list(idx_set), sample_num)
                dist_loss += (-1*self.mmd_loss(middle_set[i,sample_1_idx,:].squeeze(dim=0), middle_set[j,sample_2_idx,:].squeeze(dim=0)))


        return output, select0, balance_loss, dist_loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class MV_MOE(nn.Module):
    def __init__(self, num_view, input_dim_set, feature_dim, n_label, cluster_dim, expert_num, k, device, args, alpha=1.0, gamma=.1):
        super(MV_MOE, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.d = args.d
        self.view_num=num_view
        self.cluster_emb =cluster_emb = self.d * n_label
        self.s = None
        self.mmd_loss = MMD_loss()
        self.eta = args.eta
        self.n_clusters=n_label
        self.aggregation= aggregation = 'concat'
        # self.aggregation= aggregation = 'sum'
        # self.mv_pre = multi_view_pre(num_view, input_dim_set, feature_dim)

        self.mv_pre=torch.nn.ModuleList()
        for i in range(num_view):
            self.mv_pre.append(nn.Sequential(first_pre(input_dim_set[i], feature_dim)))


        self.moe = MoE(n_label, feature_dim, cluster_dim, expert_num, k, device)
        self.mv_decoder=torch.nn.ModuleList()
        for i in range(num_view):
            decoder = nn.Sequential(
                nn.Linear(cluster_dim*num_view, 2000),

                nn.LeakyReLU(),
                nn.Linear(2000, 500),
                nn.LeakyReLU(),
                nn.Linear(500, 500),
                nn.LeakyReLU(),
                nn.Linear(500, input_dim_set[i])
            )
            self.mv_decoder.append(decoder)


    def forward(self, x_set):
        s = None
        middle_set=[]
        num=0
        for x in x_set:

            middle_set.append(self.mv_pre[num](x))
            num+=1
        sum_balance_loss = 0
        sum_dist_loss = 0
        view_h=[]
        for num in range(self.view_num):
            output, select0, balance_loss, dist_loss = self.moe(middle_set[num])
            view_h.append(output)
            sum_balance_loss+=balance_loss
            sum_dist_loss += dist_loss
        if self.aggregation == 'concat':
            fused_h = torch.concat(view_h, dim=1)
        elif self.aggregation == 'sum':
            fused_h = torch.zeros(x_set[0].shape[0],self.cluster_emb).to(self.device)
            for tmp_h in view_h: 
                fused_h+=tmp_h.to(self.device)
        elif self.aggregation == 'avg':
            fused_h = torch.zeros(x_set[0].shape[0],self.cluster_emb).to(self.device)
            for tmp_h in view_h: 
                fused_h+=tmp_h.to(self.device)
            fused_h = fused_h/self.view_num
            
        fused_h = fused_h.to(self.device)

        # expert_distinctive_loss = 0
        # for i in range(len(view_h)):
        #     for j in range(i, len(view_h)):
        #         expert_distinctive_loss+=(-1.0*self.mmd_loss(view_h[i],view_h[j]))

        reconstructed_x=[]
        for num in range(self.view_num):
            reconstructed_x.append(self.mv_decoder[num](fused_h))

        return fused_h, reconstructed_x, sum_balance_loss, sum_dist_loss

    def get_results(self, data_loader, num_views):
        if num_views == 2:
            fused_h = []
            all_s = []
            all_y = []
            for idx, (feature_1, feature_2, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2]
                tmp_h, _, _, _= self.forward(feature_list)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        elif num_views == 3:
            fused_h = []

            all_y = []
            for idx, (feature_1, feature_2, feature_3, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2, feature_3]
                tmp_h, _, _, _= self.forward(feature_list)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        elif num_views == 4:
            fused_h = []

            all_y = []
            for idx, (feature_1, feature_2, feature_3, feature_4, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2, feature_3, feature_4]
                tmp_h, _, _, _= self.forward(feature_list)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        elif num_views == 5:
            fused_h = []

            all_y = []
            for idx, (feature_1, feature_2, feature_3, feature_4, feature_5, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5]
                tmp_h, _, _, _= self.forward(feature_list)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        elif num_views == 6:
            fused_h = []

            all_y = []
            for idx, (feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, y) in enumerate(data_loader):
                feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]
                tmp_h, _, _, _= self.forward(feature_list)
                fused_h.append(tmp_h)

                all_y.append(y)
            fused_h=torch.concat(fused_h, dim=0)

            all_y = torch.concat(all_y, dim=0)
        return fused_h, all_y


    def z_score_standardization(self, matrix):
        matrix = matrix.cpu().detach().numpy()
        complete_matrix = []
        for i in range(matrix.shape[0]):
            standardized_data = zscore(matrix[i,:])
            complete_matrix.append(standardized_data)
        complete_matrix = torch.tensor(np.stack(complete_matrix))
        return complete_matrix

    def cross_view_loss(self, h_set):
        total_loss = 0
        for i in range(len(h_set)):
            for j in range(i, len(h_set)):
                total_loss+=(-1*self.mmd_loss(h_set[i],h_set[j]))
        return total_loss
            

