from torch.utils.tensorboard import SummaryWriter
from glob import glob
import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.distributed as dist

from utils.experiment_utils import print_to_file_and_screen
from model.resunet import FCGF_net
from utils.tools_3d import apply_transformation
from dataloader.generic_balanced_loader import VOXEL_SIZE


NUM_POS = 1024
NEG_SEARCH_SET_SIZE_PER_SAMPLE = 256
NEG_THRESH = 1.4
POS_THRESH = 0.1
POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER = 4
PAIR_SEARCH_VOXEL_SIZE = POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER * VOXEL_SIZE

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rand_choice(T, N):
    idx = torch.randperm(T.shape[0])[:N]
    T_sub = T[idx,:]                    
    return T_sub, idx
    
class FCGF_trainer():

    def __init__(self,
                 outfile,                 
                 dataset_name,
                 lr=0.1,
                 l2_reg=0.0001,
                 resume=False,
                 shuffle=True,
                 device='cpu',
                 rank=-1):

        self.dataset_name = dataset_name
        self.rank = rank
        self.lr = lr
        self.shuffle = shuffle
        try:
            self.outfile_name = outfile.name
            self.outfile = outfile
        except AttributeError as E:
            if "object has no attribute" in str(E):
                self.outfile_name = outfile
                self.outfile = None
            else:
                raise E

        self.outdir = os.path.split(self.outfile_name)[0] + '/'

        self.dtype_ = torch.float
        self.device = device
        print("device=" + str(self.device))

        self.FeatureNet = FCGF_net(self.device)        
        print_to_file_and_screen(self.outfile, "number of parameters: Overall: %d" % count_parameters(self.FeatureNet))

        self.optimizer = torch.optim.SGD(
            params=self.FeatureNet.parameters(),
            lr=lr,
            weight_decay=l2_reg,
            momentum=0.8
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.995)

        self.handle_resume(resume)
        self.FeatureNet.train()

        if self.outfile is not None:
            tensorboard_outfile = self.outdir + 'tensorboard.summary'
            print_to_file_and_screen(self.outfile, "writing tensorboard data to " + tensorboard_outfile)
            print_to_file_and_screen(self.outfile, "run with: \ntensorboard --logdir=%s" % tensorboard_outfile.replace('/root/','~/'))
            self.writer = {}
            self.writer['train'] = SummaryWriter(tensorboard_outfile + '/train')
            self.writer['validation'] = SummaryWriter(tensorboard_outfile + '/validation')

    def get_FCGF_network(self):
        return self.FeatureNet

    def advance_scheduler(self):
        self.scheduler.step()

    def save_model(self, epoch=-1, keep_old_checkpoints=True):
        
        checkpoint_file = f"{self.outdir}{self.dataset_name}.{epoch}.t7"

        d = {
            'model': self.FeatureNet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        torch.save(d, checkpoint_file)        
        
        checkpoint_file_compatible = checkpoint_file.replace('.t7','.pth')
        e = {'state_dict': self.FeatureNet.Model.state_dict()}
        torch.save(e, checkpoint_file_compatible)        

        print_to_file_and_screen(self.outfile, "Trainer: wrote model to " + checkpoint_file)

        if not keep_old_checkpoints:
            for suffix in ['t7','pth']:
                older_checkpoint_files = glob(checkpoint_file_base + '*.' + suffix)
                if checkpoint_file in older_checkpoint_files:
                    older_checkpoint_files.remove(checkpoint_file)
                if checkpoint_file_compatible in older_checkpoint_files:
                    older_checkpoint_files.remove(checkpoint_file_compatible)
                for fn in older_checkpoint_files:
                    if not 'best' in fn:
                        if '.t7' in fn:
                            print("Trainer: removing " + fn)
                        os.remove(fn)
        return checkpoint_file, checkpoint_file_compatible

    def handle_resume(self, resume):
        if resume:
            checkpoint_file = resume
            print_to_file_and_screen(self.outfile, "Loading model from checkpoint " + checkpoint_file)

            d = torch.load(checkpoint_file, map_location=self.device)
            self.FeatureNet.load_state_dict(d['model'], strict=True)
            self.optimizer.load_state_dict(d['optimizer'])
            self.scheduler.load_state_dict(d['scheduler'])

    def validation_batch(self, GT_motion, arrs, deterministic=None):    
        res = self.train_batch(GT_motion, arrs, phase='validation')    
        return res

    def sum_gradients(self):
        model = self.FeatureNet
        for name, param in model.named_parameters():
            if param.grad is None:
                print("%s.grad is None, skipping" % name)
            else:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    def select_positive_pairs_per_sample(self, sample_arrs):

        P_rot = sample_arrs[0]['PC_rot']
        Q = sample_arrs[1]['PC']

        P_sub, P_idx = rand_choice(P_rot, NUM_POS)

        dist = torch.cdist(P_sub, Q)

        thresh = PAIR_SEARCH_VOXEL_SIZE
        is_close_enough = (dist < thresh)
        pairs = torch.nonzero(is_close_enough)
        pairs[:,0] = P_idx[pairs[:,0]]
        
        return pairs

    def select_positive_pairs(self, GT_motion, arrs, batch_size):
        pair_list = []
        to_ = [0, 0]
        from_ = [0,0]
        P_rot_list = []
        for sample_ind in range(batch_size):
            # extract data for sample:
            sample_arrs = [{k: None for k in arrs[j].keys()} for j in [0,1]]
            for i in [0,1]:                
                from_[i] = to_[i]
                to_[i] = from_[i] + arrs[i]['len'][sample_ind]
                for k in sample_arrs[i].keys():
                    if arrs[i][k].ndim == 2: # Minkowski Engine style batching
                        sample_arrs[i][k] = arrs[i][k][from_[i]:to_[i], :]
                    else:
                        sample_arrs[i][k] = arrs[i][k][sample_ind]

            P = sample_arrs[0]['PC']            
            P_rot = apply_transformation(GT_motion[sample_ind,...], P)
            cur_P_rot = P_rot.float()
            sample_arrs[0]['PC_rot'] = cur_P_rot
            P_rot_list.append(cur_P_rot)

            cur_pairs = self.select_positive_pairs_per_sample(sample_arrs)
            cur_pairs[:,0] += from_[0]
            cur_pairs[:,1] += from_[1]
            pair_list.append(cur_pairs)
        all_raw_pairs = torch.cat(pair_list, dim=0)
        all_pairs, _ = rand_choice(all_raw_pairs, batch_size*NUM_POS)
        P_rot = torch.cat(P_rot_list, dim=0)
        arrs[0]['PC_rot'] = P_rot
        return all_pairs, arrs

    def contrastive_hardest_negative_loss(self, arrs, pos_pairs, batch_size):
        NEG_SEARCH_SET_SIZE = NEG_SEARCH_SET_SIZE_PER_SAMPLE * batch_size

        """
        Generate negative pairs
        """
        P_xyz = arrs[0]['PC_rot'] # these are xyz values after GT rotation
        Q_xyz = arrs[1]['PC'] 
        P_feat = arrs[0]['Feature']
        Q_feat = arrs[1]['Feature']

        Pneg_xyz, Pneg_idx = rand_choice(P_xyz, NEG_SEARCH_SET_SIZE)
        Pneg_feat = P_feat[Pneg_idx,:]
        Qneg_xyz, Qneg_idx = rand_choice(Q_xyz, NEG_SEARCH_SET_SIZE)
        Qneg_feat = Q_feat[Qneg_idx,:]
    
        Ppos_feat = P_feat[pos_pairs[:,0],:]
        Ppos_xyz = P_xyz[pos_pairs[:,0],:]
        Qpos_feat = Q_feat[pos_pairs[:,1],:]
        Qpos_xyz = Q_xyz[pos_pairs[:,1],:]

        def calc_d_array(X,Y):
            NUM_PARTS = 2
            x_sep = np.round(np.linspace(0,X.shape[0], NUM_PARTS+1)).astype(int)            
            d_list = []
            for p in range(NUM_PARTS):
                cur_X = X[x_sep[p]:x_sep[p+1],:]                
                cur_d2 = torch.sum((cur_X.unsqueeze(1) - Y.unsqueeze(0))**2, dim=2)                    
                cur_d = cur_d2.clamp_min(1e-30).sqrt_()
                d_list.append(cur_d)
            d = torch.cat(d_list)
            return d

        def calc_d2(x,y): 
            return torch.sum((x-y)**2,dim=1)
        
        def calc_d(x,y): 
            d2 = calc_d2(x,y)
            d = d2.clamp_min(1e-30).sqrt_()
            return d

        def calc_neg_loss(source_xyz, source_feat, cand_xyz, cand_feat):
            d_arr_feat = calc_d_array(source_feat, cand_feat)
            _, target_inds = d_arr_feat.min(dim=1)
            target_xyz = cand_xyz[target_inds,:]
            d2_pair_xyz = calc_d2(source_xyz, target_xyz)
            is_valid = d2_pair_xyz >= PAIR_SEARCH_VOXEL_SIZE**2
            d_pairs = calc_d(source_feat, cand_feat[target_inds,:])                
            
            # notice that this is actually too strict: P and Q contain points from _multiple_ clouds (the number of clouds is batch_size).
            # if a pair of negative points is not in the same cloud, there's no harm if their xyz distance is smaller than the threhold.
            losses = F.relu(NEG_THRESH - d_pairs[is_valid]).pow(2)
            return losses.mean()

        loss_P_Qn = calc_neg_loss(Ppos_xyz, Ppos_feat, Qneg_xyz, Qneg_feat)
        loss_Pn_Q = calc_neg_loss(Qpos_xyz, Qpos_feat, Pneg_xyz, Pneg_feat)
        neg_loss = 0.5*( loss_P_Qn + loss_Pn_Q )
        
        pos_losses = F.relu((Ppos_feat - Qpos_feat).pow(2).sum(1) - POS_THRESH)
        pos_loss = pos_losses.mean()

        return pos_loss, neg_loss


    def train_batch(self, GT_motion, arrs, phase='train'): #XXXXX
        if phase == 'train':
            self.FeatureNet.train()
        elif phase == 'validation':
            self.FeatureNet.eval()
        else:
            assert False, "unknown phase: " + phase
    
        batch_size = GT_motion.shape[0]

        report_names = ['actual_batch_size']
        report_vals_list = [torch.tensor(batch_size,device=self.device)]
        report_names.append('lr')
        report_vals_list.append(torch.tensor(self.scheduler.get_last_lr()[0], device=self.device))

        for i in range(2):
            arrs[i]['Feature'] = self.FeatureNet(arrs[i]['coords'])

        pos_pairs, arrs = self.select_positive_pairs(GT_motion, arrs, batch_size)
        total_pos_loss, total_neg_loss = self.contrastive_hardest_negative_loss(arrs, pos_pairs, batch_size)
        loss = total_pos_loss + total_neg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.sum_gradients()
        self.optimizer.step()            

        report_names.append('loss')
        report_vals_list.append(loss)
        report_names.append('pos_loss')
        report_vals_list.append(total_pos_loss)        
        report_names.append('neg_loss')
        report_vals_list.append(total_neg_loss)                

        report_cuda = torch.stack(report_vals_list)
        report_cuda[1:] *= batch_size

        return report_cuda, report_names

    def record_losses(self, losses_np, iter_tuple, mode='train'):        
        epoch = iter_tuple[0]
        iter_in_epoch = iter_tuple[1]
        iters_per_epoch = iter_tuple[2]
        iter = epoch*iters_per_epoch + iter_in_epoch
        for k in losses_np.keys():
            self.writer[mode].add_scalar(k, losses_np[k], iter)

