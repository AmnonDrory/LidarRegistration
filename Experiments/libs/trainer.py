import torch
import time, os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.timer import Timer, AverageMeter
from tqdm import tqdm
# import matplotlib.pyplot as plt
import torch.distributed as dist

class Trainer(object):
    def __init__(self, args, rank):
        # parameters
        self.rank = rank
        self.max_epoch = args.max_epoch
        self.training_max_iter = args.training_max_iter
        self.val_max_iter = args.val_max_iter
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.evaluate_interval = args.evaluate_interval
        self.evaluate_metric = args.evaluate_metric
        self.metric_weight = args.metric_weight
        self.transformation_loss_start_epoch = args.transformation_loss_start_epoch
        self.writer = SummaryWriter(log_dir=args.tboard_dir)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader
        self.train_feature_extractor = args.train_feature_extractor
        self.val_feature_extractor = args.val_feature_extractor        

        if self.gpu_mode:
            self.device = 'cuda:%d' % torch.cuda.current_device()
            self.model = self.model.cuda()
        else:
            self.device = 'cpu'

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

    def sum_gradients(self):
        model = self.model
        for name, param in model.named_parameters():
            if param.grad is None:
                if not name in ['sigma_spat']:
                    print("%s.grad is None, skipping" % name)
            else:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    def train(self): #XXXXX
        best_reg_recall = 0
        if self.rank == 0:
            print(f'{time.strftime("%m/%d %H:%M:%S")} training start!!')
        start_time = time.time()

        self.model.train()
        res = self.evaluate(0)
        if self.rank==0:
            print(f'{time.strftime("%m/%d %H:%M:%S")} Evaluation: Epoch 0: SM Loss {res["sm_loss"]:.2f} Class Loss {res["class_loss"]:.2f} Trans Loss {res["trans_loss"]:.2f} Recall {res["reg_recall"]:.2f}')
        for epoch in range(self.max_epoch):
            self.train_epoch(epoch + 1)  # start from epoch 1

            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                if self.rank == 0:
                    print(f'{time.strftime("%m/%d %H:%M:%S")} Evaluation: Epoch {epoch+1}: SM Loss {res["sm_loss"]:.2f} Class Loss {res["class_loss"]:.2f} Trans Loss {res["trans_loss"]:.2f} Recall {res["reg_recall"]:.2f}')
                if res['reg_recall'] > best_reg_recall:
                    best_reg_recall = res['reg_recall']
                    self._snapshot('best')

            if (epoch + 1) % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

        # finish all epoch
        if self.rank == 0:
            print(f'{time.strftime("%m/%d %H:%M:%S")} Training finish!... save training results')

    def train_epoch(self, epoch):
        # create meters and timers
        meter_list = ['class_loss', 'trans_loss', 'sm_loss', 'reg_recall', 're', 'te', 'precision', 'recall', 'f1']
        meter_dict = {}
        for key in meter_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = len(self.train_loader)
        num_iter = min(self.training_max_iter, num_iter)
        self.train_loader.sampler.set_epoch(epoch)
        trainer_loader_iter = self.train_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            input_dict = trainer_loader_iter.next()
            (corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels, _, _) = self.train_feature_extractor.process_batch(input_dict)
            if self.gpu_mode:
                corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                    corr_pos.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr_pos,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
            }
            data_timer.toc()

            model_timer.tic()
            # forward
            self.optimizer.zero_grad()
            res = self.model(data)
            pred_trans, pred_labels = res['final_trans'], res['final_labels']
            # classification loss
            class_stats = self.evaluate_metric['ClassificationLoss'](pred_labels, gt_labels)
            class_loss = class_stats['loss']
            # spectral matching loss
            sm_loss = self.evaluate_metric['SpectralMatchingLoss'](res['M'], gt_labels)
            # transformation loss
            trans_loss, reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)

            loss = self.metric_weight['ClassificationLoss'] * class_loss + self.metric_weight['SpectralMatchingLoss'] * sm_loss
            if epoch > self.transformation_loss_start_epoch and self.metric_weight['TransformationLoss'] > 0.0:
                loss += self.metric_weight['TransformationLoss'] * trans_loss
            
            stats = {
                'class_loss': float(class_loss),
                'sm_loss': float(sm_loss),
                'trans_loss': float(trans_loss),
                'reg_recall': float(reg_recall),
                're': float(re),
                'te': float(te),
                'precision': class_stats['precision'],
                'recall': class_stats['recall'],
                'f1': class_stats['f1'],
            }

            # backward
            loss.backward()
            
            self.sum_gradients()

            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break

            if do_step is True:
                self.optimizer.step()

            model_timer.toc()

            if not np.isnan(float(loss)):
                for key in meter_list:
                    if not np.isnan(stats[key]):
                        meter_dict[key].update(stats[key])

            else:  # debug the loss calculation process.
                print("Bug found, ignoring.")
                # import pdb
                # pdb.set_trace()

            if (iter + 1) % 50 == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + iter
                report = torch.tensor([1.0, data_timer.avg, model_timer.avg, meter_dict['class_loss'].avg, meter_dict['trans_loss'].avg, meter_dict['sm_loss'].avg, meter_dict['reg_recall'].avg, meter_dict['re'].avg, meter_dict['te'].avg, meter_dict['precision'].avg, meter_dict['recall'].avg, meter_dict['f1'].avg], device=torch.cuda.current_device())
                dist.all_reduce(report, op=dist.ReduceOp.SUM)
                if self.rank == 0:
                    mean_meter_dict = {}
                    count = report[0].item()
                    data_timer_avg            = report[ 1].item() / count
                    model_timer_avg           = report[ 2].item() / count
                    mean_meter_dict['class_loss']= report[3].item() / count
                    mean_meter_dict['trans_loss']= report[4].item() / count
                    mean_meter_dict['sm_loss']= report[5].item() / count
                    mean_meter_dict['reg_recall']= report[6].item() / count
                    mean_meter_dict['re']= report[7].item() / count
                    mean_meter_dict['te']= report[8].item() / count
                    mean_meter_dict['precision']= report[9].item() / count
                    mean_meter_dict['recall']= report[10].item() / count
                    mean_meter_dict['f1'] = report[11].item() / count

                    for key in meter_list:
                        self.writer.add_scalar(f"Train/{key}", mean_meter_dict[key], curr_iter)

                    print(f"{time.strftime('%m/%d %H:%M:%S')} Epoch: {epoch} [{iter+1:4d}/{num_iter}] "
                                        f"sm_loss: {mean_meter_dict['sm_loss']:.2f} "
                                        f"class_loss: {mean_meter_dict['class_loss']:.2f} "
                                        f"trans_loss: {mean_meter_dict['trans_loss']:.2f} "
                                        f"reg_recall: {mean_meter_dict['reg_recall']:.2f}% "
                                        f"re: {mean_meter_dict['re']:.2f}degree "
                                        f"te: {mean_meter_dict['te']:.2f}cm "
                                        f"data_time: {data_timer_avg:.2f}s "
                                        f"model_time: {model_timer_avg:.2f}s "
                                        )

    def evaluate(self, epoch):
        self.model.eval()

        # create meters and timers
        meter_list = ['class_loss', 'trans_loss', 'sm_loss', 'reg_recall', 're', 'te', 'precision', 'recall', 'f1']
        meter_dict = {}
        for key in meter_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = len(self.val_loader)
        num_iter = min(self.val_max_iter, num_iter)
        val_loader_iter = self.val_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            input_dict = val_loader_iter.next()
            (corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels, _, _) = self.val_feature_extractor.process_batch(input_dict)
            if self.gpu_mode:
                corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                    corr_pos.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr_pos,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
            }
            data_timer.toc()

            model_timer.tic()
            # forward
            res = self.model(data)
            pred_trans, pred_labels = res['final_trans'], res['final_labels']
            # classification loss
            class_stats = self.evaluate_metric['ClassificationLoss'](pred_labels, gt_labels)
            class_loss = class_stats['loss']
            # spectral matching loss
            sm_loss = self.evaluate_metric['SpectralMatchingLoss'](res['M'], gt_labels)
            # transformation loss
            trans_loss, reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)
            model_timer.toc()

            stats = {
                'class_loss': float(class_loss),
                'sm_loss': float(sm_loss),
                'trans_loss': float(trans_loss),
                'reg_recall': float(reg_recall),
                're': float(re),
                'te': float(te),
                'precision': class_stats['precision'],
                'recall': class_stats['recall'],
                'f1': class_stats['f1'],
            }
            for key in meter_list:
                if not np.isnan(stats[key]):
                    meter_dict[key].update(stats[key])

        self.model.train()
        
        report = torch.tensor([1.0, meter_dict['class_loss'].avg, meter_dict['trans_loss'].avg, meter_dict['sm_loss'].avg, meter_dict['reg_recall'].avg, meter_dict['re'].avg, meter_dict['te'].avg, meter_dict['precision'].avg, meter_dict['recall'].avg, meter_dict['f1'].avg], device=torch.cuda.current_device()) 
        dist.all_reduce(report, op=dist.ReduceOp.SUM)
        mean_meter_dict = {}
        count = report[0].item()
        mean_meter_dict['class_loss']= report[1].item() / count
        mean_meter_dict['trans_loss']= report[2].item() / count
        mean_meter_dict['sm_loss']= report[3].item() / count
        mean_meter_dict['reg_recall']= report[4].item() / count
        mean_meter_dict['re']= report[5].item() / count
        mean_meter_dict['te']= report[6].item() / count
        mean_meter_dict['precision']= report[7].item() / count
        mean_meter_dict['recall']= report[8].item() / count
        mean_meter_dict['f1'] = report[9].item() / count

        res = {
            'sm_loss': mean_meter_dict['sm_loss'],
            'class_loss': mean_meter_dict['class_loss'],
            'reg_recall': mean_meter_dict['reg_recall'],
            'trans_loss': mean_meter_dict['trans_loss'],
        }
        if self.rank == 0:
            for key in meter_list:
                self.writer.add_scalar(f"Val/{key}", mean_meter_dict[key], epoch)
        return res

    def _snapshot(self, epoch):
        if self.rank == 0:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch}.pkl"))
            print(f"Save model to {self.save_dir}/model_{epoch}.pkl")

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Load model from {pretrain}.pkl")
