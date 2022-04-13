import numpy as np
import os
import datetime
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from dataloader.generic_balanced_loader import send_arrs_to_device
from dataloader.dataloader import make_dataloader, get_dataset_name
from general.TicToc import *
from utils.experiment_utils import print_to_file_and_screen, generate_output_dir
from net.train_FCGF import FCGF_trainer

def initialize_args():
    DATASET = 'A' # 
    SHUFFLE = True
    DO_VALIDATION = True
    STAT_FREQ = 40 # iters
    VALIDATION_FREQ = 5 # epochs
    SAVE_FREQ = 50 # epochs
    FIRST_EPOCH = 0
    MAX_PAIRS = np.inf
    LOCAL_BATCH_SIZE = 6
    NUM_EPOCHS = 400
    
    assert len(sys.argv) > 1, "First command line argument should be the name of the dataset (see dataloader.py)"
    DATASET = sys.argv[1]    
    
    RESUME = False
    if len(sys.argv) > 2:
        RESUME = sys.argv[2]
        if RESUME == "0":
            RESUME = False
        else:
            err_msg = "Second argument should be 0 (don't resume) or path to a checkpoint file"
            assert os.path.isfile(RESUME), err_msg
            try:
                FIRST_EPOCH = int(RESUME.split('.')[-2])
            except ValueError as E:
                if "invalid literal for int() with base 10" in str(E):
                    pass
                else:
                    raise E
            except IndexError as E:
                if "list index out of range" in str(E):
                    pass
                else:
                    raise E

    
    if MAX_PAIRS < LOCAL_BATCH_SIZE:
        LOCAL_BATCH_SIZE = MAX_PAIRS

    start_time = time()

    args = {}
    args['DATASET'] = DATASET
    args['STAT_FREQ'] = STAT_FREQ
    args['SAVE_FREQ'] = SAVE_FREQ
    args['MAX_PAIRS'] = MAX_PAIRS
    args['RESUME'] = RESUME
    args['LOCAL_BATCH_SIZE'] = LOCAL_BATCH_SIZE
    args['NUM_EPOCHS'] = NUM_EPOCHS
    args['FIRST_EPOCH'] = FIRST_EPOCH
    args['SHUFFLE'] = SHUFFLE
    args['DO_VALIDATION'] = DO_VALIDATION
    args['VALIDATION_FREQ'] = VALIDATION_FREQ
    args['start_time'] = start_time
    output_dir = generate_output_dir(get_dataset_name(DATASET), 'Train')
    args['output_dir'] = output_dir
    args['outfile_name'] = output_dir + 'log.txt'
    return args

def do_validation(rank, Engine, validation_loader, outfile, train_step, start_time, device, name='validation', deterministic=True):

    if validation_loader is None:
        return

    all_losses = None
    total_actual_batch_size = 0
    for GT_motion, arrs in validation_loader:
        GT_motion = GT_motion.to(device)
        arrs = send_arrs_to_device(arrs, device)
        report, report_names = Engine.validation_batch(GT_motion, arrs, deterministic=deterministic)
        dist.all_reduce(report, op=dist.ReduceOp.SUM)
        if rank == 0:
            losses, actual_batch_size = parse_report(report, report_names)
            total_actual_batch_size += actual_batch_size
            if all_losses is None:
                all_losses = {k: 0.0 for k in losses.keys()}
            for k in losses.keys():
                all_losses[k] += actual_batch_size*losses[k]

    if rank==0:
        for k in all_losses.keys():
            all_losses[k] /= total_actual_batch_size
        print_report_line(Engine, outfile, all_losses, start_time, train_step, name)
        outfile.flush()        

    try:
        return all_losses['angular_error'], all_losses['translation_error']
    except:
        return None, None


def print_report_line(TS, outfile, losses, start_time, train_iter_tuple, dataname):
    time_elapsed = time() -  start_time
    hours = np.floor(time_elapsed/3600)
    remainder1 = time_elapsed - hours*3600
    minutes = np.floor(remainder1/60)
    remainder2 = remainder1 - minutes*60
    seconds = np.floor(remainder2)
    millisecs = np.round(1000*(remainder2-seconds))
    time_str = '%02d:%02d:%02d.%03d' % ( hours, minutes, seconds, millisecs)
    s = "%s | %d [%d/%d]: %s: " % (time_str, train_iter_tuple[0], train_iter_tuple[1], train_iter_tuple[2], dataname)
    for k in losses.keys():
        s += k + ": %f, " % losses[k]

    print_to_file_and_screen(outfile, s[:-2])    
    outfile.flush()
    TS.record_losses(losses, train_iter_tuple, dataname)

def parse_report(report, report_names):
    assert (report_names[0] == 'actual_batch_size')
    num_fields = len(report_names)
    total_actual_batch_size = report[0].detach().cpu().numpy()
    
    losses_np = {}
    for j in range(1,num_fields): # skipping the first field that is 'actual_batch_size'
        key = report_names[j]
        val = report[j].detach().cpu().numpy()
        losses_np[key] = val / total_actual_batch_size
        
    return losses_np, total_actual_batch_size

def print_args(args, outfile):
        
    s = "===========\n"
    s += "Parameters:\n===========\n"
    for k in args:
        s += (k + "=" + str(args[k]) + "\n")
    s += "===========\n"
    print_to_file_and_screen(outfile, s)


def parse_and_print_report(report, report_names, Trainer, outfile, args, iter):
    losses_np, _ = parse_report(report, report_names)
    for i, name in enumerate(report_names):
        if '_of_successful' in name:
            losses_np[i] /= (1.0 - losses_np[report_names.index('failure_rate')])
    print_report_line(Trainer, outfile, losses_np, args['start_time'], iter, 'train')

def train_parallel(rank, world_size, seed, args):
    # This function is performed in parallel in several processes, one for each available GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8700'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(rank)

    if rank==0:
        outfile = open(args['outfile_name'], 'w+')
        print_to_file_and_screen(outfile, outfile.name)
        print_args(args, outfile)
    else:
        outfile = args['outfile_name']

    if rank==0:
        print_to_file_and_screen(outfile, "local batch size: %d" % (args['LOCAL_BATCH_SIZE']))

    device = 'cuda:%d' % torch.cuda.current_device()
    print("process %d, GPU: %s" % (rank, device))        
    
    train_loader = make_dataloader('train', args, world_size, rank, seed, outfile)
    val_loader = make_dataloader('validation', args, world_size, rank, seed, outfile)

    Trainer = FCGF_trainer(
        outfile,
        train_loader.dataset.name,
        lr=0.1,
        l2_reg=0.0001,
        resume=args['RESUME'],
        shuffle=args['SHUFFLE'],
        device=device,
        rank=rank
    )

    for epoch in range(args['FIRST_EPOCH'], args['NUM_EPOCHS']+1): # XXXXX
        it = 0
        train_loader.sampler.set_epoch(epoch)
        if (rank==0) and ((epoch % args['SAVE_FREQ']) == 0):
            Trainer.save_model(epoch)

        if args['DO_VALIDATION'] and ((epoch % args['VALIDATION_FREQ']) == 0):
            do_validation(rank, Trainer, val_loader, outfile, [epoch, 0, len(train_loader)], args['start_time'], device, name='validation')

        for GT_motion, arrs in train_loader:
            iter_tuple = [epoch, it, len(train_loader)]
            GT_motion = GT_motion.to(device)
            arrs = send_arrs_to_device(arrs, device)
            report, report_names = Trainer.train_batch(GT_motion, arrs)
            dist.all_reduce(report, op=dist.ReduceOp.SUM)
            if (rank == 0) and (it % args['STAT_FREQ'] == 0):
                parse_and_print_report(report, report_names, Trainer, outfile, args, iter_tuple)
            it += 1

        Trainer.advance_scheduler()
    
    if (rank==0):
        Trainer.save_model(epoch)
        outfile_name = outfile.name
        outfile.close()
        print("Wrote Results to file %s" % outfile_name)


def main():
    args = initialize_args()
    if args['SHUFFLE']:
        seed = np.random.randint(np.iinfo(np.int32).max)        
    else:
        seed = 0
    world_size = torch.cuda.device_count()
    print("%d GPUs are available" % world_size)
    if world_size>1:
        mp.spawn(train_parallel, nprocs=world_size, args=(world_size,seed, args))
    else: # sometimes useful for debugging
        train_parallel(0, world_size, seed, args)

if __name__ == "__main__":
    main()

# END OF CODE
