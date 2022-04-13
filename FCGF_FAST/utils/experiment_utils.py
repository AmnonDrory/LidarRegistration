from glob import glob
import open3d as o3d
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pickle
import os
import datetime
from shutil import move, copyfileobj
from time import sleep
import errno

from utils.tools_3d import euler_angles_to_rotation_matrix, rotate_3d, apply_rot_trans, quaternion_to_euler, rotation_matrix_euler_angles, calc_angular_error, calc_translational_error
import matplotlib.pyplot as plt
from utils.algo_utils import guess_best_alpha, bb_loss_soft, bb_loss, bb_dist_loss_soft, bb_dist_loss
from utils.subsampling import get_random_subset, get_local_statistics, subsample_fraction, remove_road
from utils.visualization import draw_registration_result, visualize_gaussian
from general.TicToc import *
from utils.PointCloudUtils import dist

def show_result_graphs(full_log, title=None, blocking=True):
    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(full_log["loss"])
    plt.title('loss')
    min_ind = np.argmin(full_log["loss"])
    axs = plt.axis()
    plt.plot([min_ind]*2,axs[2:],'--')
    plt.subplot(3, 3, 2)
    plt.plot(full_log["alpha"])
    plt.title('alpha')
    plt.subplot(3, 3, 3)
    if "heading" in full_log.keys():
        plt.plot(np.array(full_log["heading"])*np.array(full_log["grad_magnitude"]))
        plt.title('aligned-gradient')
    else:
        plt.plot(full_log["trans"][:, 0], full_log["loss"])
        plt.title('loss per x')

    for i in range(3):
        plt.subplot(3, 3, 4 + i)
        plt.plot(full_log["angles"][:, i])
        plt.title('angle[%d]' % i)
    for i in range(3):
        plt.subplot(3, 3, 7 + i)
        plt.plot(full_log["trans"][:, i])
        plt.title('translation[%d]' % i)
    plt.suptitle(title)
    if blocking:
        plt.show()

def visualize_loss_function_translation(GT_motion,A_sub,B_sub,init_alpha,full_log,title,loss_type,feature_mode='xyz'):
    plt.figure()
    edge1 = -GT_motion['trans'][0]
    edge2 = -2 * edge1
    m = np.minimum(edge1, edge2)
    M = np.maximum(edge1, edge2)
    offsets = np.linspace(m, M, 121)
    soft_BBS_loss = []
    hard_BBS_loss = []
    for offs in offsets:
        cur_B = B_sub + np.array([[offs, 0, 0]])
        if loss_type == 'BBS':
            soft_BBS_loss.append( bb_loss_soft(A_sub, cur_B, init_alpha, feature_mode) )
            hard_BBS_loss.append( bb_loss(A_sub, cur_B, feature_mode))
        elif loss_type == 'BD':
            soft_BBS_loss.append( bb_dist_loss_soft(A_sub, cur_B, init_alpha, feature_mode))
            hard_BBS_loss.append( bb_dist_loss(A_sub, cur_B, feature_mode))
        else:
            assert False, "unknown loss type " + loss_type

    plt.plot(offsets, soft_BBS_loss, 'b')
    plt.plot(offsets, hard_BBS_loss, 'g')
    plt.plot(full_log["trans"][:, 0], full_log["loss"], 'r')
    plt.title(title)

    plt.show()

def visualize_loss_function_yaw(GT_motion,A_sub,B_sub,init_alpha,full_log,title,loss_type,feature_mode='xyz'):
    if 'BBS' in loss_type:
        loss_type = 'BBS'
    plt.figure()
    edge1 = -GT_motion['angles'][2]
    edge2 = -2 * edge1
    m = np.minimum(edge1, edge2)
    M = np.maximum(edge1, edge2)
    yaws = np.linspace(m, M, 41)
    soft_BBS_loss = []
    hard_BBS_loss = []
    for yaw in yaws:
        cur_B = apply_rot_trans(B_sub, [0,0,yaw],[0,0,0],feature_mode)
        if loss_type == 'BBS':
            soft_BBS_loss.append( bb_loss_soft(A_sub, cur_B, init_alpha,feature_mode) )
            hard_BBS_loss.append( bb_loss(A_sub, cur_B,feature_mode))
        elif loss_type == 'BD':
            soft_BBS_loss.append( bb_dist_loss_soft(A_sub, cur_B, init_alpha,feature_mode))
            hard_BBS_loss.append( bb_dist_loss(A_sub, cur_B,feature_mode))
        else:
            assert False, "unknown loss type " + loss_type

    plt.plot(yaws, soft_BBS_loss, 'b')
    plt.plot(yaws, hard_BBS_loss, 'g')
    plt.plot(full_log["angles"][:, 2], full_log["loss"], 'r')
    plt.title(title)

    plt.show()

def show_result_point_clouds(PCs, A,B, A_sub, B_sub, res_motion, feature_mode, mot_grid, mot_BBS=None):
    B_sub_corrected = apply_rot_trans(B_sub, res_motion['angles'], res_motion['trans'], feature_mode)
    draw_registration_result(A_sub, B_sub_corrected)
    B_corrected = apply_rot_trans(B, res_motion['angles'], res_motion['trans'], feature_mode)
    draw_registration_result(A, B_corrected)
    B0 = PCs[1]
    if mot_grid is not None:
        B1 = apply_rot_trans(B0, mot_grid['angles'], mot_grid['trans'], feature_mode)
    else:
        B1 = B0
    if mot_BBS is not None:
        B2 = apply_rot_trans(B1, mot_BBS['angles'], mot_BBS['trans'], feature_mode)
    else:
        B2 = B1
    B3 = apply_rot_trans(B2, res_motion['angles'], res_motion['trans'], feature_mode)
    draw_registration_result(A, B3, "Final motion")
    

def get_quantiles(data, Q):
    Q = np.array(Q)
    assert (np.min(Q) >= 0.) and (np.max(Q) <= 1.)
    s = np.sort(data)
    inds = np.round((len(s)-1)*Q).astype(int)
    return s[inds]

def print_to_file_and_screen(outfile, s, *args, **vargs):
    if outfile is None:
        print(s, *args, **vargs)
        return
    try:
        print(s, *args, **vargs, file=outfile)
        print(s, *args, **vargs)
    except AttributeError as E:
        if "object has no attribute" in str(E):
            pass
        else:
            raise E

def loss_values_along_route(outfile, A,B_rot_trans, init_alpha, GT_mot, title=None):
    VISUALIZE = True
    LOAD_FROM_PICKLE = False
    pickle_file = bb_pc_path['pickle_dir'] + "loss_values_along_route.pickle"

    if LOAD_FROM_PICKLE and os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as fid:
            [cur_mot_list, offset_fractions,alpha_vals , hard_losses , soft_losses , hard_dist_losses , soft_dist_losses] = pickle.load(fid)
    else:
        steps = 21
        offset_fractions = np.linspace(0,1.5,steps)

        alpha_vals = np.array([0.25, 0.5, 1, 2, 4]) * init_alpha
        hard_losses = np.zeros([len(alpha_vals),len(offset_fractions)])
        soft_losses = np.zeros([len(alpha_vals),len(offset_fractions)])
        hard_dist_losses = np.zeros([len(alpha_vals),len(offset_fractions)])
        soft_dist_losses = np.zeros([len(alpha_vals),len(offset_fractions)])

        cur_mot_list = []
        for f_i, f in enumerate(offset_fractions):
            cur_mot = deepcopy(GT_mot)
            cur_mot['angles'] *= f
            cur_mot['trans'] *= f
            cur_mot_list.append(cur_mot)

            cur_B = apply_rot_trans(B_rot_trans, cur_mot['angles'], cur_mot['trans'])
            for a_i, alpha in enumerate(alpha_vals):
                soft_losses[a_i, f_i] = bb_loss_soft(A,cur_B,alpha)
                if a_i == 0:
                    hard_losses[a_i, f_i] = bb_loss(A,cur_B)
                soft_dist_losses[a_i, f_i] = bb_dist_loss_soft(A,cur_B,alpha)
                if a_i == 0:
                    hard_dist_losses[a_i, f_i] = bb_dist_loss(A,cur_B)
        with open(pickle_file, 'wb') as fid:
            pickle.dump([cur_mot_list, offset_fractions,alpha_vals , hard_losses , soft_losses , hard_dist_losses , soft_dist_losses], fid)

    best_ind = np.argmin(hard_losses[0,:])
    suggested_correction = cur_mot_list[best_ind]
    print_to_file_and_screen(outfile, "optimal correction along line: " + str(suggested_correction))
    print_to_file_and_screen(outfile, "loss at optimal correction: %f" % hard_losses[0,best_ind])
    if VISUALIZE:
        # for fraction in np.linspace(0, 1, 11):
        #     plt.figure()
        #     ax = plt.gca()
        #     cur_hard = (1.-fraction)*hard_losses + fraction*hard_dist_losses
        #     cur_soft = (1.-fraction)*soft_losses + fraction*soft_dist_losses
        #     ax.plot(offset_fractions, cur_hard[0,:], label='hard')
        #     for a_i, alpha in enumerate(alpha_vals):
        #         ax.plot(offset_fractions, cur_soft[a_i,:], label='soft alpha=%f' % alpha)
        #     ax.legend()
        #     plt.title(fraction)
        #
        # plt.show()

        plt.figure()
        ax = plt.gca()
        ax.plot(offset_fractions, hard_losses[0,:], label='hard')
        for a_i, alpha in enumerate(alpha_vals):
            ax.plot(offset_fractions, soft_losses[a_i,:], label='soft alpha=%f' % alpha)
        ax.legend()
        if title is not None:
            plt.title(title)
        plt.figure()
        ax = plt.gca()
        ax.plot(offset_fractions, hard_dist_losses[0,:], label='hard')
        for a_i, alpha in enumerate(alpha_vals):
            ax.plot(offset_fractions, soft_dist_losses[a_i,:], label='soft alpha=%f' % alpha)
        ax.legend()
        if title is not None:
            plt.title(title)
        plt.show()

    return suggested_correction


def calc_loss(type, softness, density, *args, **kwargs):
    if (type=='BBS') and (softness=='soft') and (density=='dense'):
        return bb_loss_soft(*args, **kwargs)
    elif (type=='BBS') and (softness=='soft') and (density=='sparse'):
        return bb_loss_soft_sparse_gradients(*args, **kwargs)
    elif (type=='BBS') and (softness=='hard'):
        return bb_loss(*args, **kwargs)
    else:
        return 0

def draw_loss_along_yaw(A,B,init_alpha,log, thetas=None):

    if thetas is None:
        thetas = np.linspace(0,30,21)
    if log is not None:
        thetas = np.sort(np.hstack([thetas, log['angles'][:,2]]))

    alpha_factors = [10**-3, 10**-4, 10**-5]
    per_alpha = []

    hard_BBS_loss = []
    for ai, alpha_factor in enumerate(alpha_factors):
        soft_BBS_loss = []
        for theta in thetas:
            cur_B = rotate_3d(B, [0,0,theta])
            cur_soft_loss = calc_loss('BBS', 'soft', 'sparse', A, cur_B, init_alpha)
            soft_BBS_loss.append(cur_soft_loss)
            if ai == 0:
                cur_hard_loss = calc_loss('BBS', 'hard', 'sparse', A, cur_B)
                hard_BBS_loss.append( cur_hard_loss )
        per_alpha.append(soft_BBS_loss)


    plt.figure()
    ax = plt.gca()
    for ai, alpha_factor in enumerate(alpha_factors):
        ax.plot(thetas, per_alpha[ai], label=str(ai) )
    ax.plot(thetas, hard_BBS_loss, label='hard')
    # if log is not None:
    #     plt.plot(log['angles'][:,2],log['loss'],'-ro')
    ax.legend()
    plt.title('loss along yaw axis')
    plt.show()

def draw_loss_along_x(A,B,init_alpha,log, offsets=None):

    if offsets is None:
        offsets = np.linspace(0,5,31)
    if log is not None:
        log_xs = log['trans'][:,0]
        log_ys = log['loss']
        offsets = np.sort(np.hstack([offsets, log_xs]))
    soft_BBS_loss = []
    hard_BBS_loss = []
    for offset in offsets:
        cur_B = B + np.array([offset, 0, 0])
        cur_soft_loss = calc_loss('BBS', 'soft', 'sparse', A, cur_B, init_alpha)
        soft_BBS_loss.append(cur_soft_loss)
        cur_hard_loss = calc_loss('BBS', 'hard', 'sparse', A, cur_B)
        hard_BBS_loss.append( cur_hard_loss )

    plt.figure()
    plt.plot(offsets, soft_BBS_loss, 'b')
    plt.plot(offsets, hard_BBS_loss, 'g')
    if log is not None:
        plt.plot(log_xs,log_ys,'-ro')
    plt.title('loss along x axis')
    plt.show()

def draw_loss_along_y(A,B,init_alpha,log, offsets=None):

    if offsets is None:
        offsets = np.linspace(0,3,31)
    if log is not None:
        log_xs = log['trans'][:,1]
        log_ys = log['loss']
        offsets = np.sort(np.hstack([offsets, log_xs]))
    soft_BBS_loss = []
    hard_BBS_loss = []
    for offset in offsets:
        cur_B = B + np.array([offset, 0, 0])
        cur_soft_loss = calc_loss('BBS', 'soft', 'sparse', A, cur_B, init_alpha)
        soft_BBS_loss.append(cur_soft_loss)
        cur_hard_loss = calc_loss('BBS', 'hard', 'sparse', A, cur_B)
        hard_BBS_loss.append( cur_hard_loss )

    plt.figure()
    plt.plot(offsets, soft_BBS_loss, 'b')
    plt.plot(offsets, hard_BBS_loss, 'g')
    if log is not None:
        plt.plot(log_xs,log_ys,'-ro')
    plt.title('loss along y axis')
    plt.show()


def visualize_step_size(full_log):
    if full_log is None:
        return
    H = 3
    W = 4
    heading = np.array(full_log["heading"])
    grad_heading = np.array(full_log["grad_heading"])
    inds = np.arange(len(heading))

    plt.figure()
    plt.subplot(H, W, 1 + 0 + 0)
    l = np.array(full_log['loss'])
    plt.plot(l)
    plt.title('loss')
    plt.subplot(H, W, 1 + 0 + 1)
    plt.plot(full_log['distance_to_GT'])
    plt.title('distance_to_GT')
    plt.subplot(H, W, 1 + 0 + 2)
    plt.plot(full_log['cos_angle_to_prev_heading'])
    plt.title('cos_angle_to_prev_heading')
    plt.subplot(H, W, 1 + 0 + 3)
    # plt.plot(full_log['cos_angle_between_step_and_gradient'])
    # plt.title('cos_angle_between_step_and_gradient')
    plt.plot(full_log['LR'])
    plt.title('LR')

    is_right_way_by_step = heading >= 0
    plt.subplot(H, W, 1 + W + 0)
    plt.plot(inds, heading, c='b')
    plt.plot(inds[is_right_way_by_step], heading[is_right_way_by_step], '.', c='g', markersize=3)
    plt.plot(inds[~is_right_way_by_step], heading[~is_right_way_by_step], '.', c='r', markersize=3)
    plt.title('heading')
    plt.subplot(H, W, 1 + W + 1)
    s = np.array(full_log['step_size'])
    plt.plot(s)
    plt.plot(inds[is_right_way_by_step], s[is_right_way_by_step], '.', c='g', markersize=3)
    plt.plot(inds[~is_right_way_by_step], s[~is_right_way_by_step], '.', c='r', markersize=3)
    plt.title('step_size')
    plt.subplot(H, W, 1 + W + 2)
    x = np.array(full_log['optimal_move_size_by_step'])
    plt.plot(x)
    plt.plot(inds[is_right_way_by_step], x[is_right_way_by_step], '.', c='g', markersize=3)
    plt.plot(inds[~is_right_way_by_step], x[~is_right_way_by_step], '.', c='r', markersize=3)
    plt.title('optimal_move_size_by_step')
    plt.subplot(H, W, 1 + W + 3)
    x = np.array(full_log['optimal_LR_by_step'])
    plt.plot(x)
    plt.plot(inds[is_right_way_by_step], x[is_right_way_by_step], '.', c='g', markersize=3)
    plt.plot(inds[~is_right_way_by_step], x[~is_right_way_by_step], '.', c='r', markersize=3)
    plt.title('optimal_LR_by_step')

    is_right_way_by_grad = grad_heading >= 0
    plt.subplot(H, W, 1 + 2 * W + 0)
    plt.plot(inds, grad_heading, c='b')
    plt.plot(inds[is_right_way_by_grad], grad_heading[is_right_way_by_grad], '.', c='g', markersize=3)
    plt.plot(inds[~is_right_way_by_grad], grad_heading[~is_right_way_by_grad], '.', c='r', markersize=3)
    plt.title('grad_heading')
    plt.subplot(H, W, 1 + 2 * W + 1)
    s = np.array(full_log['grad_magnitude'])
    plt.plot(s)
    plt.plot(inds[is_right_way_by_grad], s[is_right_way_by_grad], '.', c='g', markersize=3)
    plt.plot(inds[~is_right_way_by_grad], s[~is_right_way_by_grad], '.', c='r', markersize=3)
    plt.title('grad_magnitude')
    plt.subplot(H, W, 1 + 2 * W + 2)
    x = np.array(full_log['optimal_move_size_by_grad'])
    plt.plot(x)
    plt.plot(inds[is_right_way_by_grad], x[is_right_way_by_grad], '.', c='g', markersize=3)
    plt.plot(inds[~is_right_way_by_grad], x[~is_right_way_by_grad], '.', c='r', markersize=3)
    plt.title('optimal_move_size_by_grad')
    plt.subplot(H, W, 1 + 2 * W + 3)
    x = np.array(full_log['optimal_LR_by_grad'])
    plt.plot(x)
    plt.plot(inds[is_right_way_by_grad], x[is_right_way_by_grad], '.', c='g', markersize=3)
    plt.plot(inds[~is_right_way_by_grad], x[~is_right_way_by_grad], '.', c='r', markersize=3)
    plt.title('optimal_LR_by_grad')
    plt.show()

def extract_accuracy_from_log(GT_motion,log):
    ang = log['angles']
    trans = log['trans']
    num_iters = ang.shape[0]
    mot = [{'angles': ang[i, :], 'trans': trans[i, :]} for i in range(num_iters)]
    ang_err = np.zeros(num_iters)
    trans_err = np.zeros(num_iters)
    for iter in range(num_iters):
        ang_err[iter] = calc_angular_error(GT_motion['angles'], mot[iter]['angles'])
        trans_err[iter] = calc_translational_error(GT_motion, mot[iter])
    return ang_err, trans_err

def extract_batch_accuracy_from_logs(all_GT_motions,all_logs):
    ang_err = 0
    trans_err = 0
    num_pairs = len(all_logs)
    for pair_ind in range(num_pairs):
        log = all_logs[pair_ind]
        GT_motion = all_GT_motions[pair_ind]
        cur_ang_err, cur_trans_err = extract_accuracy_from_log(GT_motion, log)
        ang_err += cur_ang_err
        trans_err += cur_trans_err
    return ang_err/num_pairs, trans_err/num_pairs

def analyze_accuracy_over_time(logs_file_1=None, logs_file_2=None):

    if logs_file_1 is None:
        logs_file_1 = '/root/PycharmProjects/CloudBuddies_Fork/shared_data/registration_results/Apollo.Result.20200602_08_55_02.txt.logs'
        logs_file_2 = '/root/PycharmProjects/CloudBuddies_Fork/shared_data/registration_results/Apollo.Result.20200602_08_58_54.txt.logs'

    with open(logs_file_1, 'rb') as fid:
        _,_,all_GT_motions_1,all_logs_1 = pickle.load(fid)

    if logs_file_2 is not None:
        with open(logs_file_2, 'rb') as fid:
            _,_,all_GT_motions_2,all_logs_2 = pickle.load(fid)

    ang_err_1, trans_err_1 = extract_batch_accuracy_from_logs(all_GT_motions_1, all_logs_1)
    if logs_file_2 is not None:
        ang_err_2, trans_err_2 = extract_batch_accuracy_from_logs(all_GT_motions_2, all_logs_2)

    plt.subplot(121)
    plt.plot(ang_err_1,'g')
    if logs_file_2 is not None:
        plt.plot(ang_err_2, 'r')
    plt.title('angular')
    plt.subplot(122)
    plt.plot(trans_err_1, 'g')
    if logs_file_2 is not None:
        plt.plot(trans_err_2, 'r')
    plt.title('translation')
    plt.show()

def analyze_LR():
    nIterations = 3
    with open(bb_pc_path['pickle_dir'] + 'all_logs_%d.pickle' % nIterations, 'rb') as fid:
        all_logs = pickle.load(fid)

    optimal_LR_by_step = []
    for log in all_logs:
        optimal_LR_by_step.append(np.array(log['optimal_LR_by_step']).reshape(1,-1))
    optimal_LR_by_step = np.vstack(optimal_LR_by_step)

    optimal_LR_by_step[optimal_LR_by_step<0] = np.nan
    a50 = np.nanmedian(optimal_LR_by_step,axis=0)
    a10 = np.nanpercentile(optimal_LR_by_step,10,axis=0)
    a90 = np.nanpercentile(optimal_LR_by_step, 90,axis=0)
    plt.plot(a50,'r')
    plt.plot(a10, 'g')
    plt.plot(a90, 'b')

    # x = 1 + np.arange(optimal_LR_by_step.shape[1])
    # A = np.vstack([x, np.ones(len(x))]).T
    # m,c = np.linalg.lstsq(A, np.log(a10).reshape([-1,1]),rcond=None)[0]
    # print(m,c)
    # y = np.exp(m*x + c)
    # plt.plot(y,'k')
    #
    # def LR_func(iter):
    #     m = -0.019
    #     c = -2.776
    #     b = 0.30093351103364097
    #     a = -0.01877210099534973
    #     f1 = np.exp(m * iter + c)
    #     f2 = a * iter + b
    #     return np.maximum(f1, f2)
    #
    # z = LR_func(x)
    # #print(a10[0], ((a10[13]-a10[0])/13))
    # plt.plot(z, 'm')

    plt.show()
    # assume a_i = A*exp(B*i)
    # then log(a_i) = log(A) + B*i
    # find linear fit for log(a_i)
    # L = prod_1_100[ (A*np.exp(B*x))**2 ]
    # logL =

def analyze_single_pair_test_validation(filename=None):
    if filename is None:
        filename = '/root/PycharmProjects/CloudBuddies_Fork/shared_data/registration_results/LearnedFeatures.Result.20200914_15_40_43.txt.validation'

    with open(filename, 'r') as fid:
        text = fid.read().splitlines()
    header = text[0].split()
    text = text[1:]
    num_flds = int(header[0])
    flds = header[1:num_flds+1]
    init_mot_factors = np.array([float(a) for a in header[(num_flds+1):]])
    dist_factors = 1-init_mot_factors

    data = {fld: np.zeros([len(text),len(init_mot_factors)]) for fld in flds}
    factor_inds = []
    for line_ind, line in enumerate(text):
        l = line.split()
        factor_inds.append(int(l[0]))
        to_ = 1
        for fld in flds:
            from_ = to_
            to_ = from_ + len(init_mot_factors)
            data[fld][line_ind,:] = np.array([float(a) for a in l[from_:to_]])

    ranges = {fld: [np.min(data[fld]), np.max(data[fld])] for fld in flds }

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()

    fig.show()
    fig.canvas.draw()
    fld = flds[1]
    for i in range(data[fld].shape[0]):
        ax.clear()
        plt.plot(dist_factors, data[fld][i,:])
        min_y = ranges[fld][0]
        max_y = ranges[fld][1]
        plt.plot([dist_factors[factor_inds[i]]] * 2, [min_y, max_y], 'g')
        plt.title(fld)
        fig.canvas.draw()
        sleep(0.1)

    input("Press Enter to finish")


    # for subplot_ind, fld in enumerate(flds):
    #     plt.subplot(3, 1, subplot_ind+1)
    #     plt.plot(dist_factors, data[fld])
    #     min_y = np.min(data[fld])
    #     max_y = np.max(data[fld])
    #     plt.plot([dist_factors[factor_ind]]*2, [min_y,max_y],'g')
    #     plt.title(fld)
    # plt.show()

def generate_output_dir(dataset_name, phase, start_time=None):
    if start_time is not None:
        current_time_str = start_time
    else:
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    output_dir =  f"outputs/{dataset_name}.{phase}.{current_time_str}/"
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e        
    
    return output_dir

