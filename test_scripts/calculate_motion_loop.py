from absl import app
from absl import flags
from absl import logging

import os
from librosa import beat
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal
from aist_plusplus.loader import AISTDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--anno_dir', type=str, default='[data_root]/aist_plusplus_final',
    help='Path to the AIST++ annotation files.')
parser.add_argument(
    '--audio_dir', type=str, default='[data_root]/wav',
    help='Path to the AIST wav files.')
parser.add_argument(
    '--audio_cache_dir', type=str, default='[data_root]/aistpp_feature_mirror/audio_feature_20',
    help='Path to cache dictionary for audio features.')
parser.add_argument(
    '--motion_cache_dir', type=str, default='[data_root]/aistpp_feature_mirror/motion_feature_20',
    help='Path to cache dictionary for audio features.')
parser.add_argument(
    '--split', type=str, default='testval',
    help='Whether do training set or testval set.')
parser.add_argument(
    '--result_files', type=str, default='',
    )

FLAGS = parser.parse_args()

def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    transl = motion[:, :, :3]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 3:219], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
    return keypoints3d

import random
def random_sample(keypoints3d, sample_len=40):
    seq_len = keypoints3d.shape[0]
    seq1_start = random.sample(list(range(seq_len-sample_len*2)), 1)[0]
    seq2_start = random.sample(list(range(seq1_start+sample_len, seq_len-sample_len)), 1)[0]
    return keypoints3d[seq1_start:seq1_start+sample_len], keypoints3d[seq2_start:seq2_start+sample_len]


from fastdtw import fastdtw
def dtw_sim_pos(motion1, motion2):
    """
    :param motion1: [t1, v, 3]
    :param pose2: [t2, v, 3]
    :return:
    """
    def dist_fun(v1, v2):
        dist = np.linalg.norm((v1-v2), axis=-1)
        res = np.mean(dist)
        return res
    return fastdtw(motion1, motion2, radius=4, dist=dist_fun)

def motion_loop_score(keypoints3d, sample_num=20):
    scores = []
    keypoints3d_root = keypoints3d[:, 0:1, :]
    keypoints3d_del_root = keypoints3d - keypoints3d_root
    for _ in range(sample_num):
        seq_1, seq_2 = random_sample(keypoints3d_del_root)
        seq_1 = keypoints3d_del_root[:40, :, :]
        dist = dtw_sim_pos(seq_1, seq_2)[0]
        scores.append(dist)
    return sum(scores) / sample_num


def main(_):
    import glob
    import tqdm
    from smplx import SMPL

    # set smpl
    smpl = SMPL(model_path="[data_root]/SMPL_python_v.1.1.0/smpl/models", gender='MALE', batch_size=1)

    # create list
    seq_names = []
    if "train" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_train.txt"), dtype=str
        ).tolist()
    if "val" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_val.txt"), dtype=str
        ).tolist()
    if "test" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_test.txt"), dtype=str
        ).tolist()
    ignore_list = np.loadtxt(
        os.path.join(FLAGS.anno_dir, "ignore_list.txt"), dtype=str
    ).tolist()
    seq_names = [name for name in seq_names if name not in ignore_list]

    # calculate score on real data
    # dataset = AISTDataset(FLAGS.anno_dir)
    # n_samples = len(seq_names)
    # consistancy_scores = []
    # for i, seq_name in enumerate(seq_names):
    #     logging.info("processing %d / %d" % (i + 1, n_samples))
    #     # get real data motion beats
    #     smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
    #         dataset.motion_dir, seq_name)
    #     smpl_trans /= smpl_scaling
    #     keypoints3d = smpl.forward(
    #         global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
    #         body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
    #         transl=torch.from_numpy(smpl_trans).float(),
    #     ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
    #     # get beat alignment scores
    #     consistancy_score = motion_loop_score(keypoints3d)
    #     consistancy_scores.append(consistancy_score)
    # print ("\nConsistancy score on real data: %.3f\n" % (sum(consistancy_scores) / n_samples))

    # calculate score on generated motion data
    result_files = sorted(glob.glob(FLAGS.result_files+"/*.npy"))

    n_samples = len(result_files)
    consistancy_scores = []
    for result_file in tqdm.tqdm(result_files):
        result_motion = np.load(result_file)[None, ...]  # [1, 120 + 1200, 225]
        keypoints3d = recover_motion_to_keypoints(result_motion, smpl)

        consistancy_score = motion_loop_score(keypoints3d)
        consistancy_scores.append(consistancy_score)
    print ("\nConsistancy score on generated data: %.3f\n" % (sum(consistancy_scores) / n_samples))


if __name__ == '__main__':
    app.run(main)
