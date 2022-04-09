import vedo
import torch
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg

# See https://github.com/google/aistplusplus_api/ for installation
from aist_plusplus.utils import ffmpeg_video_write
from aist_plusplus.loader import AISTDataset
from aist_plusplus.visualizer import plot_kpt
from aist_plusplus.visualizer import plot_on_video


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
    # assert dim == 221
    transl = motion[:, :, :3]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 3:219], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def visualize(keypoints3d):
    bbox_center = (
                          keypoints3d.reshape(-1, 3).max(axis=0)
                          + keypoints3d.reshape(-1, 3).min(axis=0)
                  ) / 2.0
    bbox_size = (
            keypoints3d.reshape(-1, 3).max(axis=0)
            - keypoints3d.reshape(-1, 3).min(axis=0)
    )
    world = vedo.Box(bbox_center, bbox_size[0], bbox_size[1], bbox_size[2]).wireframe()
    vedo.show(world, axes=True, viewup="y", interactive=0)
    for kpts in keypoints3d:
        pts = vedo.Points(kpts).c("red")
        plotter = vedo.show(world, pts)
        if plotter.escaped: break  # if ESC
        time.sleep(0.05)
    vedo.interactive().close()


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]  # (seq_len, 24, 3)
    return keypoints3d


import subprocess
def run_bash_command(bashCommand):
    print(bashCommand)
    try:
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output
    except:
        print("couldn't run bash command, try running it manually")


def join_video_and_audio(video_file, audio_file, trim_audio=0):
    video_file2 = video_file+"_music.mp4"
    audio_format = "mp3"
    new_audio_file = video_file+"."+audio_format
    bash_command = "ffprobe -v 0 -show_entries format=duration -of compact=p=0:nk=1 "+video_file
    duration = float(run_bash_command(bash_command))
    bash_command = "ffmpeg -y -i "+audio_file+" -ss "+str(trim_audio)+" -t "+str(duration)+" "+new_audio_file
    run_bash_command(bash_command)
    bash_command = "ffmpeg -y -i "+video_file+" -i "+new_audio_file+" "+video_file2
    run_bash_command(bash_command)



import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def render_mp4(keypoints, filename, fps=20):
    keypoints /= 5.
    keypoints[:, :, 0] += 0.5
    keypoints[:, :, 2] += 0.5

    fig = plt.figure()
    ax = Axes3D(fig)

    dots = []
    dots.append([ax.plot([], [], [], 'r.')[0] for _ in
                  range(keypoints.shape[1])])

    # 定义动画每一帧的更新函数
    def update(n):
        frame = keypoints[n]
        x = frame[:, 0]
        y = frame[:, 1]
        z = frame[:, 2]
        changed = []
        for i in range(keypoints.shape[1]):
            dots[0][i].set_data([x, z])
            dots[0][i].set_3d_properties(y)
            changed += dots
        return changed

    ani = animation.FuncAnimation(fig, update, np.arange(keypoints.shape[0]), interval=1000 / fps)

    if filename != None:
        ani.save(filename, fps=fps, bitrate=13934)
        ani.event_source.stop()
        del ani
        plt.close()
    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    import glob
    import tqdm
    from smplx import SMPL

    # get cached motion features for the real data
    real_features = {
        "kinetic": [np.load(f) for f in glob.glob("./data/aist_features/*_kinetic.npy")],
        "manual": [np.load(f) for f in glob.glob("./data/aist_features/*_manual.npy")],
    }

    # set smpl
    smpl = SMPL(model_path="./smpl/models", gender='MALE', batch_size=1)

    anno_dir = "../../aist_plusplus_final"
    save_dir = "./remote/vis_np_result_mag_small_data_55352/"
    # get motion features for the results
    result_features = {"kinetic": [], "manual": []}
    # result_files = glob.glob("../vis_scripts/np_datas/np_result_rawmusic_final_90440/*.npy")
    result_files = glob.glob("./remote/np_result_mag_small_data_55352/*.npy")
    # result_files = [f for f in result_files if f[-8:-4] in f[:-8]]
    beat_alignment_score = 0
    aist_dataset = AISTDataset(anno_dir)
    for result_file in tqdm.tqdm(result_files):
        result_file = result_file.replace("\\", "/")
        print(result_file)
        seq_name = "_".join(result_file.split("/")[-1].split(".")[-2].split("_")[-6:])
        print(seq_name)
        seq_name, view = AISTDataset.get_seq_name(seq_name)
        # view_idx = AISTDataset.VIEWS.index(view)
        result_motion = np.load(result_file)[None, ...]  # [1, 120 + 1200, 225]
        # visualize(result_motion, smpl)
        keypoints3d = recover_motion_to_keypoints(result_motion, smpl)


        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{seq_name}.mp4')
        keypoints3d = keypoints3d
        render_mp4(keypoints3d, save_path)

        # visualize(keypoints3d)
        # keypoints2d = keypoints3d[:, :, :2]
        # keypoints2d[:, :, 0] -= np.min(keypoints2d[:, :, 0])
        # keypoints2d[:, :, 1] -= np.min(keypoints2d[:, :, 1])
        # keypoints2d *= 40
        #
        # video = []
        # for iframe, keypoint in enumerate(keypoints2d):
        #     video.append(plot_kpt(keypoint, np.zeros((250, 400, 3), dtype="uint8")))
        # video = np.array(video)
        # # write video
        # ffmpeg_video_write(video, save_path, fps=20)
        # bind music
        # audio_path = "../datas/wav"
        # audio_names = seq_name.split("_")[-2]
        # audio_file = os.path.join(audio_path, audio_names + ".wav")
        # join_video_and_audio(save_path, audio_file)