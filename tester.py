import os
import sys
import torch
import numpy as np


class Tester:
    def __init__(self, args, loader, model, criterion, target_convert=None, data_aug=None):
        self.args = args
        self.loader = loader
        self.criterion = criterion
        self.task = args.task
        self.target_convert = target_convert
        if args.gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        model = model.to(self.device)
        self.model = model
        self.iter = 0
        self.aug = data_aug

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--gpu", type=bool, default=True)
        parser.add_argument("--n_gpu", type=int, default=1)
        port = (
                2 ** 15
                + 2 ** 14
                + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        )
        parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
        parser.add_argument("--print_every", type=int, default=50)
        parser.add_argument("--checkpoint_path", type=str, default="")
        parser.add_argument("--output_save_path", type=str, default="./outputs_posevq_2.12")
        parser.add_argument("--save_sample_num", type=int, default=100)
        parser.add_argument("--task", type=str, default="pose3d_vqvae_with_music")
        return parser

    def test(self):
        loader = self.loader
        model = self.model
        criterion = self.criterion
        total_loss = {}
        total_loss["cnt"] = 0
        total_loss["acc"] = 0
        iter_num = 0

        model.eval()
        with torch.no_grad():
            for i, datas in enumerate(loader):
                iter_num += 1
                model.zero_grad()
                model_kwargs = {}
                if self.task == "seq_gen":  # circle forward
                    inputs = datas[0].to(self.device)
                    targets = datas[1].to(self.device)
                    # model_kwargs["epoch"] = self.epoch
                elif self.task == "seq_gen_style":
                    inputs = datas[0].to(self.device)
                    targets = datas[1].to(self.device)
                    styles = datas[2].to(self.device)
                    # model_kwargs["epoch"] = self.epoch
                    model_kwargs["styles"] = styles
                elif self.task == "seq_gen_mirror":
                    inputs = datas[0].to(self.device)
                    targets = datas[1].to(self.device)
                    targets_mirror = datas[2].to(self.device)
                    # model_kwargs["epoch"] = self.epoch
                    model_kwargs["targets_mirror"] = targets_mirror
                elif self.task == "seq_gen_mirror_style":
                    inputs = datas[0].to(self.device)
                    targets = datas[1].to(self.device)
                    targets_mirror = datas[2].to(self.device)
                    styles = datas[3].to(self.device)
                    # model_kwargs["epoch"] = self.epoch
                    model_kwargs["targets_mirror"] = targets_mirror
                    model_kwargs["styles"] = styles

                if self.aug is not None:
                    inputs = self.aug.run(inputs, self.iter)

                if self.target_convert is not None:
                    targets = self.target_convert.run(targets)

                output = model(inputs, targets=targets, iter=self.iter, **model_kwargs)
                loss_dict, loss = criterion.compute(output, targets, **model_kwargs)

                for k, v in loss_dict.items():
                    if k not in total_loss.keys():
                        total_loss[k] = 0.
                    total_loss[k] += v
                total_loss["cnt"] += 1

        model.train()
        self.print_loss(iter_num, total_loss)

    def save(self, i, path):
        torch.save(self.model.state_dict(), f"{path}/vqvae_{str(i).zfill(3)}.pt")

    def print_loss(self, i, total_loss):
        loss_str = f""
        for k, v in total_loss.items():
            if k == "cnt":
                continue
            loss_str += f"{k}: {v/total_loss['cnt']:.5f};\t"

        print(
            f"valid\t iter:{i};\t" + loss_str
        )

    def inference(self):
        loader = self.loader
        model = self.model
        criterion = self.criterion
        total_loss = {}
        total_loss["cnt"] = 0
        iter_num = 0
        sample_num = 0

        model.eval()
        with torch.no_grad():
            for i, datas in enumerate(loader):
                iter_num += 1
                model.zero_grad()
                model_kwargs = {}
                if self.task == "auto_encode":
                    inputs = datas.to(self.device)
                    targets = inputs.clone()
                elif self.task == "translate":
                    inputs = datas[0].to(self.device)
                    targets = datas[1].to(self.device)
                elif self.task == "seq_gen":
                    if len(datas) == 3:
                        inputs = datas[0].to(self.device)
                        dct = datas[1].to(self.device)
                        targets = datas[2].to(self.device)
                        model_kwargs["dct"] = dct
                    elif len(datas) == 2:
                        inputs = datas[0].to(self.device)
                        targets = datas[1].to(self.device)
                elif self.task == "seq_pred":
                    inputs = None
                    dct = datas[0].to(self.device)
                    model_kwargs["dct"] = dct
                    targets = datas[1].to(self.device)
                elif self.task == "seq_gen_random_hint":
                    inputs = datas[0].to(self.device)
                    targets = datas[1].to(self.device)
                    hints = datas[2].to(self.device)
                    model_kwargs["hints"] = hints
                elif self.task == "pose3d_vqvae_with_music":
                    targets = datas[0].to(self.device)  # pose
                    inputs = datas[1].to(self.device)  # music
                    idxes = datas[2].to(self.device)
                    model_kwargs["idx"] = idxes

                if self.aug is not None:
                    inputs = self.aug.run(inputs, self.iter)

                if self.target_convert is not None:
                    targets = self.target_convert.run(targets)

                output = model(inputs, targets=targets, iter=self.iter, **model_kwargs)
                loss_dict, loss = criterion.compute(output, targets)

                pred = output["pred"].cpu().detach().numpy()
                gt = targets.cpu().detach().numpy()

                for k, v in loss_dict.items():
                    if k not in total_loss.keys():
                        total_loss[k] = 0.
                    total_loss[k] += v
                total_loss["cnt"] += 1

                for idx in range(pred.shape[0]):
                    sample_num += 1
                    if not os.path.exists(self.args.output_save_path):
                        os.mkdir(self.args.output_save_path)

                    np.save(os.path.join(self.args.output_save_path, f"outputres_{sample_num}"),
                            pred[idx])

                    np.save(os.path.join(self.args.output_save_path, f"outputres_gt_{sample_num}"),
                            gt[idx])

                    if sample_num >= self.args.save_sample_num:
                        break

                if sample_num >= self.args.save_sample_num:
                    break

        model.train()
        self.print_loss(iter_num, total_loss)

    def load_from_checkpoint(self):
        self.model.load_state_dict(torch.load(self.args.checkpoint_path), strict=False)