import os
import sys
import torch
from torch import nn
from utils import str2bool

class Trainer:
    def __init__(self, args, loader, model, optimizer, criterion, tester=None, scheduler=None, aug=None, target_convert=None):
        self.args = args
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.tester = tester
        self.aug = aug
        self.target_convert = target_convert
        self.max_norm = args.max_norm
        self.task = args.task
        if args.gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        model = model.to(self.device)
        self.model = model
        self.epoch = args.start_epoch
        self.iter = 0
        if args.load_from_chechpoint:
            self.load_from_checkpoint()

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
        parser.add_argument("--epoch", type=int, default=300)
        parser.add_argument("--save_every", type=int, default=8, help="set 1 to save every epoch")
        parser.add_argument("--print_every", type=int, default=100)
        parser.add_argument("--valid_every", type=int, default=-1, help="set -1 to valid every epoch")
        parser.add_argument("--save_path", type=str, default="./checkpoint_transformer_mag_nomirror")
        parser.add_argument("--load_from_chechpoint", type=str2bool, default=False)
        parser.add_argument("--task", type=str, default="seq_gen_style")
        parser.add_argument("--max_norm", type=str2bool, default=True)
        parser.add_argument("--checkpoint_path", type=str,
                            default=None)
        parser.add_argument("--start_epoch", type=int,
                            default=0)
        return parser

    def start_train(self):
        for _ in range(self.epoch, self.args.epoch):
            self.__epoch()

    def __epoch(self):
        loader = self.loader
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.criterion
        epoch = self.epoch
        print_every = self.args.print_every
        valid_every = self.args.valid_every
        save_every = self.args.save_every

        self.epoch += 1
        total_loss = {}

        for i, datas in enumerate(loader):
            self.iter += 1
            model.zero_grad()
            model_kwargs = {}
            if self.task == "seq_gen":  # circle forward
                inputs = datas[0].to(self.device)
                targets = datas[1].to(self.device)
                model_kwargs["epoch"] = self.epoch
            elif self.task == "seq_gen_style":
                inputs = datas[0].to(self.device)
                targets = datas[1].to(self.device)
                styles = datas[2].to(self.device)
                model_kwargs["epoch"] = self.epoch
                model_kwargs["styles"] = styles

            if self.aug is not None:
                inputs = self.aug.run(inputs, self.iter)

            if self.target_convert is not None:
                targets = self.target_convert.run(targets)

            output = model(inputs, targets=targets, iter=self.iter, **model_kwargs)
            loss_dict, loss = criterion.compute(output, targets, **model_kwargs)

            loss.backward()
            if self.max_norm:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

            optimizer.step()

            for k, v in loss_dict.items():
                if k not in total_loss.keys():
                    total_loss[k] = 0.
                total_loss[k] += v

            if (i + 1) % print_every == 0:
                lr = optimizer.param_groups[0]["lr"]

                loss_str = f""
                for k, v in total_loss.items():
                    loss_str += f"{k}: {v / print_every:.5f};\t"
                total_loss = {}

                print(
                    f"epoch: {epoch};\t iter: {i};\t lr: {lr:.5f};\t" + loss_str
                )
            if valid_every > 0 and self.iter % valid_every == 0 and self.tester is not None:
                self.tester.test()

        if scheduler is not None:
            scheduler.step()

        if valid_every == -1 and self.tester is not None:
            self.tester.test()
        if self.epoch % save_every == 0:
            self.save(self.iter, self.args.save_path)


    def save(self, i, path):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.model.state_dict(), f"{path}/checkpoint_{str(i).zfill(3)}.pt")

    def load_from_checkpoint(self):
        if self.args.checkpoint_path is not None:
            self.model.load_state_dict(torch.load(self.args.checkpoint_path), strict=True)
