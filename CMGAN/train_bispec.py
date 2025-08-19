import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from models.generator_bispec import TSCNet
from models import discriminator
from data import dataloader
import torch.nn.functional as F
from utils import *
import logging
from torchinfo import summary
import argparse
import timeit

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from data.dataloader import DemandDataset
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='/data_directory/',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='checkpoints_bispec',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
#parser.add_argument("--loss_weights", type=list, default=[0.3, 0.7, 1.0, 0.01],
#                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12528"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(self, gpu_id: int, args=None):
        self.n_fft = 400
        self.hop = 100
        self.args = args
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1, causal=False).cuda()
        summary(self.model, [(1, 2, args.cut_len//self.hop+1, int(self.n_fft/2)+1)])
        self.discriminator = discriminator.Discriminator(ndf=16).cuda()
        summary(self.discriminator, [(1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1),
                                     (1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1)])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=2*args.init_lr)

        train_data = DemandDataset(os.path.join(args.data_dir, 'train'), args.cut_len)
        test_data = DemandDataset(os.path.join(args.data_dir, 'test'), args.cut_len)

        self.train_sampler = DistributedSampler(train_data, shuffle=True)
        self.test_sampler = DistributedSampler(test_data)

        self.train_ds = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, pin_memory=True, sampler=self.train_sampler, drop_last=True, num_workers=5)
        self.test_ds = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, pin_memory=True, sampler=self.test_sampler, drop_last=False, num_workers=5)

        self.model = DDP(self.model, device_ids=[gpu_id])
        self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id

    def train_step(self, batch):
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(clean.shape[0]).to(self.gpu_id)

        # Normalization
        c = torch.sqrt(noisy.size(-1) / (torch.sum((noisy ** 2.0), dim=-1)+1e-5))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        self.optimizer.zero_grad()
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).to(self.gpu_id),
                                onesided=True, return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).to(self.gpu_id),
                                onesided=True, return_complex=True)
        noisy_spec = torch.stack([noisy_spec.real, noisy_spec.imag], axis=-1)
        clean_spec = torch.stack([clean_spec.real, clean_spec.imag], axis=-1)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2 + 1e-7)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2 + 1e-7)

        predict_fake_metric = self.discriminator(clean_mag, est_mag)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

        est_spec_uncompress = power_uncompress(est_real, est_imag, return_complex=True).squeeze(1)
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).to(self.gpu_id), onesided=True)

        time_loss = torch.mean(torch.abs(est_audio - clean))
        length = est_audio.size(-1)
        loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
               + args.loss_weights[3] * gen_loss_GAN
        loss.backward()
        #torch.nn.utils.clip_grad_value_(self.model.parameters(), 2.0)
        self.optimizer.step()

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
            discrim_loss_metric.backward()
            #torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), 2.0)
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(clean.shape[0]).to(self.gpu_id)

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).to(self.gpu_id),
                                onesided=True, return_complex=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).to(self.gpu_id),
                                onesided=True, return_complex=True)
        noisy_spec = torch.stack([noisy_spec.real, noisy_spec.imag], axis=-1)
        clean_spec = torch.stack([clean_spec.real, clean_spec.imag], axis=-1)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2 + 1e-7)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

        predict_fake_metric = self.discriminator(clean_mag, est_mag)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)

        est_spec_uncompress = power_uncompress(est_real, est_imag, return_complex=True).squeeze(1)
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).to(self.gpu_id), onesided=True)

        time_loss = torch.mean(torch.abs(est_audio - clean))
        length = est_audio.size(-1)
        loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
               + args.loss_weights[3] * gen_loss_GAN

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = torch.tensor([0.])

        return loss.item(), discrim_loss_metric.item()

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = 'GPU: {}, Generator loss: {}, Discriminator loss: {}'
        logging.info(
            template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_epoch, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5)
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(args.epochs):
            beg_t = timeit.default_timer()
            #self.train_ds, self.test_ds = dataloader.load_data(args.data_dir, args.batch_size, 2, args.cut_len)
            self.model.train()
            self.discriminator.train()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss = self.train_step(batch)
                template = 'GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}'
                if (step % args.log_interval) == 0:
                    logging.info(template.format(self.gpu_id, epoch, step, loss, disc_loss))
            gen_loss = self.test()
            path = os.path.join(args.save_model_dir, 'CMGAN_epoch_' + str(epoch) + '_' + str(gen_loss)[:5])
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            if self.gpu_id == 0:
                torch.save(self.model.module.state_dict(), path)
            end_t = timeit.default_timer()
            print('Time Taken: {:.2f}'.format(end_t-beg_t))
            scheduler_G.step()
            scheduler_D.step()


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    if rank == 0:
        print(args)
        available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(available_gpus)
    trainer = Trainer(rank, args)
    trainer.train()
    destroy_process_group()


if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
