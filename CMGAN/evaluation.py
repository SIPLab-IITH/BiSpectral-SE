import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import numpy as np
from models import generator_new
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt


@torch.no_grad()
def enhance_one_track(model, audio_path, clean_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False):
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len/cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True, return_complex=False)
    #noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft), onesided=True)
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag, return_complex=True).squeeze(1)
    est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True)
    #est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft), onesided=True)
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, clean_path.split('/')[-1])
        sf.write(saved_path, est_audio/max(abs(est_audio)), sr)

    return est_audio, length


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir):
    n_fft = 400
    print(model_path)
    model = generator_new.TSCNet(num_channel=64, num_features=n_fft//2+1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    #audio_list = os.listdir(noisy_dir)
    #audio_list = natsorted(audio_list)
    noisy_paths = glob.glob(noisy_dir+'/*.wav')
    clean_paths = glob.glob(clean_dir+'/*.wav')
    clean_paths.sort()
    noisy_paths.sort()#key=lambda x:x.split('/')[-1].split('_')[-1])
    num = len(noisy_paths)
    metrics_total = np.zeros(7)
    results_log_file = open('results_log_noisy_phase.txt', 'w')
    metric_list = []
    for noisy_path, clean_path in tqdm(zip(noisy_paths, clean_paths)):
        est_audio, length = enhance_one_track(model, noisy_path, clean_path, saved_dir, 16000*16, n_fft, n_fft//4, save_tracks)
        clean_audio, sr = sf.read(clean_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio/max(abs(clean_audio)), est_audio/max(abs(est_audio)), sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics
        results_log_file.write(noisy_path.split('/')[-1]+"\t"+"\t".join([str(metric) for metric in metrics])+"\n")
        metric_list.append(metrics)
        #break

    metric_list = np.array(metric_list)
    np.save('all_metrics_noisy_phase.npy', metric_list)
    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 'covl: ',
          metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='checkpoints_new_dataShuffle_2/CMGAN_epoch_119_0.054',#CMGAN_epoch_92_0.147',
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='/DATA/siplab/DNS-Challenge/ValentiniData/test/',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

args = parser.parse_args()


if __name__ == '__main__':
    noisy_dir = os.path.join(args.test_dir, 'noisy')
    clean_dir = os.path.join(args.test_dir, 'clean')
    evaluation(args.model_path, noisy_dir, clean_dir, args.save_tracks, args.save_dir)
