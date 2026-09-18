from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append("..")
import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet
import soundfile as sf
from rich.progress import track

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def enhance_chunk(model, noisy_wav):
    """Enhance one normalized waveform chunk and return it on the CPU."""
    noisy_wav = noisy_wav.unsqueeze(0).to(device)
    noisy_amp, noisy_pha, _ = mag_pha_stft(
        noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    amp_g, pha_g, _ = model(noisy_amp, noisy_pha)
    audio_g = mag_pha_istft(
        amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    return audio_g.squeeze(0).cpu()

def enhance_in_chunks(model, noisy_wav, chunk_size, chunk_overlap):
    """Enhance a long waveform without placing the whole utterance on the GPU."""
    if chunk_size <= 0:
        return enhance_chunk(model, noisy_wav)
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError('--chunk_overlap must be non-negative and smaller than --chunk_size.')

    num_samples = noisy_wav.numel()
    enhanced = torch.zeros_like(noisy_wav)
    weights = torch.zeros_like(noisy_wav)
    step = chunk_size - chunk_overlap
    for start in range(0, num_samples, step):
        end = min(start + chunk_size, num_samples)
        chunk_length = end - start
        enhanced_chunk = enhance_chunk(model, noisy_wav[start:end])[:chunk_length]
        if enhanced_chunk.numel() < chunk_length:
            enhanced_chunk = torch.nn.functional.pad(
                enhanced_chunk, (0, chunk_length - enhanced_chunk.numel()))

        window = torch.ones(chunk_length, dtype=noisy_wav.dtype)
        fade_length = min(chunk_overlap, chunk_length)
        if start > 0 and fade_length > 0:
            window[:fade_length] = torch.linspace(0, 1, fade_length, dtype=noisy_wav.dtype)
        if end < num_samples and fade_length > 0:
            window[-fade_length:] = torch.linspace(1, 0, fade_length, dtype=noisy_wav.dtype)
        enhanced[start:end] += enhanced_chunk * window
        weights[start:end] += window
        if end == num_samples:
            break
    return enhanced / weights.clamp_min(1e-8)

def inference(a):
    model = MPNet(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    test_indexes = os.listdir(a.input_noisy_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for index in track(test_indexes):
            noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, index), sr=h.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0).clamp_min(1e-8))
            noisy_wav = noisy_wav * norm_factor
            audio_g = enhance_in_chunks(model, noisy_wav, a.chunk_size, a.chunk_overlap)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, index)

            sf.write(output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/testset_noisy')
    parser.add_argument('--output_dir', default='../generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Samples per inference chunk. Default: config segment_size.')
    parser.add_argument('--chunk_overlap', type=int, default=None,
                        help='Overlap in samples between chunks. Default: 10%% of chunk_size.')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    if a.chunk_size is None:
        a.chunk_size = h.segment_size
    if a.chunk_overlap is None:
        a.chunk_overlap = a.chunk_size // 10
    print('Chunked inference: {} samples with {}-sample overlap.'.format(
        a.chunk_size, a.chunk_overlap))

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
