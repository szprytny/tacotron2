import numpy as np
import torch
import os

from hparams import create_hparams
from layers import TacotronSTFT
from text import text_to_sequence
from scipy.io.wavfile import write
from audio_processing import griffin_lim

hparams = create_hparams()
hparams.max_wav_value=32768.0
hparams.sampling_rate = 22050
hparams.filter_length=1024
hparams.hop_length=256
hparams.win_length=1024

taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)

if torch.cuda.is_available():
  torch.cuda.manual_seed(42)
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

def line_to_text_sequence(line):
  text = line.replace('\n', '')
  sequence = np.array(text_to_sequence(text, ['shmart_cleaner']))[None, :]
  return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

def get_spec_from_mel(mel_outputs_postnet):
  mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
  mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
  spec_from_mel_scaling = 1000
  spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
  spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
  return spec_from_mel * spec_from_mel_scaling

def text_sequence_to_mel_outputs(model, text_sequence):
  model.cuda().eval()
  _, mel_outputs_postnet, _, _ = model.inference(text_sequence)
  spec_from_mel = get_spec_from_mel(mel_outputs_postnet)
  return mel_outputs_postnet, spec_from_mel

def get_griffin_audio(spec):
  waveform = griffin_lim(torch.autograd.Variable(spec[:, :, :-1]), taco_stft.stft_fn, 60)[0]
  return waveform.data.cpu().numpy()

def get_waveglow_audio(waveglow, denoiser, mel, sigma=0.5, max_wav_value=32768.0):
  with torch.no_grad():
    audio = waveglow.infer(mel, sigma)
    audio = denoiser(audio, strength=0.01)
    audio = audio * max_wav_value
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
  return audio

def save_audio_to_drive(audio, file_name, output_dir):
  print(f'saving {file_name}')
  audio_path = os.path.join(output_dir, file_name)
  write(audio_path, hparams.sampling_rate, audio)

def mel_from_file(file_path):
  x = np.load(file_path)
  mel_output = torch.FloatTensor(x).to(device)
  return mel_output, get_spec_from_mel(mel_output)

def load_hifigan(filepath):
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict