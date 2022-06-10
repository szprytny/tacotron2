import sys
import os
sys.path.append('waveglow')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser
from scipy.io.wavfile import write

hparams = create_hparams()
#hparams.max_wav_value=32768.0
hparams.sampling_rate = 22050
hparams.filter_length=1024
hparams.hop_length=256
hparams.win_length=1024

taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)

def line_to_text_sequence(line):
  text = line.replace('\n', '')
  sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
  return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

def text_sequence_to_mel_outputs(text_sequence):
  _, mel_outputs_postnet, _, _ = model.inference(text_sequence)

  mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
  mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
  spec_from_mel_scaling = 1000
  spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis.half())
  spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
  spec_from_mel = spec_from_mel * spec_from_mel_scaling

  return mel_outputs_postnet, spec_from_mel

def mel_to_audio(mels):
  mel, spec_from_mel = mels

  with torch.no_grad():
      audio = hparams.max_wav_value * waveglow.infer(mel, sigma=0.5)[0]
      audio = audio.cpu().numpy()
      audio = audio.astype('int16')
  
  waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, 60)[0]
  waveform = waveform.data.cpu().numpy()

  return audio, waveform

def save_audio_to_drive(audio, file_name):
  output_dir = 'out'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  print(f'saving {file_name}')
  audio_path = os.path.join(output_dir, file_name)
  write(audio_path, hparams.sampling_rate, audio)



model = load_model(hparams)
model.load_state_dict(torch.load('tacotron_shmart.pt')['state_dict'])
_ = model.cuda().eval().half()

waveglow = torch.load('waveglow_shmart.pt')['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()

denoiser = Denoiser(waveglow)

lines = open('sentences.txt', encoding="utf-8").readlines()
mels = list(map(text_sequence_to_mel_outputs, map(line_to_text_sequence, lines)))

for index, mel in enumerate(mels):
  audio_wg, audio_griffin = mel_to_audio(mel)
  save_audio_to_drive(audio_wg, f'{index}_wg.wav')
  save_audio_to_drive(audio_griffin,  f'{index}_griffinlim.wav')
