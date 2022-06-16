import sys
import argparse
import os
sys.path.append('waveglow')
import numpy as np
import torch

from hparams import create_hparams
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write

hparams = create_hparams()
#hparams.max_wav_value=32768.0
hparams.sampling_rate = 22050
hparams.filter_length=1024
hparams.hop_length=256
hparams.win_length=1024

if torch.cuda.is_available():
  torch.cuda.manual_seed(42)
  device = torch.device('cuda')
else:
  device = torch.device('cpu')


taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)

waveglow = torch.load('waveglow_shmart.pt')['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
  k.float()

def line_to_text_sequence(line):
  text = line.replace('\n', '')
  sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
  return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

def get_spec_from_mel(mel_outputs_postnet):
  mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
  mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
  spec_from_mel_scaling = 1000
  spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis.half())
  spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
  return spec_from_mel * spec_from_mel_scaling

def text_sequence_to_mel_outputs(text_sequence):
  model = load_model(hparams)
  model.load_state_dict(torch.load('tacotron_shmart.pt')['state_dict'])
  _ = model.cuda().eval().half()
  _, mel_outputs_postnet, _, _ = model.inference(text_sequence)
  spec_from_mel = get_spec_from_mel(mel_outputs_postnet)
  return mel_outputs_postnet, spec_from_mel

def get_griffin_audio(spec):
  waveform = griffin_lim(torch.autograd.Variable(spec[:, :, :-1]), taco_stft.stft_fn, 60)[0]
  return waveform.data.cpu().numpy()

def get_waveglow_audio(mel, sigma=0.5):
  with torch.no_grad():
    audio = hparams.max_wav_value * waveglow.infer(mel, sigma)[0]
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
  return audio

def save_audio_to_drive(audio, file_name):
  output_dir = 'out'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  print(f'saving {file_name}')
  audio_path = os.path.join(output_dir, file_name)
  write(audio_path, hparams.sampling_rate, audio)

def mel_from_file(file_path):
  x = np.load(file_path)
  mel_output = torch.FloatTensor(x).to(device).half()
  return mel_output, get_spec_from_mel(mel_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str, default='waveglow_shmart.pt')
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-i', "--input_dir") #, default="input_mels/")
    parser.add_argument('-o', "--output_dir", default="out/")
    parser.add_argument("-s", "--sigma", default=0.5, type=float)
    parser.add_argument('-v', "--vocoder", default='waveglow', type=str)
    args = parser.parse_args()

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    if args.input_dir is not None:
      file_names = os.listdir(args.input_dir)
      mels = [mel_from_file(f'{os.path.join(args.input_dir, file_name)}') for i, file_name in enumerate(file_names)]
    elif args.text is not None:
      mels = list(map(text_sequence_to_mel_outputs, map(line_to_text_sequence, [args.text])))
    else:
      lines = open('sentences.txt', encoding="utf-8").readlines()
      mels = list(map(text_sequence_to_mel_outputs, map(line_to_text_sequence, lines)))

    generate_griffin = False
    generate_waveglow = False

    for _, vocoder in enumerate(args.vocoder.split(',')):
      if vocoder == 'griffinlim':
        generate_griffin = True
      elif vocoder == 'waveglow':
        generate_waveglow = True

    if generate_griffin == True or generate_waveglow == True:
      for index, mel in enumerate(mels):
        mel_output, spec_from_mel = mel

        if generate_griffin:
          audio_griffin = get_griffin_audio(spec_from_mel)
          save_audio_to_drive(audio_griffin,  f'{index}_griffinlim.wav') 
        
        if generate_waveglow:
          audio_waveglow = get_waveglow_audio(mel_output, args.sigma)
          save_audio_to_drive(audio_waveglow, f'{index}_waveglow_{args.sigma}.wav')
