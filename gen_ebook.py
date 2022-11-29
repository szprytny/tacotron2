import argparse
import os
import numpy as np
import torch

from pathlib import Path
from hparams import create_hparams
from layers import TacotronSTFT
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
from hifigan_denoiser import Denoiser
from hifigan_models import Generator, AttrDict
import json
from scipy.io.wavfile import write

def line_to_text_sequence(line):
  # speaker, text = line.split('|')
  # if speaker != '0':
  #   return None
  text = line
  sequence = np.array(text_to_sequence(text, ['shmart_cleaner']))[None, :]
  return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

def load_hifigan(filepath, device):
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

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

waveglow = None
denoiser = None
model = load_model(hparams)

def get_spec_from_mel(mel_outputs_postnet):
  mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
  mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
  spec_from_mel_scaling = 1000
  spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
  spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
  return spec_from_mel * spec_from_mel_scaling

def text_sequence_to_mel_outputs(text_sequence):
  _ = model.cuda().eval()
  _, mel_outputs_postnet, _, _ = model.inference(text_sequence)
  spec_from_mel = get_spec_from_mel(mel_outputs_postnet)
  return mel_outputs_postnet, spec_from_mel


def save_audio_to_drive(audio, file_name, output_dir):
  print(f'saving {file_name}')
  audio_path = os.path.join(output_dir, file_name)
  write(audio_path, hparams.sampling_rate, audio)

def mel_from_file(file_path):
  x = np.load(file_path)
  mel_output = torch.FloatTensor(x).to(device).half()
  return mel_output, get_spec_from_mel(mel_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path',
                        help='Path to tacotron state dict', type=str, default='/debug/checkpoint_4000')
    parser.add_argument('-v', '--vocoder_path',
                        help='Path to waveglow state dict', type=str, default='c:/shmart/hifigan/cp_hifigan/g_latest')
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-i', "--input_dir") #, default="input_mels/")
    parser.add_argument('-b', "--book", help='path to ebook', default="c:/model/tadeusz.txt")
    parser.add_argument('-o', "--output_dir", default="out/")
    parser.add_argument("-s", "--sigma", default=0.667, type=float)
    args = parser.parse_args()

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
      os.makedirs(args.output_dir)
      os.chmod(args.output_dir, 0o775)

    model.load_state_dict(torch.load(args.model_path)['state_dict'])

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    with torch.no_grad():
      with open('c:/shmart/hifigan/cp_hifigan/config.json') as f:
        data = f.read()

      json_config = AttrDict(json.loads(data))
      generator = Generator(json_config).to(device)

      state_dict_g = load_hifigan(args.vocoder_path, device)
      generator.load_state_dict(state_dict_g['generator'])

      generator.remove_weight_norm()
      denoiser = Denoiser(generator)
      generator.cuda()
      denoiser.cuda()
      generator.eval()
      denoiser.eval()
      
      path = args.book
      out_path = Path(args.output_dir)
      lines = open(path, encoding="utf-8").readlines()
      for index, line in enumerate(lines):
        wav_name = f'{index+1:04}.wav'
        if (out_path / wav_name).exists():
            continue
        text_sequence = line_to_text_sequence(line)

        if text_sequence is None: continue

        mel_output, _ = text_sequence_to_mel_outputs(text_sequence)
        audio = generator(mel_output).float()[0]
        audio_denoised = denoiser(
            audio, strength=0.006)[0].float()
            
        audio_denoised = audio_denoised[0].cpu().numpy()
        audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))
        save_audio_to_drive(audio_denoised, wav_name, args.output_dir)
      
          
