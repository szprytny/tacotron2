import sys
import argparse
import os
sys.path.append('waveglow')
import numpy as np
import torch
import natsort 

from hparams import create_hparams
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
from denoiser import Denoiser
from hifigan_models import Generator, AttrDict
import json
from scipy.io.wavfile import write

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

def text_sequence_to_mel_outputs(text_sequence):
  _ = model.cuda().eval()
  _, mel_outputs_postnet, _, _ = model.inference(text_sequence)
  spec_from_mel = get_spec_from_mel(mel_outputs_postnet)
  return mel_outputs_postnet, spec_from_mel

def get_griffin_audio(spec):
  waveform = griffin_lim(torch.autograd.Variable(spec[:, :, :-1]), taco_stft.stft_fn, 60)[0]
  return waveform.data.cpu().numpy()

def get_waveglow_audio(mel, sigma=0.5):
  with torch.no_grad():
    audio = waveglow.infer(mel, sigma)
    audio = denoiser(audio, strength=0.01)
    audio = audio * hparams.max_wav_value
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
  mel_output = torch.FloatTensor(x).to(device).half()
  return mel_output, get_spec_from_mel(mel_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path',
                        help='Path to tacotron state dict', type=str, default='tacotron_shmart.pt')
    parser.add_argument('-w', '--vocoder_path',
                        help='Path to waveglow state dict', type=str, default='waveglow_shmart.pt')
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-i', "--input_dir") #, default="input_mels/")
    parser.add_argument('--sentences', help='path to file with sentences to infer')
    parser.add_argument('-o', "--output_dir", default="out/")
    parser.add_argument("-s", "--sigma", default=0.5, type=float)
    parser.add_argument('-v', "--vocoder", default='waveglow', type=str)
    args = parser.parse_args()

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
      os.makedirs(args.output_dir)
      os.chmod(args.output_dir, 0o775)

    model.load_state_dict(torch.load(args.model_path)['state_dict'])

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    generate_griffin = False
    generate_waveglow = False
    generate_hifigan = False

    with torch.no_grad():
      for _, vocoder in enumerate(args.vocoder.split(',')):
        if vocoder == 'griffinlim':
          generate_griffin = True
        elif vocoder == 'waveglow':
          waveglow = torch.load(args.vocoder_path)['model']
          waveglow.cuda().eval().half()
          for k in waveglow.convinv:
            k.float()
          denoiser = Denoiser(waveglow) 
          generate_waveglow = True
        elif vocoder == 'hifigan':
          with open('c:/shmart/hifigan/cp_hifigan/config.json') as f:
              data = f.read()

          json_config = AttrDict(json.loads(data))
          generator = Generator(json_config).to(device)

          state_dict_g = load_hifigan(args.vocoder_path, device)
          generator.load_state_dict(state_dict_g['generator'])

          generator.eval()
          generator.remove_weight_norm()
          generate_hifigan = True

      if generate_griffin == False and generate_waveglow == False and generate_hifigan == False:
        exit()

      def save_audios(mel, index):
        mel_output, spec_from_mel = mel
        if generate_griffin:
          audio_griffin = get_griffin_audio(spec_from_mel)
          save_audio_to_drive(audio_griffin,  f'{index}_griffinlim.wav', args.output_dir) 
        
        if generate_waveglow:
          audio_waveglow = get_waveglow_audio(mel_output, args.sigma)
          save_audio_to_drive(audio_waveglow, f'{index}_waveglow_{args.sigma}.wav', args.output_dir)
          
        if generate_hifigan:
          y_g_hat = generator(mel_output)
          audio = y_g_hat.squeeze()
          audio = audio * 32768
          audio = audio.cpu().numpy().astype('int16')
          save_audio_to_drive(audio, f'{index}_hifigan_{args.sigma}.wav', args.output_dir)
      
      if args.input_dir is not None:
        file_names = natsort.natsorted(os.listdir(args.input_dir))
        for i, file_name in enumerate(file_names):
          mel = mel_from_file(f'{os.path.join(args.input_dir, file_name)}')
          save_audios(mel, i)
      elif args.text is not None:
        text_sequence = line_to_text_sequence(args.text)
        mel = text_sequence_to_mel_outputs(text_sequence)
        save_audios(mel, 0)
      else:
        path = args.sentences or 'sentences.txt'
        lines = open(path, encoding="utf-8").readlines()
        for index, line in enumerate(lines):
          text_sequence = line_to_text_sequence(line)
          mel = text_sequence_to_mel_outputs(text_sequence)
          save_audios(mel, index)
      
          
