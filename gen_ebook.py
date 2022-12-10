import argparse
import os
import numpy as np
import torch

from pathlib import Path
from infer_helpers import line_to_text_sequence, load_hifigan, save_audio_to_drive, text_sequence_to_mel_outputs, hparams, device
from train import load_model
from text import text_to_sequence
from hifigan_denoiser import Denoiser
from hifigan_models import Generator, AttrDict
import json

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

denoiser = None
model = load_model(hparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path',
                        help='Path to tacotron state dict', type=str, default='/users/user/downloads/geralt.pt')
    parser.add_argument('-v', '--vocoder_path',
                        help='Path to vocoder state dict', type=str, default='/shmart/hifigan/cp_hifigan/g_latest')
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-i', "--input_dir") #, default="input_mels/")
    parser.add_argument('-b', "--book", help='path to ebook', default="/model/bokp.txt")
    parser.add_argument('-o', "--output_dir", default="/outdir/bokp")
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

        mel_output, _ = text_sequence_to_mel_outputs(model, text_sequence)
        audio = generator(mel_output).float()[0]
        audio_denoised = denoiser(
            audio, strength=0.006)[0].float()
            
        audio_denoised = audio_denoised[0].cpu().numpy()
        audio_denoised = audio_denoised / np.max(np.abs(audio_denoised))
        save_audio_to_drive(audio_denoised, wav_name, args.output_dir)
      
          
