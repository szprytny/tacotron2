import argparse
import os
import torch
import json
import natsort 

from infer_helpers import get_griffin_audio, get_waveglow_audio, line_to_text_sequence, load_hifigan, mel_from_file, save_audio_to_drive, text_sequence_to_mel_outputs, hparams, device

from train import load_model
from hifigan_models import Generator, AttrDict
from waveglow_denoiser import Denoiser
import glow
  
waveglow = None
denoiser = None
model = load_model(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path',
                        help='Path to tacotron state dict', type=str, default='/outdir/models/tacotron/taco/yen.pt')
    parser.add_argument('-v', '--vocoder_path',
                        help='Path to vocoder state dict', type=str, default='/shmart/hifigan/cp_hifigan/g_latest')
    parser.add_argument('-t', '--text', help='Text to synthesize', type=str)
    parser.add_argument('-i', "--input_dir") #, default="input_mels/")
    parser.add_argument('--sentences', help='path to file with sentences to infer')
    parser.add_argument('-o', "--output_dir", default="/outdir/synth")
    parser.add_argument("-s", "--sigma", default=0.667, type=float)
    parser.add_argument('-w', "--vocoder", default='hifigan', type=str)
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
          waveglow.cuda().eval()
          for k in waveglow.convinv:
            k.float()
          denoiser = Denoiser(waveglow) 
          generate_waveglow = True
        elif vocoder == 'hifigan':
          with open('c:/shmart/hifigan/cp_hifigan/config.json') as f:
              data = f.read()

          json_config = AttrDict(json.loads(data))
          generator = Generator(json_config).to(device)

          state_dict_g = load_hifigan(args.vocoder_path)
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
          audio_waveglow = get_waveglow_audio(waveglow, denoiser, mel_output, args.sigma)
          save_audio_to_drive(audio_waveglow, f'{index}_waveglow_{args.sigma}.wav', args.output_dir)
          
        if generate_hifigan:
          y_g_hat = generator(mel_output)
          audio = y_g_hat.squeeze()
          audio = audio * hparams.max_wav_value
          audio = audio.cpu().numpy().astype('int16')
          save_audio_to_drive(audio, f'{index}_hifigan_{args.sigma}.wav', args.output_dir)
      
      if args.input_dir is not None:
        file_names = natsort.natsorted(os.listdir(args.input_dir))
        for i, file_name in enumerate(file_names):
          mel = mel_from_file(f'{os.path.join(args.input_dir, file_name)}')
          save_audios(mel, i)
      elif args.text is not None:
        text_sequence = line_to_text_sequence(args.text)
        mel = text_sequence_to_mel_outputs(model, text_sequence)
        save_audios(mel, 0)
      else:
        path = args.sentences or 'sentences.txt'
        lines = open(path, encoding="utf-8").readlines()
        for index, line in enumerate(lines):
          text_sequence = line_to_text_sequence(line)
          mel = text_sequence_to_mel_outputs(model, text_sequence)
          save_audios(mel, index)
      
          
