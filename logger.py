import random
import torch
from torch.utils.tensorboard import SummaryWriter
from infer_helpers import get_griffin_audio, get_spec_from_mel, line_to_text_sequence, text_sequence_to_mel_outputs
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')

        spec_from_mel = get_spec_from_mel(mel_outputs)
        self.add_audio('audio_validation', get_griffin_audio(spec_from_mel), iteration, 22050)
        _, spec_from_mel = text_sequence_to_mel_outputs(model, line_to_text_sequence('W krzakach rzekł do trznadla trznadel: – Możesz mi pożyczyć szpadel? Muszę nim przetrzebić chaszcze, bo w nich straszą straszne paszcze. Odrzekł na to drugi trznadel: – Niepotrzebny, trznadlu, szpadel! Gdy wytrzeszczysz oczy w chaszczach, z krzykiem pierzchnie każda paszcza!'))
        self.add_audio('audio_test', get_griffin_audio(spec_from_mel), iteration, 22050)