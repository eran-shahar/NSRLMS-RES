import torch
import torch.nn as nn
import time
from torchaudio import transforms as T


class Encoder(nn.Module):

    def __init__(self, in_channels=1, out_channels=2,
                 kernel_size=(5, 3), padding=(2, 1)):

        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):

        return self.encoder(x)



class GRUNet(nn.Module):

    def __init__(
            self,
            sr=16000,
            gru_layers=2,
            hidden_size=128,
            win_len=0.02,  # Seconds
            win_inc=0.01,  # Seconds
            fft_len=320,
            win_type='hanning',
            in_channels = 4,
            encoder_dim = 2):

        super(GRUNet, self).__init__()

        self.sr = sr
        self.win_len = int(sr*win_len)
        self.win_inc = int(sr*win_inc)
        self.fft_len = fft_len
        self.win_type = win_type
        self.in_channels = in_channels
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.encoder_dim = encoder_dim

        self.stft = T.Spectrogram(n_fft=self.fft_len,
                                  win_length=self.win_len,
                                  hop_length=self.win_inc,
                                  power=None,
                                  return_complex=True)

        encoders = []

        for channels in range(self.in_channels):
            encoders.append(Encoder(out_channels=encoder_dim))

        self.encoders = nn.ModuleList(encoders)

        self.GRU = nn.GRU(
                input_size= self.in_channels*encoder_dim*(self.fft_len//2 + 1),
                hidden_size=hidden_size,
                num_layers=gru_layers,
                dropout=0.2,
                bidirectional=False,
                batch_first=True
            )

        self.transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.encoder_dim*(self.fft_len//2 + 1)),
            nn.ReLU())

        self.masker = nn.Sequential(
            nn.Conv2d(in_channels=self.encoder_dim*(self.in_channels+1),
                      out_channels=1,
                      kernel_size=(3,3), padding=(1, 1)),
            nn.Sigmoid()
        )


    def forward(self, inputs):

        in_specs = self.stft(inputs)

        error_mags = torch.abs(in_specs[:, 2, ...])
        error_phase = torch.angle(in_specs[:, 2, ...])
        
        in_mags = torch.abs(in_specs)

        out = torch.log(in_mags + 1e-8)

        out_lst = []

        for channel in range(self.in_channels):

            curr_out = out[:, channel, ...].unsqueeze(1)
            curr_out = self.encoders[channel](curr_out)
            out_lst.append(curr_out)

        out = torch.cat(out_lst, 1)

        batch, channels, freq, t_steps = out.size()
        out = out.permute(0, 3, 1, 2)
        out = torch.reshape(out, [batch, t_steps, channels * freq])

        out, _ = self.GRU(out)

        out = self.transform(out)

        out = torch.reshape(out, [batch, t_steps, self.encoder_dim,
                                  (self.fft_len//2 + 1)])
        out = out.permute(0, 2, 3, 1)
        out = torch.cat([out] + out_lst, 1)

        mask = self.masker(out).squeeze(1)

        out_mags = mask*error_mags
        
        real = out_mags*torch.cos(error_phase)
        imag = out_mags*torch.sin(error_phase)

        out_spec = torch.view_as_complex(torch.stack([real, imag], -1))

        nearend_wav = torch.istft(out_spec,
                              n_fft=self.fft_len,
                              hop_length=self.win_inc,
                              win_length=self.win_len,
                              window=torch.hann_window(self.win_len).to(inputs.device))

        nearend_wav = torch.squeeze(nearend_wav, 1)
        nearend_wav = torch.clamp_(nearend_wav, -1, 1)

        return nearend_wav


if __name__ == "__main__":


    win_len = 0.02
    stft_num = 320
    win_inc = 0.01

    sr = 16000
    in_channels = 4

    gru_layers = 2
    hidden_size = 128


    model = GRUNet(gru_layers=gru_layers, hidden_size=hidden_size,
                     win_len=win_len, win_inc=win_inc, fft_len=stft_num,
                     in_channels = in_channels)

    print('\nNumber of parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    
    x = torch.zeros((8, in_channels, 4*sr))

    out = model(x)
