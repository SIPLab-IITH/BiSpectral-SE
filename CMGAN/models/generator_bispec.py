from models.conformer import ConformerBlock
from utils import *
from torchinfo import summary
import matplotlib.pyplot as plt

class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class Encoder(nn.Module):
    def __init__(self, n_fft=257, channels=64):
        super(Encoder, self).__init__()
        self.n_fft = n_fft
        self.inp_norm = nn.InstanceNorm2d(n_fft+4, affine=True)
        self.conv = nn.Sequential(
            nn.Conv2d(n_fft+4, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        row_index = torch.cumsum(torch.ones(size=(n_fft, n_fft), dtype=torch.int32), axis=-2)
        col_index = torch.cumsum(torch.ones(size=(n_fft, n_fft), dtype=torch.int32), axis=-1)
        #return (row_index - col_index) <= context
        self.mask = (row_index - col_index) >= 0

    def forward(self, x):
        complex_spec = torch.complex(x[:,0,:,:], x[:,1,:,:])
        mag = torch.abs(complex_spec).unsqueeze(1)
        f1_f2 = torch.matmul(complex_spec.unsqueeze(-1), complex_spec.unsqueeze(-2))
        f1_f2 = f1_f2 * self.mask.type(f1_f2.dtype).to(x.device)
        f1f2 = torch.cat([complex_spec[..., 1:], torch.zeros_like(complex_spec)], dim=-1).unfold(-1, complex_spec.shape[-1], 1)
        f1_f2_f1f2 = f1_f2 * torch.conj(f1f2)
        f1_f2_f1f2 = f1_f2_f1f2[..., :self.n_fft//2+1]
        f1_f2_f1f2 = f1_f2_f1f2.permute(0, 3, 1, 2)
        inp = torch.cat([mag, x, f1_f2_f1f2.real, f1_f2_f1f2.imag], dim=1)
        inp = self.inp_norm(inp)
        out = self.conv(inp)
        out = self.conv_2(out)
        return out
        

class TSCB(nn.Module):
    def __init__(self, num_channel=64, causal=False):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2, causal=causal)
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.causal = causal
        self.causal_mask = None
    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        if self.causal:
            self.causal_mask = (torch.triu(torch.ones(x_t.shape[1], x_t.shape[1]))==1).transpose(0,1)
            self.causal_mask = self.causal_mask.to(x_t.device)
        x_t = self.time_conformer(x_t, mask=self.causal_mask) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x


class TSCNet(nn.Module):
    def __init__(self, num_channel=64, num_features=201, causal=False):
        super(TSCNet, self).__init__()
        self.dense_encoder = Encoder(n_fft=num_features, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel, causal=causal)
        self.TSCB_2 = TSCB(num_channel=num_channel, causal=causal)
        self.TSCB_3 = TSCB(num_channel=num_channel, causal=causal)
        self.TSCB_4 = TSCB(num_channel=num_channel, causal=causal)

        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1)
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        #x_in = torch.cat([mag, x], dim=1)

        out_1 = self.dense_encoder(x)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        mask = self.mask_decoder(out_5)
        out_mag = mask * mag

        complex_out = self.complex_decoder(out_5)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag
        
if __name__=="__main__":
    summary(TSCNet(), (1, 2, 321, 201), device='cpu')
