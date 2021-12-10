import functools

import torch
import torch.nn as nn


class GANLoss(nn.Module):

    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        assert gan_mode in ['vanilla', 'lsgan', 'wgangp'], f'Unknown GAN mode: {gan_mode}'
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError

    def get_target_tensor(self, predicts, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(predicts)

    def __call__(self, predicts, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(predicts, target_is_real)
            loss = self.loss(predicts, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -predicts.mean()
            else:
                loss = predicts.mean()
        return loss


class Identity(nn.Module):

    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError(f'Unknown norm layer {norm_type}')
    return


class Encoder(nn.Module):

    def __init__(
        self,
        inp_channels=3,
        filter_base=64,
        norm_layer=nn.BatchNorm2d,
    ):
        super(Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.encoder = nn.Sequential(
            nn.Conv2d(inp_channels, filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_base, 2 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(2 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * filter_base, 4 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(4 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * filter_base, 8 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * filter_base, 8 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * filter_base, 4 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(4 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * filter_base, filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(filter_base),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image):
        out = self.encoder(image)
        return out


class Decoder(nn.Module):

    def __init__(
        self,
        oup_channels=1,
        filter_base=64,
        norm_layer=nn.BatchNorm2d,
    ):
        super(Decoder, self).__init__()
        label_dim = 13
        noise_dim = 100

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        inp_channels = label_dim + noise_dim + filter_base
        embed_oup_dim = label_dim + noise_dim + filter_base * 4

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(inp_channels, embed_oup_dim, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(embed_oup_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_oup_dim, 8 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8 * filter_base, 8 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8 * filter_base, 4 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(4 * filter_base),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4 * filter_base, 2 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(2 * filter_base),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * filter_base, filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(filter_base),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(filter_base, oup_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, feats, label, noise):
        label_embed = label.unsqueeze(2).unsqueeze(3)
        noise_embed = noise.unsqueeze(2).unsqueeze(3)
        out = torch.cat([feats, label_embed, noise_embed], 1)
        out = self.decode(out)
        return out


class Decider(nn.Module):

    def __init__(
        self,
        oup_channels=1,
        filter_base=64,
        norm_layer=nn.BatchNorm2d,
    ):
        super(Decider, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.image_discriminator = nn.Sequential(
            nn.Conv2d(3, filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_base, 2 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(2 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * filter_base, 4 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(4 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * filter_base, 8 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * filter_base, 8 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.audio_discriminator = nn.Sequential(
            nn.Conv2d(1, filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filter_base, 2 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(2 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * filter_base, 4 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(4 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * filter_base, 8 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * filter_base, 8 * filter_base, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final_discriminator = nn.Sequential(
            nn.Conv2d(16 * filter_base, 8 * filter_base, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(8 * filter_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 4 * 8 * filter_base, oup_channels),
        )

    def forward(self, image, audio):
        image_out = self.image_discriminator(image)
        audio_out = self.audio_discriminator(audio)
        fused_out = torch.cat((image_out, audio_out), 1)
        final_out = self.final_discriminator(fused_out)
        # need out = nn.Sigmoid()(out)
        return final_out


class CMC_G(nn.Module):

    def __init__(self, norm_layer=nn.InstanceNorm2d):
        super(CMC_G, self).__init__()

        self.GE_V2A = Encoder(inp_channels=3, norm_layer=norm_layer)
        self.GD_V2A = Decoder(oup_channels=1, norm_layer=norm_layer)

        self.GE_A2V = Encoder(inp_channels=1, norm_layer=norm_layer)
        self.GD_A2V = Decoder(oup_channels=3, norm_layer=norm_layer)

        self.GE_A2A = Encoder(inp_channels=1, norm_layer=norm_layer)
        self.GD_A2A = Decoder(oup_channels=1, norm_layer=norm_layer)

        self.GE_V2V = Encoder(inp_channels=3, norm_layer=norm_layer)
        self.GD_V2V = Decoder(oup_channels=3, norm_layer=norm_layer)

    def forward(self, real_image, real_audio, label, noise):
        cvrt_audio = self.GD_V2A(self.GE_V2A(real_image), label, noise)
        recv_image = self.GD_A2V(self.GE_A2V(cvrt_audio), label, noise)

        cvrt_image = self.GD_A2V(self.GE_A2V(real_audio), label, noise)
        recv_audio = self.GD_V2A(self.GE_V2A(cvrt_image), label, noise)

        self_recv_audio = self.GD_A2A(self.GE_A2A(real_audio), label, noise)
        self_recv_image = self.GD_V2V(self.GE_V2V(real_image), label, noise)

        cvrt_self_recv_image = self.GD_A2V(self.GE_A2V(self_recv_audio), label, noise)
        cvrt_self_recv_audio = self.GD_V2A(self.GE_V2A(self_recv_image), label, noise)

        G_stat_dict = {
            'cvrt_audio': cvrt_audio,
            'cvrt_image': cvrt_image,
            'recv_image': recv_image,  # fake
            'recv_audio': recv_audio,  # fake
            'cvrt_self_recv_image': cvrt_self_recv_image,
            'cvrt_self_recv_audio': cvrt_self_recv_audio,
        }
        return G_stat_dict


class CMC_D(nn.Module):

    def __init__(self, norm_layer=nn.InstanceNorm2d):
        super(CMC_D, self).__init__()

        self.D_CMC = Decider(norm_layer=norm_layer)

    def forward(self, real_image, real_audio, fake_image, fake_audio, G_stat_dict, detach=False):
        cvrt_audio = G_stat_dict['cvrt_audio']
        cvrt_image = G_stat_dict['cvrt_image']
        if detach:
            cvrt_audio = cvrt_audio.detach()
            cvrt_image = cvrt_image.detach()

        disc_rira = self.D_CMC(real_image, real_audio)
        disc_rifa = self.D_CMC(real_image, fake_audio)  # shuffled audio
        disc_rica = self.D_CMC(real_image, cvrt_audio)  # generate audio
        disc_fira = self.D_CMC(fake_image, real_audio)  # shuffled image
        disc_cira = self.D_CMC(cvrt_image, real_audio)  # generate image

        D_stat_dict = {
            'disc_rira': disc_rira,
            'disc_rifa': disc_rifa,
            'disc_rica': disc_rica,
            'disc_fira': disc_fira,
            'disc_cira': disc_cira,
        }
        return D_stat_dict


class CMCLoss():

    def __init__(self, cfg):
        super(CMCLoss, self).__init__()
        # GAN mode: vanilla
        self.crit_disc = GANLoss(gan_mode='vanilla')
        self.crit_recv = nn.L1Loss()

    def optim_D(self, D_stat_dict):
        d_rira_loss = self.crit_disc(D_stat_dict['disc_rira'], True)
        d_rifa_loss = self.crit_disc(D_stat_dict['disc_rifa'], False)
        d_rica_loss = self.crit_disc(D_stat_dict['disc_rica'], False)
        d_fira_loss = self.crit_disc(D_stat_dict['disc_fira'], False)
        d_cira_loss = self.crit_disc(D_stat_dict['disc_cira'], False)

        d_rixa_loss = (d_rira_loss + d_rifa_loss + d_rica_loss) / 3
        d_xira_loss = (d_rira_loss + d_fira_loss + d_cira_loss) / 3

        D_loss_dict = {
            'd_rixa_loss': d_rixa_loss,
            'd_xira_loss': d_xira_loss,
        }
        return D_loss_dict

    def optim_G(self, G_stat_dict, D_stat_dict, data_dict=None):
        g_rica_loss = self.crit_disc(D_stat_dict['disc_rica'], True)
        g_cira_loss = self.crit_disc(D_stat_dict['disc_cira'], True)

        i2a2i_loss = self.crit_recv(G_stat_dict['recv_image'], data_dict['real_image'])
        a2i2a_loss = self.crit_recv(G_stat_dict['recv_audio'], data_dict['real_audio'])

        a2a2i_loss = self.crit_recv(G_stat_dict['cvrt_self_recv_image'], data_dict['real_image'])
        i2i2a_loss = self.crit_recv(G_stat_dict['cvrt_self_recv_audio'], data_dict['real_audio'])

        g_i2a_loss = g_rica_loss + 1.0 * (a2i2a_loss + i2a2i_loss + a2a2i_loss + i2i2a_loss)
        g_a2i_loss = g_cira_loss + 1.0 * (a2i2a_loss + i2a2i_loss + a2a2i_loss + i2i2a_loss)

        G_loss_dict = {
            'g_i2a_loss': g_i2a_loss,
            'g_a2i_loss': g_a2i_loss,
        }
        return G_loss_dict

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


if __name__ == '__main__':
    image = torch.rand(2, 3, 128, 128)
    audio = torch.rand(2, 1, 128, 64)
    label = torch.rand(2, 13)
    noise = torch.rand(2, 100)

    GE_V2A = Encoder(inp_channels=3)
    GD_V2A = Decoder(oup_channels=1)
    cvrt_audio = GD_V2A(GE_V2A(image), audio, noise)
