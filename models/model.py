import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from datasets.utils.configs import get_output_num


# Normalization on every element of input vector
class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        # Pretrained ResNet-18
        base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        base_layers = list(base_model.children())
        # Conv1 Layer output size=(N, 64, H/2, W/2)
        self.layer0 = nn.Sequential(*base_layers[:3])
        # Conv2_x Layer output size=(N, 64, H/4, W/4)
        self.layer1 = nn.Sequential(*base_layers[3:5])
        # Conv3_x Layer output size=(N, 128, H/8, W/8)
        self.layer2 = base_layers[5]
        # Conv4_x Layer output size=(N, 256, H/16, W/16)
        self.layer3 = base_layers[6]
        # Conv5_x Layer output size=(N, 512, H/32, W/32)
        self.layer4 = base_layers[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layers_output = [layer0, layer1, layer2, layer3, layer4]

        return layers_output


class TaskEmbedding(nn.Module):

    def __init__(self, n_fc, dim_latent):
        """
        Task Embedding Module, composed by fully-connection layers
        :param fc_num: Number of FC layers
        :param embed_dim: Dimension of each FC layer
        """
        super(TaskEmbedding, self).__init__()
        layers = [PixelNorm()]
        for _ in range(n_fc):
            layers.append(nn.Linear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2))

        self.fcs = nn.Sequential(*layers)

    def forward(self, latent_z):
        latent_w = self.fcs(latent_z)
        return latent_w


class CondConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel, padding, dim_latent):
        """
        Condition Convolution Module
        :param in_ch: Input channels
        :param out_ch: Output channels
        :param embed_dim: Dimension of task embedding vector
        """
        super(CondConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, padding=padding)

        # AdaIN
        self.fc_gamma = nn.Linear(dim_latent, out_ch)
        self.fc_gamma.bias.data = torch.ones(out_ch)
        self.fc_beta = nn.Linear(dim_latent, out_ch)
        self.fc_beta.bias.data = torch.zeros(out_ch)
        self.IN = nn.InstanceNorm2d(out_ch)

        self.act = nn.LeakyReLU()

    def forward(self, feature_map, latent_embedding):
        x = self.conv(feature_map)

        # AdaIN
        gamma = self.fc_gamma(latent_embedding).unsqueeze(-1).unsqueeze(-1)
        beta = self.fc_beta(latent_embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.IN(x)
        x = x * gamma + beta

        x = self.act(x)
        return x


class ConvINRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel, padding):
        super(ConvINRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, padding=padding)
        self.IN = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feature_map):
        x = self.conv(feature_map)
        x = self.IN(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):

    def __init__(self, enc_n_ch, dec_n_ch, dim_latent, last_conv_ch):
        super(Decoder, self).__init__()
        assert len(enc_n_ch) == len(dec_n_ch)
        self.N = len(enc_n_ch)

        self.skip = nn.ModuleList()
        for i in range(self.N):
            self.skip.append(CondConv(enc_n_ch[i], enc_n_ch[i], 1, 0, dim_latent))

        # Up-sampling feature maps in Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoders
        self.dec = nn.ModuleList()
        for i in range(self.N):
            if i == 0:
                in_ch = dec_n_ch[i + 1]
            elif i == (self.N - 1):
                in_ch = enc_n_ch[i - 1] + enc_n_ch[i]
            else:
                in_ch = enc_n_ch[i - 1] + dec_n_ch[i + 1]
            self.dec.append(CondConv(in_ch, dec_n_ch[i], 3, 1, dim_latent))

        # Final prediction
        self.conv_last = nn.Conv2d(dec_n_ch[0], last_conv_ch, 1)

    def forward(self, enc_output, latent_w):
        # Skip connection
        skip = []
        for i in range(len(self.skip)):
            skip.append(self.skip[i](enc_output[i], latent_w))

        # Decoder
        x = skip[-1]
        for i in range(self.N - 1, -1, -1):
            x = self.upsample(x)
            if i > 0:
                x = torch.cat([x, skip[i - 1]], dim=1)
            x = self.dec[i](x, latent_w)

        if self.N == 4:
            x = self.upsample(x)

        output = self.conv_last(x)

        return output


class Decoder_noadain(nn.Module):

    def __init__(self, enc_n_ch, dec_n_ch, last_conv_ch):
        super(Decoder_noadain, self).__init__()
        assert len(enc_n_ch) == len(dec_n_ch)
        self.N = len(enc_n_ch)

        self.skip = nn.ModuleList()
        for i in range(self.N):
            self.skip.append(ConvINRelu(enc_n_ch[i], enc_n_ch[i], 1, 0))

        # Up-sampling feature maps in Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoders
        self.dec = nn.ModuleList()
        for i in range(self.N):
            if i == 0:
                in_ch = dec_n_ch[i + 1]
            elif i == (self.N - 1):
                in_ch = enc_n_ch[i - 1] + enc_n_ch[i]
            else:
                in_ch = enc_n_ch[i - 1] + dec_n_ch[i + 1]
            self.dec.append(ConvINRelu(in_ch, dec_n_ch[i], 3, 1))

        # Final prediction
        self.conv_last = nn.Conv2d(dec_n_ch[0], last_conv_ch, 1)

    def forward(self, enc_output):
        # Skip connection
        skip = []
        for i in range(len(self.skip)):
            skip.append(self.skip[i](enc_output[i]))

        # Decoder
        x = skip[-1]
        for i in range(self.N - 1, -1, -1):
            x = self.upsample(x)
            if i > 0:
                x = torch.cat([x, skip[i - 1]], dim=1)
            x = self.dec[i](x)

        if self.N == 4:
            x = self.upsample(x)

        output = self.conv_last(x)

        return output


class TSN(nn.Module):

    def __init__(self, backbone, n_class, n_fc=8, dim_latent=512):
        """
        Task Switching Network
        :param embed_dim: Dimension of task vector
        :param task_out_ch: Number of output channels for each task
        """
        super(TSN, self).__init__()
        self.n_class = n_class

        self.task_embedding = TaskEmbedding(n_fc=n_fc, dim_latent=dim_latent)

        if backbone == 'resnet18':
            self.encoder = ResNet18()
            # Number of output channels in encoder and decoder
            enc_n_ch = [64, 64, 128, 256, 512]
            dec_n_ch = [64, 128, 256, 256, 512]
        elif backbone == 'swin-t':
            from .swin import swin_t
            self.encoder = swin_t(pretrained=True)
            enc_n_ch = [96, 192, 384, 768]
            dec_n_ch = [96, 192, 384, 768]

        self.decoder = Decoder(enc_n_ch=enc_n_ch, dec_n_ch=dec_n_ch, dim_latent=dim_latent, last_conv_ch=n_class)

    def forward(self, x, latent_z, task, dataname):
        latent_w = self.task_embedding(latent_z)
        enc_output = self.encoder(x)
        dec_output = self.decoder(enc_output, latent_w)

        # Adaptive average pooling along channels
        task_out_ch = get_output_num(task, dataname)
        if task_out_ch != self.n_class:
            output = F.adaptive_avg_pool3d(dec_output, (task_out_ch, None, None))
        else:
            output = dec_output
        # Normalize to unit vectors in normals
        if task == 'normals':
            output = F.normalize(output, p=2, dim=1)

        return {task: output}


class MD_model(nn.Module):

    def __init__(self, backbone, tasks, dataname):
        """
        Multi-Decoder Model
        """
        super(MD_model, self).__init__()

        if backbone == 'resnet18':
            self.encoder = ResNet18()
            # Number of output channels in encoder and decoder
            enc_n_ch = [64, 64, 128, 256, 512]
            dec_n_ch = [64, 128, 256, 256, 512]
        elif backbone == 'swin-t':
            from .swin import swin_t
            self.encoder = swin_t(pretrained=True)
            enc_n_ch = [96, 192, 384, 768]
            dec_n_ch = [96, 192, 384, 768]

        self.decoder = nn.ModuleDict()
        for task in tasks:
            self.decoder[task] = Decoder_noadain(enc_n_ch=enc_n_ch,
                                                 dec_n_ch=dec_n_ch,
                                                 last_conv_ch=get_output_num(task, dataname))

    def forward(self, x):
        enc_output = self.encoder(x)
        output = {}
        for task in self.decoder:
            dec_output = self.decoder[task](enc_output)
            if task == 'normals':
                dec_output = F.normalize(dec_output, p=2, dim=1)
            output[task] = dec_output

        return output
