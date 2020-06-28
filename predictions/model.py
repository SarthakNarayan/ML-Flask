import torch
import torch.nn as nn


class StandardBlock(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 16) -> None:
        super(StandardBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int = 16,
                 out_ch: int = 16, dil: int = 1) -> None:
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch))

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,
                      padding=dil, dilation=dil, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                      padding=dil, dilation=dil, bias=True),
            nn.BatchNorm2d(out_ch))

        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.conv(x)
        x = x + res
        x = self.elu(x)
        return x


class UpConvDRUNet(nn.Module):
    def __init__(self) -> None:
        super(UpConvDRUNet, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x


class DRUnet(nn.Module):
    def __init__(self) -> None:
        super(DRUnet, self).__init__()

        # Can define as constant
        input_channel = 1
        output_channel = 6
        num_features = 16

        filters = [num_features, num_features * 2,
                   num_features * 4, num_features * 8]
        self.img_ch = input_channel

        self.conv_1 = StandardBlock(in_ch=input_channel, out_ch=filters[0])
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = ResidualBlock(
            in_ch=filters[0], out_ch=filters[1], dil=2)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = ResidualBlock(
            in_ch=filters[1], out_ch=filters[2], dil=4)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4 = ResidualBlock(
            in_ch=filters[2], out_ch=filters[3], dil=8)

        self.up_conv_3 = UpConvDRUNet()
        self.reduce_filters_3 = nn.Conv2d(
            in_channels=filters[3] + filters[2], out_channels=filters[3],
            kernel_size=1, stride=1, padding=0, bias=True)
        self.up_conv_res_3 = ResidualBlock(
            in_ch=filters[3], out_ch=filters[2], dil=4)

        self.up_conv_2 = UpConvDRUNet()
        self.reduce_filters_2 = nn.Conv2d(
            in_channels=filters[2] + filters[1], out_channels=filters[2],
            kernel_size=1, stride=1, padding=0, bias=True)
        self.up_conv_res_2 = ResidualBlock(
            in_ch=filters[2], out_ch=filters[1], dil=2)

        self.up_conv_1 = UpConvDRUNet()
        self.reduce_filters_1 = nn.Conv2d(
            in_channels=filters[1] + filters[0], out_channels=filters[1],
            kernel_size=1, stride=1, padding=0, bias=True)
        self.up_conv_stand_1 = StandardBlock(
            in_ch=filters[1], out_ch=filters[0])

        self.conv_1x1 = nn.Conv2d(
            in_channels=filters[0], out_channels=output_channel, kernel_size=1,
            stride=1, padding=0, bias=True)

        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        c_1 = self.conv_1(x)
        m_1 = self.maxpool_1(c_1)

        c_2 = self.conv_2(m_1)
        m_2 = self.maxpool_2(c_2)

        c_3 = self.conv_3(m_2)
        m_3 = self.maxpool_2(c_3)

        c_4 = self.conv_4(m_3)

        u_3 = self.up_conv_3(c_4)

        con_3 = torch.cat((c_3, u_3), dim=1)
        red_3 = self.reduce_filters_3(con_3)
        res_3 = self.up_conv_res_3(red_3)

        u_2 = self.up_conv_2(res_3)
        con_2 = torch.cat((c_2, u_2), dim=1)
        red_2 = self.reduce_filters_2(con_2)
        res_2 = self.up_conv_res_2(red_2)

        u_1 = self.up_conv_1(res_2)
        con_1 = torch.cat((c_1, u_1), dim=1)
        red_1 = self.reduce_filters_1(con_1)
        res_1 = self.up_conv_stand_1(red_1)

        x = self.conv_1x1(res_1)

        if x.shape[1] == 1:
            x = self.output(x)
        return x