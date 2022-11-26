import math
import torch
import torch.nn.functional as F
from torch import nn
import sys
from IPython import embed

sys.path.append('./')
sys.path.append('../')
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead


class TSRN(nn.Module):
    """
    TSRN模型
    """
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=True, hidden_units=32):
        super(TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(            # 网络中的第1个卷积层
            nn.Conv2d(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            nn.PReLU()                          # PReLU激活函数就是ReLU激活函数在x<0那段是有斜率的情况
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):               # 定义5个SRB，下面的写法跟self.xxx=yyy是一个意思
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2*hidden_units)
                ))
        
        block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [32, 64]          # TPS模块要求输入的长宽
        tps_outputsize = [height//scale_factor, width//scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        """
        输入：(N, 4, 16, 64)
        """
        if self.stn and self.training:
            # align_corners=True和False是两种不同的插值方式，最终都能得到(N, 4, 32, 64)
            x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)               # (N, 4, 16, 64)
        block = {'1': self.block1(x)}                       # (N, 2*hidden_units, 16, 64)
        for i in range(self.srb_nums + 1):                  # (N, 2*hidden_units, 16, 64)
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))  # (N, 4, 32, 128)
        output = torch.tanh(block[str(self.srb_nums + 3)])  # 保证像素值在[0, 1]的范围内
        return output


class RecurrentResidualBlock(nn.Module):
    """
    SRB模块，输入(N, C, H, W)
    """
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        self.prelu = mish()                     # y = x*tanh(ln(1+exp(x)))，也是ReLU变种的激活函数
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)                # (N, C, H, W)
        residual = self.bn1(residual)           # (N, C, H, W)
        residual = self.prelu(residual)         # (N, C, H, W)
        residual = self.conv2(residual)         # (N, C, H, W)
        residual = self.bn2(residual)           # (N, C, H, W)

        # (N, C, H, W)->(N, C, W, H)->(N, C, H, W)，通过这种调换建模水平、竖直方向
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)

        return self.gru2(x + residual)          # (N, C, H, W)


class UpsampleBLock(nn.Module):
    """
    用PixelShuffle实现上采样，输入(N, in_channels, H, W)
    """
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)                    # (N, in_channels*up_scale**2, H, W)
        x = self.pixel_shuffle(x)           # (N, in_channels, up_scale*H, up_scale*W)
        x = self.prelu(x)                   # (N, in_channels, up_scale*H, up_scale*W)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    """
    GRU块，输入(N, in_channels, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)                           # (N, out_channels, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()      # (N, H, W, out_channels)
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])         # (N*H, W, out_channels)
        x, _ = self.gru(x)                          # (N*H, W, out_channels)
        x = x.view(b[0], b[1], b[2], b[3])          # (N, H, W, out_channels)
        x = x.permute(0, 3, 1, 2)                   # (N, out_channels, H, W)
        return x


class TSRN_TL(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=True, hidden_units=32, word_vec_d=300, text_emb=37, out_text_channels=32):
        super(TSRN_TL, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockTL(2 * hidden_units, out_text_channels))

        # self.w2v_proj = ImFeat2WordVec(2 * hidden_units, word_vec_d)
        # self.semantic_R = ReasoningTransformer(2 * hidden_units)


        self.feature_enhancer = None #FeatureEnhancerW2V(
                                #        vec_d=300,
                                #        feature_size=2 * hidden_units,
                                #        head_num=4,
                                #        dropout=True)

        # From [1, 1] -> [16, 16]
        self.infoGen = InfoGen(text_emb, out_text_channels)
        self.emb_cls = text_emb

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)

    def forward(self, x, text_emb=None):
        # embed()

        # print("self.tps_inputsize:", self.tps_inputsize)

        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        # all_pred_vecs = []

        if text_emb is None:
            N, C, H, W = x.shape
            text_emb = torch.zeros((N, self.emb_cls, 1, 26))

        spatial_t_emb = self.infoGen(text_emb)
        spatial_t_emb = F.interpolate(spatial_t_emb, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        # print("x", x.shape, spatial_t_emb.shape)

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in [2, 3, 4, 5, 6]:
                # pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                # all_pred_vecs.append(pred_word_vecs)
                # if not self.training:
                #     word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], spatial_t_emb)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])

        return output

class RecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTL, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels + text_channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

        # self.concat_conv = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)

    def forward(self, x, text_emb):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        ############ Fusing with TL ############
        cat_feature = torch.cat([residual, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################

        residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)


class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=(1, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        return x


if __name__ == '__main__':
    img = torch.zeros(7, 3, 16, 64)
    embed()
