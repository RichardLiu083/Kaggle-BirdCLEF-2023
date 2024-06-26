import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class TimmSED(nn.Module):
    def __init__(
        self, 
        base_model_name: str, 
        config=None,
        pretrained=False, 
        num_classes=264, 
        in_channels=3,
    ):
        super().__init__()
        
        self.config = config

        self.bn0 = nn.BatchNorm2d(self.config.n_mels)

        base_model = timm.create_model(
            base_model_name, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool="",
            in_chans=in_channels,
        )
        
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.num_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        
    def forward(self, input_data):
        if self.config.in_channels == 3:
            x = input_data
        else:
            x = input_data[:, [0], :, :] # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)


        x = x.transpose(2, 3)

        x = self.encoder(x)
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=2)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)

        output_dict = {
            "clipwise_output": clipwise_output,
        }

        return clipwise_output

    
    
    
    
    
class SED_Model(nn.Module):
    def __init__(self, model_name, classes_num):
        super().__init__()
        self.encoder= timm.create_model(model_name, 
                                        num_classes=0, 
                                        pretrained= True,
                                        drop_rate= CFG['drop_out'], 
                                        drop_path_rate= CFG['drop_path'])
        self.encoder.global_pool= nn.Identity()
        in_feat= 1280
        self.cla= nn.Conv1d(
                        in_channels=in_feat,
                        out_channels=classes_num,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    )
        self.att= nn.Conv1d(
                        in_channels=in_feat,
                        out_channels=classes_num,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    )
    
    def forward(self, x):
        x= self.encoder(x)
        x= torch.mean(x, dim=2)
        
        att_map= self.att(x).sigmoid()
        x= self.cla(x).sigmoid()
#         print(att_map)
        out= x*att_map
        out= torch.sum(out, dim=2)
        out= torch.clamp(out, 0, 1)

        return out
    
    
    
class PA_Model(nn.Module):
    def __init__(self, model_name, classes_num):
        super().__init__()
        self.output_att= False
        self.encoder= timm.create_model(model_name, 
                                        num_classes=0, 
                                        pretrained= True,
                                        drop_rate= 0, 
                                        drop_path_rate= 0)
        self.encoder.global_pool= nn.Identity()
        in_feat= 1280
        self.cla= nn.Conv2d(
                        in_channels=in_feat,
                        out_channels=classes_num,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    )
        self.att= nn.Conv2d(
                        in_channels=in_feat,
                        out_channels=classes_num,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    )
    
    def forward(self, x):
        x= self.encoder(x)
        att_map= self.att(x).sigmoid()
#         print(att_map)
        
        dim= (x.shape[2], x.shape[3])
        x= self.cla(x)
        x= x.reshape(x.shape[0], x.shape[1], -1)
        x= x.softmax(dim=-1)
        x= x.reshape(x.shape[0], x.shape[1], *dim)
        
        out= x*att_map
        out= out.sum(dim=-1).sum(dim=-1)
        out= torch.clamp(out, 0, 1)

        if self.output_att:
            return out, att_map
        else:
            return out