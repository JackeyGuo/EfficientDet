import math
from torch import nn
from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, onnx_export=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))

        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef], onnx_export=onnx_export)
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef], onnx_export=onnx_export)

        # 将分类和回归层的参数置为0，让初始训练时分类损失更加稳定
        prior = 0.01
        self.classifier.header.pointwise_conv.weight.data.fill_(0)
        self.classifier.header.pointwise_conv.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressor.header.pointwise_conv.weight.data.fill_(0)
        self.regressor.header.pointwise_conv.bias.data.fill_(0)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()

    def forward(self, inputs):

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)

        return regression, classification

    # def init_backbone(self, path):
    #     state_dict = torch.load(path)
    #     try:
    #         ret = self.load_state_dict(state_dict, strict=False)
    #         print(ret)
    #     except RuntimeError as e:
    #         print('Ignoring ' + str(e) + '"')
