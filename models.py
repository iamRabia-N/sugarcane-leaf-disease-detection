import torch
import torch.nn as nn
import timm


def get_model(model_name, num_classes=5):
    if model_name == 'SwinT':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True,
                                  num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.2)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layers[2].parameters():
            param.requires_grad = True
        for param in model.layers[3].parameters():
            param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True
        for param in model.head.parameters():
            param.requires_grad = True
        return model

    if model_name == 'ViT':
        model = timm.create_model('vit_base_patch16_224', pretrained=True,
                                  num_classes=num_classes, drop_rate=0.1, attn_drop_rate=0.1)
        for param in model.parameters():
            param.requires_grad = False
        for block in model.blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True
        for param in model.head.parameters():
            param.requires_grad = True
        return model

    if model_name == 'DeiT':
        model = timm.create_model('deit_small_patch16_224', pretrained=True,
                                  num_classes=num_classes, drop_rate=0.1, attn_drop_rate=0.1)
        for param in model.parameters():
            param.requires_grad = False
        for block in model.blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True
        for param in model.head.parameters():
            param.requires_grad = True
        return model

    if model_name == 'ResNet50':
        model = timm.create_model('resnet50', pretrained=True,
                                  num_classes=num_classes, drop_rate=0.2)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        if hasattr(model, 'global_pool'):
            for param in model.global_pool.parameters():
                param.requires_grad = True
        return model

    if model_name == 'EfficientNetB3':
        model = timm.create_model('efficientnet_b3', pretrained=True,
                                  num_classes=num_classes, drop_rate=0.3, drop_path_rate=0.2)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.blocks[-3:].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        if hasattr(model, 'conv_head'):
            for param in model.conv_head.parameters():
                param.requires_grad = True
        if hasattr(model, 'bn2'):
            for param in model.bn2.parameters():
                param.requires_grad = True
        if hasattr(model, 'global_pool'):
            for param in model.global_pool.parameters():
                param.requires_grad = True
        return model

    if model_name == 'CNN3D':
        return CNN3DModel(num_classes)

    raise ValueError(f"Unknown model: {model_name}")


class CNN3DModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)