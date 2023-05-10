from torchvision.models import resnet18, ResNet18_Weights
import logging
import torch
import torch._dynamo


def get_resnet(pretrained=False, n_channels=3, compile=False, device=None):
    assert n_channels in [1, 3]
    net = resnet18(weights=ResNet18_Weights if pretrained else None, num_classes=1)
    if n_channels == 1:

        def new_forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            return self._forward_impl(x.expand(B, 3, H, W))  # rough conversion to RGB

        net.forward = new_forward.__get__(net)
    net.maxpool = torch.nn.Identity()  # to handle sizes correctly
    device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    net.to(device)
    if compile and torch.__version__.startswith("2"):
        torch._dynamo.config.log_level = logging.INFO
        net = torch.compile(
            net, mode="reduce-overhead"
        )  # experimental, locally decreases speed
    return net


def get_embedding(predictor, x: torch.Tensor) -> torch.Tensor:
    # See note [TorchScript super()]
    x = predictor.conv1(x)
    x = predictor.bn1(x)
    x = predictor.relu(x)
    x = predictor.maxpool(x)

    x = predictor.layer1(x)
    x = predictor.layer2(x)
    x = predictor.layer3(x)
    x = predictor.layer4(x)

    x = predictor.avgpool(x)
    embedding = torch.flatten(x, 1)
    x = predictor.fc(embedding)

    return x, embedding
