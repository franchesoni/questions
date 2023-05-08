from torchvision.models import resnet18, ResNet18_Weights
import torch

def get_resnet(pretrained=False, n_channels=3, compile=True):
  assert n_channels in [1, 3]
  net = resnet18(weights=ResNet18_Weights if pretrained else None,
          num_classes=1)
  if n_channels == 1:
    def new_forward(self, x: torch.Tensor) -> torch.Tensor:
      B, C, H, W = x.shape
      return self._forward_impl(x.expand(B, 3, H, W))  # rough conversion to RGB
    net.forward = new_forward.__get__(net)
  net.maxpool = torch.nn.Identity()  # to handle sizes correctly
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net.to(device)
  if compile and torch.__version__.startswith('2'):
    net = torch.compile(net)  # experimental
  return net

