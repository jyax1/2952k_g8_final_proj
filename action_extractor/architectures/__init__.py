# action_extractor/architectures/__init__.py
from .direct_cnn_mlp import ActionExtractionCNN, PoseExtractionCNN3D 
from .direct_cnn_vit import ActionExtractionViT
from .direct_resnet_mlp import ActionExtractionResNet
from .direct_variational_resnet import ActionExtractionVariationalResNet
from .latent_decoders import *
from .latent_encoders import LatentEncoderPretrainCNNUNet, LatentEncoderPretrainResNetUNet
from .resnet import *

__all__ = ['ActionExtractionResNet',
           'ActionExtractionVariationalResNet']