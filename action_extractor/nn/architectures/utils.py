from .direct_resnet_mlp import BasicBlock, Bottleneck, ResNet

def center_crop(tensor, output_size=112):
    '''
    Temporary function to match input size of DINOv2
    '''
    # Ensure the input tensor is of shape [n, 3*m, 128, 128]
    assert tensor.ndim == 4 and tensor.shape[1] % 3 == 0 and tensor.shape[2] == 128 and tensor.shape[3] == 128, \
    "Input tensor must be of shape [n, 3*m, 128, 128]"

    # Calculate the starting points for the crop
    crop_size = output_size
    h_start = (tensor.shape[2] - crop_size) // 2
    w_start = (tensor.shape[3] - crop_size) // 2

    # Perform the crop
    cropped_tensor = tensor[:, :, h_start:h_start + crop_size, w_start:w_start + crop_size]
    
    return cropped_tensor

def resnet_builder(resnet_version, video_length, in_channels=3, use_spatial_softmax=False):
    if resnet_version == 'resnet18':
        block = BasicBlock
        layers = [2, 2, 2, 2]
        resnet_out_dim = 512
    elif resnet_version == 'resnet50':
        block = Bottleneck
        layers = [3, 4, 6, 3]
        resnet_out_dim = 2048
    else:
        raise ValueError("Unsupported ResNet version. Choose 'resnet18' or 'resnet50'.")
    
    return ResNet(block, layers, video_length, in_channels=in_channels, use_spatial_softmax=use_spatial_softmax), resnet_out_dim
