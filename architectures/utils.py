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