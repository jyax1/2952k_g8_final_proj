ARCHITECTURES = [
    'direct_cnn_mlp', 
    'direct_cnn_vit',
    'direct_resnet_mlp',
    'direct_N_variational_resnet',
    'direct_S_variational_resnet',
    'latent_encoder_resnet_unet',
    'latent_encoder_cnn_unet', 
    'latent_decoder_mlp', 
    'latent_decoder_vit', 
    'latent_decoder_obs_conditioned_unet_mlp',
    'latent_decoder_aux_separate_unet_mlp',
    'latent_decoder_aux_separate_unet_vit',
    'latent_decoder_aux_combined_unet_mlp',
    'latent_decoder_aux_combined_vit',
    'flownet2'
]

VALID_LATENT_DIMS = [4, 8, 16, 32]