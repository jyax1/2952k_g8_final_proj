from datasets import *



def load_datasets(
        architecture, 
        datasets_path, 
        train=True, 
        validation=True, 
        horizon=2, 
        demo_percentage=0.9, 
        cameras=['frontview_image'],
        motion=False,
        image_plus_motion=False
):
    if 'latent' in architecture and 'decoder' not in architecture and 'aux' not in architecture:
        if train:
            train_set = DatasetVideo(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=demo_percentage, cameras=cameras)
        if validation:
            validation_set = DatasetVideo(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                                    demo_percentage=.9, cameras=cameras, validation=True)
    elif 'latent' in architecture and 'aux' in architecture:
        if train:
            train_set = DatasetVideo2VideoAndAction(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                            demo_percentage=demo_percentage, cameras=cameras)
        if validation:
            validation_set = DatasetVideo2VideoAndAction(path=datasets_path, x_pattern=[0,1], y_pattern=[1],
                                                    demo_percentage=.9, cameras=cameras, validation=True)
    else:
        if train:
            train_set = DatasetVideo2DeltaAction(path=datasets_path, video_length=horizon, 
                                            demo_percentage=demo_percentage, cameras=cameras,
                                            motion=motion, image_plus_motion=image_plus_motion)
        if validation:
            validation_set = DatasetVideo2DeltaAction(path=datasets_path, video_length=horizon, 
                                                demo_percentage=demo_percentage, cameras=cameras, validation=True, 
                                                motion=motion, image_plus_motion=image_plus_motion)

    if train and validation:
        return train_set, validation_set
    elif train and not validation:
        return train_set
    elif validation and not train:
        return validation_set
    
def load_model():
        if args.architecture == 'direct_cnn_mlp':
        model = ActionExtractionCNN(latent_dim=args.latent_dim, 
                                    video_length=args.horizon, 
                                    motion=args.motion, 
                                    image_plus_motion=args.image_plus_motion)
    elif args.architecture == 'direct_cnn_vit':
        model = ActionExtractionViT(latent_dim=args.latent_dim, 
                                    video_length=args.horizon, 
                                    motion=args.motion, 
                                    image_plus_motion=args.image_plus_motion,
                                    vit_patch_size=args.vit_patch_size)
    elif args.architecture == 'direct_resnet_mlp':
        resnet_version = 'resnet' + str(args.resnet_layers_num)
        model = ActionExtractionResNet(resnet_version, action_length=args.horizon-1, num_mlp_layers=3)
    elif args.architecture == 'latent_cnn_unet':
        model = ActionExtractionCNNUNet(latent_dim=args.latent_dim, video_length=args.horizon) # doesn't support motion
    elif 'latent_decoder' in args.architecture:
        idm_model_path = str(Path(results_path)) + f'/{args.idm_model_name}'
        latent_dim = int(re.search(r'lat_(.*?)_', args.idm_model_name).group(1))
        if args.architecture == 'latent_decoder_mlp':
            model = LatentDecoderMLP(idm_model_path, 
                                     latent_dim=latent_dim, 
                                     video_length=args.horizon, 
                                     latent_length=args.horizon-1, 
                                     mlp_layers=10)
        elif args.architecture == 'latent_decoder_vit':
            model = LatentDecoderTransformer(idm_model_path, 
                                             latent_dim=latent_dim, 
                                             video_length=args.horizon, 
                                             latent_length=args.horizon-1,
                                             vit_patch_size=args.vit_patch_size)
        elif args.architecture == 'latent_decoder_obs_conditioned_unet_mlp':
            model = LatentDecoderObsConditionedUNetMLP(idm_model_path,
                                                       latent_dim=latent_dim,
                                                       video_length=args.horizon,
                                                       latent_length=args.horizon-1,
                                                       mlp_layers=10)
        elif args.architecture == 'latent_decoder_aux_separate_unet_vit':
            fdm_model_path = str(Path(results_path)) + f'/{args.fdm_model_name}'
            model = LatentDecoderAuxiliarySeparateUNetTransformer(idm_model_path, 
                                                        fdm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=args.horizon, 
                                                        freeze_idm=args.freeze_idm, 
                                                        freeze_fdm=args.freeze_fdm,
                                                        vit_patch_size=args.vit_patch_size)
        elif args.architecture == 'latent_decoder_aux_separate_unet_mlp':
            fdm_model_path = str(Path(results_path)) + f'/{args.fdm_model_name}'
            model = LatentDecoderAuxiliarySeparateUNetMLP(idm_model_path, 
                                                        fdm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=args.horizon, 
                                                        freeze_idm=args.freeze_idm, 
                                                        freeze_fdm=args.freeze_fdm)
        elif args.architecture == 'latent_decoder_aux_combined_vit':
            fdm_model_path = str(Path(results_path)) + f'/{args.fdm_model_name}'
            model = LatentDecoderAuxiliaryCombinedViT(idm_model_path, 
                                                        latent_dim=latent_dim, 
                                                        video_length=args.horizon, 
                                                        freeze_idm=args.freeze_idm)
