from datasets import DatasetVideo2DeltaAction, DatasetVideo, DatasetVideo2VideoAndAction


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