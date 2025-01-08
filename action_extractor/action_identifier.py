import argparse
import numpy as np
import torch
import torch.nn as nn
from action_extractor.architectures.direct_resnet_mlp import ResNetMLP, ActionExtractionResNet
from action_extractor.architectures.direct_variational_resnet import (
    ActionExtractionVariationalResNet,
    ActionExtractionHypersphericalResNet,
    ActionExtractionSLAResNet
)
from action_extractor.utils.utils import load_model, load_trained_model, load_datasets

# --------------------------------------------------------------------------
# Example encoder classes
# --------------------------------------------------------------------------
class VariationalEncoder(nn.Module):
    """
    Simple example for a normal (mu, logvar) encoder in a 2-part model 
    (encoder, decoder).
    """
    def __init__(self, conv, fc_mu, fc_logvar):
        super(VariationalEncoder, self).__init__()
        self.conv = conv
        self.flatten = nn.Flatten()
        self.fc_mu = fc_mu
        self.fc_logvar = fc_logvar

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VMFEncoder(nn.Module):
    """
    Example for a vMF-based encoder:
      conv -> flatten -> (fc_mu, fc_kappa).
    We'll handle c if it’s SLA (use_distribution_for_c or not).
    """
    def __init__(self, 
                 conv, 
                 fc_mu, 
                 fc_kappa, 
                 fc_c=None, 
                 fc_c_mu=None, 
                 fc_c_logvar=None, 
                 fc_gripper=None,
                 use_distribution_for_c=False):
        super(VMFEncoder, self).__init__()
        self.conv = conv
        self.flatten = nn.Flatten()
        self.fc_mu = fc_mu
        self.fc_kappa = fc_kappa
        # For SLA
        self.fc_c = fc_c
        self.fc_c_mu = fc_c_mu
        self.fc_c_logvar = fc_c_logvar
        self.fc_gripper = fc_gripper
        self.use_distribution_for_c = use_distribution_for_c

    def forward(self, x):
        """
        We'll return different shapes depending on if we have c, c_mu, etc.
        """
        h = self.conv(x)
        h = self.flatten(h)

        mu = nn.functional.normalize(self.fc_mu(h), dim=-1)
        kappa = nn.functional.softplus(self.fc_kappa(h)) + 1.0

        # If we're an SLA model, also produce gripper + c
        if self.fc_gripper is not None:
            raw_gripper = self.fc_gripper(h)
        else:
            raw_gripper = None

        if self.fc_c_mu is not None and self.fc_c_logvar is not None:
            # distribution for c
            c_mu = self.fc_c_mu(h)
            c_logvar = self.fc_c_logvar(h)
            return mu, kappa, c_mu, c_logvar, raw_gripper
        elif self.fc_c is not None:
            # deterministic c
            c_det = self.fc_c(h)
            return mu, kappa, c_det, raw_gripper
        else:
            # normal vMF (no c)
            return mu, kappa


# --------------------------------------------------------------------------
# The ActionIdentifier that can handle all
# --------------------------------------------------------------------------
class ActionIdentifier(nn.Module):
    def __init__(
        self, 
        encoder, 
        decoder, 
        stats_path='action_statistics_delta_position+gripper.npz', 
        coordinate_system='global', 
        camera_name='frontview', 
        deterministic=True
    ):
        super(ActionIdentifier, self).__init__()
        self.deterministic = deterministic

        self.encoder = encoder
        self.decoder = decoder
        
        # Load standardization stats
        stats = np.load(stats_path)
        self.action_mean = torch.tensor(stats['action_mean'], dtype=torch.float32)
        self.action_std = torch.tensor(stats['action_std'], dtype=torch.float32)
        
        # Set coordinate system and camera parameters  
        self.coordinate_system = coordinate_system
        self.camera_name = camera_name
        
        # Select appropriate camera matrix based on camera_name
        from action_extractor.utils.dataset_utils import frontview_R, sideview_R, agentview_R, sideagentview_R
        if camera_name == 'frontview':
            self.R = frontview_R
        elif camera_name == 'sideview':
            self.R = sideview_R  
        elif camera_name == 'agentview':
            self.R = agentview_R
        elif camera_name == 'sideagentview':
            self.R = sideagentview_R
        else:
            raise ValueError(f"Unknown camera name: {camera_name}")

    @staticmethod
    def reparameterize(mu, logvar):
        """For normal distribution (mu, logvar)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        """
        We'll detect whether our encoder returns:
          - (mu, logvar)
          - (mu, kappa)
          - (mu, kappa, c, [gripper?]) or (mu, kappa, c_mu, c_logvar, [gripper?])
          - or is just a standard non-variational conv stack.

        We'll pass it forward and see what shape it has, 
        or use 'isinstance(encoder, ...)' approach. 
        But let's do a simpler approach: we actually just call the encoder 
        and see how many items are returned.
        """
        result = self.encoder(x)

        if not isinstance(result, tuple):
            # non-variational => just a feature map
            return result
        else:
            # We have some form of variational or hyperspherical output
            # We'll handle each based on length
            if len(result) == 2:
                # (mu, logvar) => normal
                mu, logvar = result
                if self.deterministic:
                    # no reparam
                    return mu
                else:
                    z = self.reparameterize(mu, logvar)
                    return z, mu, logvar
            elif len(result) == 3:
                # (mu, kappa) or (mu, kappa, c_det)
                # We can't directly reparam a c_det. 
                # Let's see if c_det is a single dimension or not. 
                # For non-SLA Hyperspherical: (mu, kappa)
                # We'll do a simpler approach: we treat them as "non-deterministic" if we want?
                # Typically, vMF is reparam done in the decoder or model forward. 
                # So if we just have (mu, kappa), we'll treat it as a "latent" that doesn't do normal reparam.
                # We'll just return them as is for the user to decode?
                mu, kappa, maybe_c = result
                # We'll guess if maybe_c is None => we do (mu, kappa).
                # We'll check shape. If maybe_c is a single dim => it's probably c_det. 
                # For now, let's just return the triple.
                return result  # We pass forward as is.
            elif len(result) == 4:
                # (mu, kappa, c_det, raw_gripper) or something else
                return result
            elif len(result) == 5:
                # (mu, kappa, c_mu, c_logvar, raw_gripper)
                return result
            else:
                return result  # fallback

    def decode(self, x):
        """
        x here might be the "features" or the "latent" from the reparam step.
        Then we pass it into the decoder if it's an nn.Module. 
        For vMF-based models, the "forward" might already do the reparam & decode 
        in one pass, so we might only need standardization, etc.
        """
        if callable(self.decoder):
            action = self.decoder(x)
        else:
            # If there's no decoder, just assume x is the final output
            action = x

        # Unstandardize
        if isinstance(action, torch.Tensor):
            action = action * self.action_std.to(action.device) + self.action_mean.to(action.device)
            
            # Convert to global coordinates if needed
            if self.coordinate_system in ['camera', 'disentangled']:
                action = self.transform_to_global(action)
                
        return action
    
    def forward(self, x):
        # We'll do a standard approach: 
        #  1) encode => might get multiple items
        #  2) decode if needed
        # 
        # But for your vMF-based classes, the entire forward pass might produce 
        # final actions already, so we have to detect that too.
        # If self.encoder + self.decoder is truly separated, we do the standard approach.
        # If the model lumps it all in "forward," then our encoder might actually produce final actions.
        # We'll do a check if the encoder returns a single Tensor of shape [B, something].
        # We'll do a naive approach: if the encoder returns shape (B, ...), we decode. 
        # Otherwise, if it's a tuple with shape (B, ...) as the first item, we decode that. 
        # But for vMF-based classes, the reparam is done in the "model forward" not here. 
        # So maybe we bypass decode? 
        # 
        # A simpler approach: if "decoder" is None, we assume the "encoder" is actually the entire model.
        # If the decoder is not None, we do the old approach.

        encoded = self.encode(x)

        if self.decoder is None:
            # Then "encode" is the entire model forward => final output
            # see if we got a single or tuple
            if isinstance(encoded, tuple):
                # typically final output is encoded[0]
                out = encoded[0]
                return out
            else:
                return encoded
        else:
            # We have a separate decoder => do the old flow
            if not isinstance(encoded, tuple):
                # Non-variational => final step
                return self.decode(encoded)
            else:
                # We might have (z, mu, logvar) for normal
                # or (mu, kappa, c, ...) for vMF
                if len(encoded) == 2:
                    # (mu, logvar) => the user didn't want reparam? 
                    # We'll do decode on mu? Not typical. 
                    # This is tricky. 
                    # Possibly we do self.decode(mu).
                    # But let’s keep consistent with the standard reparam approach 
                    # from our "encode" method => if deterministic => returned mu, else returned (z, mu, logvar).
                    mu, logvar = encoded
                    return self.decode(mu)
                elif len(encoded) == 3:
                    # Possibly (z, mu, logvar) from normal 
                    # or (mu, kappa, c_det)
                    # If it's (z, mu, logvar), we decode z:
                    z, mu, logvar = encoded
                    return self.decode(z)
                elif len(encoded) == 5:
                    # (out, mu, kappa, c_mu, c_logvar) from an SLA forward? 
                    # Actually might see we do "model forward" in training. 
                    # But if using the "encoder/decoder" approach, we didn't do model's forward. 
                    # We just parted out the submodules. 
                    # So let's assume it's the actual latents => we'd decode the first. 
                    # We'll do something minimal.
                    return self.decode(encoded[0])
                else:
                    return self.decode(encoded[0])

    def transform_to_global(self, action):
        # same from your original code
        pos = action[:, :3]
        other = action[:, 3:]

        if self.coordinate_system == 'camera':
            pos_homog = torch.cat([pos, torch.ones(pos.shape[0], 1).to(pos.device)], dim=1)
            R_inv = torch.inverse(torch.from_numpy(self.R).float().to(pos.device))
            pos_global = (R_inv @ pos_homog.unsqueeze(-1)).squeeze(-1)[:, :3]
        elif self.coordinate_system == 'disentangled':
            x_over_z, y_over_z, log_z = pos[:, 0], pos[:, 1], pos[:, 2]
            z = torch.exp(log_z)
            x = x_over_z * z
            y = y_over_z * z
            pos_global = torch.stack([x, y, z], dim=1)
        return torch.cat([pos_global, other], dim=1)

# --------------------------------------------------------------------------
# Updated load_action_identifier
# --------------------------------------------------------------------------
def load_action_identifier(
    checkpoint_path=None,
    # separate param paths
    conv_path=None,
    mlp_path=None,
    fc_mu_path=None,
    fc_logvar_path=None,
    fc_kappa_path=None,
    fc_c_path=None,
    fc_c_mu_path=None,
    fc_c_logvar_path=None,
    fc_gripper_path=None,
    # model config
    resnet_version='resnet18',
    video_length=2,
    in_channels=3,
    action_length=1,
    num_classes=7,
    num_mlp_layers=3,
    # additional options
    stats_path='action_statistics_delta_position+gripper.npz',
    coordinate_system='global',
    camera_name='frontview',
    split_layer='avgpool',  # The last layer name to split the ResNet
    deterministic=True,
    # new param to pick which architecture we want:
    arch_type='resnet',   # or 'variational', 'hyperspherical', 'sla'
    use_distribution_for_c=False
):
    """
    Modified so that for ActionExtractionResNet, we ALWAYS split into encoder+decoder,
    whether loading from checkpoint or partial param paths. This ensures we call
    'decode(...)' -> un-standardization in ActionIdentifier.forward().
    """
    from action_extractor.architectures.direct_resnet_mlp import ActionExtractionResNet
    from action_extractor.architectures.direct_variational_resnet import (
        ActionExtractionVariationalResNet,
        ActionExtractionHypersphericalResNet,
        ActionExtractionSLAResNet
    )
    from .action_identifier import (
        ActionIdentifier, VariationalEncoder, VMFEncoder
    )
    import torch
    import torch.nn as nn

    # 1) Instantiate the correct model class
    if arch_type == 'variational':
        model_class = ActionExtractionVariationalResNet
    elif arch_type == 'hyperspherical':
        model_class = ActionExtractionHypersphericalResNet
    elif arch_type == 'sla':
        model_class = ActionExtractionSLAResNet
    else:
        # default non-variational
        model_class = ActionExtractionResNet

    model = model_class(
        resnet_version=resnet_version,
        video_length=video_length,
        in_channels=in_channels,
        action_length=action_length,
        num_classes=num_classes,
        num_mlp_layers=num_mlp_layers,
    )

    # 2) If checkpoint_path is provided, load entire model state
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_key = k[len("module."):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=True)

    # 3) Otherwise, do partial param loading (if provided)
    else:
        if conv_path is not None:
            model.conv.load_state_dict(torch.load(conv_path, map_location='cpu'))
        if mlp_path is not None and hasattr(model, 'mlp'):
            model.mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
        if hasattr(model, 'fc_mu') and fc_mu_path is not None:
            model.fc_mu.load_state_dict(torch.load(fc_mu_path, map_location='cpu'))
        if hasattr(model, 'fc_logvar') and fc_logvar_path is not None:
            model.fc_logvar.load_state_dict(torch.load(fc_logvar_path, map_location='cpu'))
        if hasattr(model, 'fc_kappa') and fc_kappa_path is not None:
            model.fc_kappa.load_state_dict(torch.load(fc_kappa_path, map_location='cpu'))

        # SLA optional c or c_mu/logvar
        if arch_type == 'sla':
            if use_distribution_for_c:
                if hasattr(model, 'fc_c_mu') and fc_c_mu_path is not None:
                    model.fc_c_mu.load_state_dict(torch.load(fc_c_mu_path, map_location='cpu'))
                if hasattr(model, 'fc_c_logvar') and fc_c_logvar_path is not None:
                    model.fc_c_logvar.load_state_dict(torch.load(fc_c_logvar_path, map_location='cpu'))
            else:
                if hasattr(model, 'fc_c') and fc_c_path is not None:
                    model.fc_c.load_state_dict(torch.load(fc_c_path, map_location='cpu'))
            if hasattr(model, 'fc_gripper') and fc_gripper_path is not None:
                model.fc_gripper.load_state_dict(torch.load(fc_gripper_path, map_location='cpu'))

    # -------------------------------------------------------------------------
    # 4) Now return an ActionIdentifier. We'll unify the logic so that for
    #    ActionExtractionResNet, we ALWAYS do the "encoder/decoder" split,
    #    whether we loaded from checkpoint or partial param. This ensures
    #    we run "decode(...)" -> un-standardize in ActionIdentifier.
    # -------------------------------------------------------------------------
    if isinstance(model, ActionExtractionResNet):
        # --- Non-variational => do the old sequential split ---
        encoder_layers = nn.Sequential()
        decoder_layers = nn.Sequential()
        add_to_decoder = False
        for name, module in model.conv.named_children():
            if not add_to_decoder:
                encoder_layers.add_module(name, module)
                if name == split_layer:
                    add_to_decoder = True
            else:
                decoder_layers.add_module(name, module)

        # Combine into final "decoder"
        from torch import nn as nn2
        full_decoder = nn.Sequential(
            decoder_layers,
            nn2.Flatten(),
            model.mlp
        )

        return ActionIdentifier(
            encoder=encoder_layers,
            decoder=full_decoder,
            stats_path=stats_path,
            coordinate_system=coordinate_system,
            camera_name=camera_name,
            deterministic=deterministic
        )

    elif isinstance(model, ActionExtractionVariationalResNet):
        encoder = VariationalEncoder(
            conv=model.conv,
            fc_mu=model.fc_mu,
            fc_logvar=model.fc_logvar
        )
        decoder = model.mlp
        return ActionIdentifier(
            encoder=encoder,
            decoder=decoder,
            stats_path=stats_path,
            coordinate_system=coordinate_system,
            camera_name=camera_name,
            deterministic=deterministic
        )

    elif isinstance(model, ActionExtractionHypersphericalResNet) or isinstance(model, ActionExtractionSLAResNet):
        # vMF or SLA path
        from .action_identifier import VMFEncoder
        fc_c = getattr(model, 'fc_c', None)
        fc_c_mu = getattr(model, 'fc_c_mu', None)
        fc_c_logvar = getattr(model, 'fc_c_logvar', None)
        fc_gripper = getattr(model, 'fc_gripper', None)
        encoder = VMFEncoder(
            conv=model.conv,
            fc_mu=model.fc_mu,
            fc_kappa=model.fc_kappa,
            fc_c=fc_c,
            fc_c_mu=fc_c_mu,
            fc_c_logvar=fc_c_logvar,
            fc_gripper=fc_gripper,
            use_distribution_for_c=use_distribution_for_c
        )
        decoder = model.mlp
        return ActionIdentifier(
            encoder=encoder,
            decoder=decoder,
            stats_path=stats_path,
            coordinate_system=coordinate_system,
            camera_name=camera_name,
            deterministic=deterministic
        )
    else:
        # fallback => treat entire model as single forward
        return ActionIdentifier(
            encoder=model,
            decoder=None,
            stats_path=stats_path,
            coordinate_system=coordinate_system,
            camera_name=camera_name,
            deterministic=deterministic
        )



def validate_pose_estimator(args):
    architecture = 'direct_resnet_mlp'
    data_modality = 'cropped_rgbd+color_mask'
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100

    resnet_layers_num = 18
    action_type = 'delta_position+gripper'
    video_length = 2

    model = load_model(
        architecture,
        horizon=video_length,
        results_path='',
        latent_dim=0,
        motion=False,
        image_plus_motion=False,
        num_mlp_layers=3, # to be extracted
        vit_patch_size=0, 
        resnet_layers_num=resnet_layers_num, # to be extracted
        idm_model_name='',
        fdm_model_name='',
        freeze_idm=None,
        freeze_fdm=None,
        action_type=action_type,
        data_modality=data_modality # to be extracted
        )
    
    trained_model = load_trained_model(model, args.results_path, args.trained_model_name, device)
    
    validation_set = load_datasets(
        architecture, 
        args.datasets_path, 
        args.datasets_path,
        train=False,
        validation=True,
        horizon=1,
        demo_percentage=0.9,
        cameras=['frontview_image'],
        motion=False,
        image_plus_motion=False,
        action_type='pose',
        data_modality=data_modality
        )
    
    from action_extractor.utility_scripts.validation_visualization import validate_and_record
    validate_and_record(trained_model, validation_set, args.trained_model_name[:-4], batch_size, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load model for pose estimation")
    parser.add_argument(
        '--trained_model_name', '-mn', 
        type=str, 
        default='direct_resnet_mlp_res18_optadam_lr0.001_mmt0.9_18_coffee_pose_std_rgb_resnet-20-1559.pth', 
        help='trained model to load'
    )
    parser.add_argument(
        '--results_path', '-rp', 
        type=str, 
        default='/home/yilong/Documents/action_extractor/results',
        help='Path to where the results should be stored'
    )
    parser.add_argument(
        '--datasets_path', '-dp', 
        type=str, 
        default='/home/yilong/Documents/ae_data/datasets/mimicgen_core/coffee_rel',
        help='Path to where the datasets are stored'
    )
    args = parser.parse_args()
    
    validate_pose_estimator(args)
