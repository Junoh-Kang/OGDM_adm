import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, EncoderUNetModel
from .stylegan2.networks import Discriminator
NUM_CLASSES = 1000

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

def model_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        image_size              = 64,
        num_channels            = 128,
        num_res_blocks          = 2,
        num_heads               = 4,        
        num_head_channels       = -1,
        num_heads_upsample      = -1,        
        attention_resolutions   = "16, 8",
        channel_mult            = "",
        dropout                 = 0.0,
        class_cond              = False,
        use_checkpoint          = False,
        use_scale_shift_norm    = True,
        resblock_updown         = False,
        use_fp16                = False,
        use_new_attention_order = False,
    )

def discriminator_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        c_dim                   = 0,        # Conditioning label (C) dimensionality.
        t_dim                   = 0,        # Diffusion timestep dimensionality
        img_resolution          = 64,      # Input resolution.
        img_channels            = 3,        # Number of input color channels.
        architecture            = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base            = 32768,    # Overall multiplier for the number of channels.
        channel_max             = 512,      # Maximum number of channels in any layer.
        num_fp16_res            = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp              = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim                = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs            = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs          = {},       # Arguments for MappingNetwork.
        epilogue_kwargs         = {},       # Arguments for DiscriminatorEpilogue.
    )

def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    skip_type="uniform",   # uniform or quad
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing, skip_type),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )

def create_discriminator(
    c_dim=0,
    t_dim=0,
    image_size=64,         
    img_channels=3,
    architecture='resnet',
    channel_base=32768,
    channel_max=512,
    num_fp16_res=0,
    conv_clamp=None,
    cmap_dim=None,
    block_kwargs={},       
    mapping_kwargs={},      
    epilogue_kwargs={},
):
    return Discriminator(
        c_dim=c_dim,
        t_dim=t_dim,     
        img_resolution=image_size,   
        img_channels=img_channels,   
        architecture=architecture,
        channel_base=channel_base,   
        channel_max=channel_max,
        num_fp16_res=num_fp16_res,   
        conv_clamp=conv_clamp,
        cmap_dim=cmap_dim,     
        block_kwargs=block_kwargs,
        mapping_kwargs=mapping_kwargs,
        epilogue_kwargs=epilogue_kwargs,
    )



def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        if k in ['lr_model', 'lr_disc']:
            v_type = float
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")