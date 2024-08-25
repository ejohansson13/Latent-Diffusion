from autoencoder import VariationalAutoEncoder
from diffusion import Diffusion
from conditioner import CLIP

import model_weights_conversion

# load model with pre-trained weights
def load_model_weights(
    ckpt_path: str,
    device: str,
):
    state_dict = model_weights_conversion.load_from_model_weights(ckpt_path, device)

    # load autoencoder weights
    autoencoder = VariationalAutoEncoder().to(device)
    autoencoder.load_state_dict(state_dict["autoencoder"], strict=True)

    # load diffusion weights
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    # load CLIP weights
    clip = CLIP()
    clip.load_state_dict(state_dict["conditioner"], strict=True)
    
    return {
        "autoencoder": autoencoder,
        "conditioner": clip,
        "diffusion": diffusion
    }