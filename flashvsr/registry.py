from diffsynth.models import ModelManager
from .models.wan_video_dit import WanModel
from .models.wan_video_text_encoder import WanTextEncoder
from .models.wan_video_image_encoder import WanImageEncoder
from .models.wan_video_vae import WanVideoVAE
from .models.wan_video_motion_controller import WanMotionControllerModel
from .models.wan_video_vace import VaceWanModel

def register_wan_models():
    """
    Registers Wan models with the diffsynth ModelManager configuration.
    This allows ModelManager to recognize and load these models.
    """
    from diffsynth.configs import model_config
    
    # Add Wan models to model_loader_configs if not present
    wan_configs = [
        (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
        (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
        (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
        (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
        (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
        (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
        (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
        (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
        (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
        (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
        (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
        (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
        (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
        (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    ]
    
    # We append these to the global config list in diffsynth
    # This is a runtime patch
    model_config.model_loader_configs.extend(wan_configs)
    
    # Re-initialize ModelDetectorFromSingleFile in ModelManager if it was already instantiated
    # This is tricky because ModelManager is usually instantiated by the user.
    # The user should call register_wan_models() BEFORE instantiating ModelManager.
