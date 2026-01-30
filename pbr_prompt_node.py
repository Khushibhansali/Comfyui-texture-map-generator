print("✅ comfyui-texture-map-generator loaded")

import torch
import numpy as np
from PIL import Image, ImageFilter

# --- Comfy image format helpers ---
# Comfy IMAGE = float32 numpy/tensor in [0..1], shape [B,H,W,C] typically C=3
def pil_to_comfy(img: Image.Image):
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr[None, ...]  # [1,H,W,3]
    return torch.from_numpy(arr)

def pil_gray_to_comfy(img: Image.Image):
    # convert 1-channel to 3-channel (repeat) because most Comfy workflows expect RGB IMAGE
    g = img.convert("L")
    arr = np.asarray(g).astype(np.float32) / 255.0
    arr = np.repeat(arr[..., None], 3, axis=2)  # [H,W,3]
    arr = arr[None, ...]  # [1,H,W,3]
    return torch.from_numpy(arr)

def comfy_to_pil(img_tensor: torch.Tensor):
    # expects [B,H,W,3]
    t = img_tensor[0].detach().cpu().clamp(0, 1).numpy()
    arr = (t * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def approx_ao_from_height(height_pil: Image.Image, blur_radius=6, strength=1.25):
    """
    Simple AO approximation from height:
    - blur height
    - compare blurred vs original to estimate cavities
    """
    h = height_pil.convert("L")
    blurred = h.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    h_np = np.asarray(h).astype(np.float32) / 255.0
    b_np = np.asarray(blurred).astype(np.float32) / 255.0

    cavities = np.clip(b_np - h_np, 0.0, 1.0)  # where original is "lower" than surroundings
    ao = 1.0 - np.clip(cavities * strength, 0.0, 1.0)

    ao_u8 = (ao * 255.0).astype(np.uint8)
    return Image.fromarray(ao_u8, mode="L")


class PromptToFullPBRStableMaterials:
    """
    Prompt -> Albedo/Basecolor, Normal, Roughness, Metallic, AO, Displacement
    Uses gvecchio/StableMaterials Diffusers pipeline.
    """

    _pipe = None
    _device = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Old rusty metal bars with peeling paint"}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 80}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "tileable": ("BOOLEAN", {"default": True}),
                "ao_blur_radius": ("INT", {"default": 6, "min": 0, "max": 32}),
                "ao_strength": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 5.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "normal", "roughness", "metallic", "ao", "displacement")
    FUNCTION = "run"
    CATEGORY = "Texture/PBR"

    def _get_pipe(self):
        if self.__class__._pipe is not None:
            return self.__class__._pipe

        from diffusers import DiffusionPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = DiffusionPipeline.from_pretrained(
            "gvecchio/StableMaterials",
            trust_remote_code=True,          # required by model card
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
        pipe = pipe.to(device)

        self.__class__._pipe = pipe
        self.__class__._device = device
        return pipe

    def run(self, prompt, steps, guidance_scale, seed, tileable, ao_blur_radius, ao_strength):
        pipe = self._get_pipe()
        device = self.__class__._device

        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        # StableMaterials returns a "material" object containing multiple maps
        out = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            tileable=bool(tileable),
            num_images_per_prompt=1,
            generator=generator,
        )

        material = out.images[0]
        # attributes per model card: basecolor, normal, height, roughness, metallic :contentReference[oaicite:2]{index=2}
        basecolor_pil = material.basecolor
        normal_pil = material.normal
        height_pil = material.height
        roughness_pil = material.roughness
        metallic_pil = material.metallic

        ao_pil = approx_ao_from_height(height_pil, blur_radius=int(ao_blur_radius), strength=float(ao_strength))

        # Convert to Comfy IMAGE tensors
        albedo = pil_to_comfy(basecolor_pil)
        normal = pil_to_comfy(normal_pil)

        roughness = pil_gray_to_comfy(roughness_pil)
        metallic = pil_gray_to_comfy(metallic_pil)
        displacement = pil_gray_to_comfy(height_pil)
        ao = pil_gray_to_comfy(ao_pil)

        return (albedo, normal, roughness, metallic, ao, displacement)


NODE_CLASS_MAPPINGS = {
    "PromptToFullPBRStableMaterials": PromptToFullPBRStableMaterials
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptToFullPBRStableMaterials": "Prompt → Full PBR (StableMaterials)"
}
