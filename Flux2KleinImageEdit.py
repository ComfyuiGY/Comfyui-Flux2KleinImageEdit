import node_helpers
import comfy.utils
import math
import torch
import comfy.model_management


class Flux2KleinImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP模型，用于编码文本和图像"}),
                "vae": ("VAE", {"tooltip": "VAE模型，用于编码图像"}),
                "inputcount": (["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"], {"default": "1", "tooltip": "图像输入数量"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8, "tooltip": "输出图像宽度"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8, "tooltip": "输出图像高度"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "批量大小"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "正向提示词"}),
                "image_1": ("IMAGE", {"tooltip": "参考图像1"}),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Flux2 Klein 图像编辑 - 动态增减图像输入（1-20张）"

    def encode(self, clip, vae, inputcount, width, height, batch_size, prompt,
               image_1=None, unique_id=None, **kwargs):
        
        if vae is None:
            raise RuntimeError("VAE是必需的，请连接VAE加载器。")
        
        # 将字符串转换为整数
        inputcount = int(inputcount)
        
        # 收集所有图像：image_1 加上动态传入的 image_2..image_N
        images = []
        if image_1 is not None:
            images.append(image_1)
        for i in range(2, inputcount + 1):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                images.append(img)
        
        # 如果没有图像输入，返回空条件
        if not images:
            negative_tokens = clip.tokenize("")
            negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
            device = comfy.model_management.get_torch_device()
            dummy_pixels = torch.zeros(1, height, width, 3, device=device)
            empty_latent = vae.encode(dummy_pixels)
            latent = {"samples": empty_latent}
            return (negative_conditioning, negative_conditioning, latent)
        
        ref_latents = []
        vl_images = []
        image_prompt_prefix = ""
        
        for i, image in enumerate(images):
            # 准备VL图像
            samples = image.movedim(-1, 1)
            current_total = samples.shape[3] * samples.shape[2]
            
            vl_total = int(384 * 384)
            vl_scale_by = math.sqrt(vl_total / current_total)
            vl_width = round(samples.shape[3] * vl_scale_by)
            vl_height = round(samples.shape[2] * vl_scale_by)
            
            s_vl = comfy.utils.common_upscale(samples, vl_width, vl_height, "area", "center")
            vl_image = s_vl.movedim(1, -1)
            vl_images.append(vl_image)
            
            image_prompt_prefix += f"image{i+1}: <|vision_start|><|image_pad|><|vision_end|> "
            
            # 准备VAE输入canvas
            vae_input_canvas = torch.zeros(
                (samples.shape[0], height, width, 3),
                dtype=samples.dtype,
                device=samples.device
            )
            
            resized_img = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
            resized_img = resized_img.movedim(1, -1)
            
            img_h, img_w = resized_img.shape[1], resized_img.shape[2]
            vae_input_canvas[:, :img_h, :img_w, :] = resized_img
            
            ref_latent = vae.encode(vae_input_canvas)
            ref_latents.append(ref_latent)
        
        full_prompt = image_prompt_prefix + prompt
        
        tokens = clip.tokenize(full_prompt, images=vl_images)
        positive_conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if ref_latents:
            positive_conditioning = node_helpers.conditioning_set_values(
                positive_conditioning, {"reference_latents": ref_latents}, append=True
            )
        
        negative_tokens = clip.tokenize("")
        negative_conditioning = clip.encode_from_tokens_scheduled(negative_tokens)
        
        if ref_latents:
            negative_conditioning = node_helpers.conditioning_set_values(
                negative_conditioning, {"reference_latents": ref_latents}, append=True
            )
        
        device = comfy.model_management.get_torch_device()
        dummy_pixels = torch.zeros(1, height, width, 3, device=device)
        empty_latent = vae.encode(dummy_pixels)
        
        latent = {"samples": empty_latent}
        
        if ref_latents:
            latent["samples"] = ref_latents[0]
            
        if batch_size > 1:
            positive_conditioning = positive_conditioning * batch_size
            negative_conditioning = negative_conditioning * batch_size
            
            samples = latent["samples"]
            if samples.shape[0] != batch_size:
                target_shape = [batch_size] + [1] * (samples.dim() - 1)
                samples = samples.repeat(*target_shape)
            latent["samples"] = samples
        
        return (positive_conditioning, negative_conditioning, latent)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinImageEdit": Flux2KleinImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinImageEdit": "Flux2 Klein 图像编辑",
}