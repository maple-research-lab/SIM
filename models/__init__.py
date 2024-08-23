from .PixArt import PixArt_XL_2

import torch 
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, AutoTokenizer


def ddp_module_name_clear(state_dict):
    final_weight = {}
    for key in state_dict:
        final_weight[key[7:]] = state_dict[key]
    return final_weight


class PixArt_alpha(torch.nn.Module):
    
    def __init__(self, 
        img_channels    = 4,                # Number of color channels.
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        beta_start      = .0001,
        beta_end        = .02,
        dit_model_path  = '',
        text_enc_path   = '',
        vae_path        = '',
        **model_kwargs
    ):
        super().__init__()
        self.img_channels = img_channels
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.beta_start = beta_start
        self.beta_end = beta_end
        model = PixArt_XL_2(**model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (1 - self.beta_t(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)

        # load text encoder
        text_encoder = T5EncoderModel.from_pretrained(text_enc_path)
        self.tokenizer = AutoTokenizer.from_pretrained(text_enc_path)
        text_encoder.eval().requires_grad_(False)
        self.text_encoder = text_encoder
        
        # load vae
        vae = AutoencoderKL.from_pretrained(vae_path)
        vae.eval().requires_grad_(False)
        self.vae = vae

        # load model
        self.model = model
        missing_keys, unexpected_keys = self.load_state_dict(ddp_module_name_clear(torch.load(dit_model_path, map_location='cpu')['ema']), strict=False)
        model.eval().requires_grad_(False)


    @torch.no_grad()
    def encode_prompts(self, prompts, device: str = 'cuda'):
        token_and_mask = self.tokenizer(
            prompts, 
            max_length=120,
            padding="max_length", 
            truncation=True, 
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        input_ids = token_and_mask.input_ids.to(device)
        mask = token_and_mask.attention_mask.to(device)

        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids,
            attention_mask=mask,
        )['last_hidden_state'].detach()
        
        encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1)
        
        return encoder_hidden_states, mask

    @torch.no_grad()
    def decode_latents(self, latents):
        # return B, C, H, W
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def forward(self, x, encoder_hidden_states, sigma, mask, device: str = 'cuda'):
        latents = x 
        rnd_j = self.round_sigma(sigma, return_index=True).reshape(-1,)
        sigma = sigma.reshape(-1, 1, 1, 1)
        
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - rnd_j
        
        F_x = self.model(
            x=c_in * latents, 
            timestep=c_noise,
            y=encoder_hidden_states,
            mask=mask
        )
        D_x = c_skip * latents + c_out * F_x[:, :self.img_channels]

        return D_x

    def beta_t(self, j):
        j = torch.as_tensor(j)
        return self.beta_end + (self.beta_start - self.beta_end) * j / self.M

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1).to(torch.float32)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)
