import os 
import torch 
import click
import numpy as np
import random 
from PIL import Image
import torch.distributed as dist
from models import PixArt_alpha
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()
def generate(net, latents, encoder_hidden_states, mask, dtype, sigma_init: float = 2.5):
    # Adjust noise levels based on what's supported by the network.
    net = net.to(dtype)
    sigma_init = torch.tensor([sigma_init], dtype=dtype, device='cuda').view(1, 1, 1, 1).repeat(latents.shape[0], 1, 1, 1)
    x = latents.to(dtype) * sigma_init
    
    samples = net(
        x=x, 
        encoder_hidden_states=encoder_hidden_states.to(dtype), 
        sigma=sigma_init, 
        mask=mask,
    )

    return samples 


def batchify(lst, batch_size):
    # split by rank
    local_lst = lst[dist.get_rank()::dist.get_world_size()]
    for i in range(0, len(local_lst), batch_size):
        yield lst[i:i+batch_size] 
        

def dist_init():
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29501')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    dist.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK')))
    
    
    
@click.command(name='One-Step Inference')
@click.option('--dit_model_path', type=click.Path(exists=True), default='/huangzemin/lute/distill-base/models/dit.pth')
@click.option('--text_enc_path', type=click.Path(exists=True), default='/huangzemin/lute/distill-base/models/text-encoder.pth')
@click.option('--vae_path', type=click.Path(exists=True), default='/huangzemin/lute/distill-base/models/vae.pth')
@click.option('--prompt', type=str, default='a astronaut in a jungle')
@click.option('--output_dir', type=click.Path(), default='/huangzemin/lute/distill-base/models/vae.pth')
@click.option('--batch', type=click.IntRange(min=1), default=16)
@click.option('--seed', type=click.INT, default=112)
@click.option('--dtype', type=str, default='bf16')
@click.option('--device', type=str, default='cuda')
@click.option('--init_sigma', type=click.FLOAT, default=2.5)

def main(**kwargs):
    dist_init()
    
    os.makedirs(kwargs['output_dir'], exist_ok=True)
    # set seed
    random.seed(kwargs['seed'])
    np.random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    
    # set device & dtype
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[kwargs['dtype']]
    device = kwargs['device']
    
    pipe = PixArt_alpha(
        img_channels=4,
        C_1=0.001,
        C_2=0.008,
        M=1000,
        beta_start=0.0001,
        beta_end=0.02,
        dit_model_path=kwargs['dit_model_path'],
        text_enc_path=kwargs['text_enc_path'],
        vae_path=kwargs['vae_path'],
        input_size=64
    ).to(device)
    
    pipe.text_encoder.to(device, dtype=dtype)
    pipe.vae.to(device, dtype=dtype)
    
    # read prompts
    if os.path.exists(kwargs['prompt']):
        print("reading prompts from file...")
        with open(kwargs['prompt'], 'r') as f:
            prompts = f.readlines()
    else:
        prompts = [kwargs['prompt']]
    
    image_ix = [i for i in range(len(prompts))]
    batch_image_ix = batchify(image_ix, kwargs['batch'])
    batch_prompts = batchify(prompts, kwargs['batch'])
    
    for local_ix, batch in tqdm(zip(batch_image_ix, batch_prompts), disable=dist.get_rank() != 0):
        encoder_hidden_states, masks = pipe.encode_prompts(batch)
        
        latents = torch.randn([len(batch), 4, 64, 64]).to(device)
        latents = generate(pipe, latents, encoder_hidden_states, masks, dtype, sigma_init=kwargs['init_sigma'])
        
        images = pipe.decode_latents(latents)
        
        for ix, img in zip(local_ix, images):
            image_np = (img * 255).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            pil_img = Image.fromarray(image_np, 'RGB')
            pil_img.save(os.path.join(kwargs['output_dir'], f'{ix}.png'))
            with open(os.path.join(kwargs['output_dir'], f'{ix}.txt'), 'w') as f:
                f.write(batch[ix])


if __name__ == '__main__':
    main()