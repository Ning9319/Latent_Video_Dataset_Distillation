import torch
import torch.nn as nn
import torch.quantization
import json
import os
from models.modeling_vae import CVVAEModel  # Ensure you're using the same model!
from decord import VideoReader, cpu
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fractions import Fraction


save_dir = "quantized_3dvae"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_quantized_vae():
    # Step 1: Load the original VAE model
    vae3d = CVVAEModel.from_pretrained(
        "./vae_weights/3d_vae",
        subfolder="vae3d",
        torch_dtype=torch.float16  
    ).to('cpu')

    vae3d.requires_grad_(False)  # Freeze weights
    vae3d.eval()  # Set to inference mode
    
    # Step 2: Apply Dynamic Quantization (Only Affects Linear Layers)
    quantized_model = torch.quantization.quantize_dynamic(
        vae3d, {torch.nn.Linear}, dtype=torch.qint8  # Quantizes only Linear layers to INT8
    )



    quantized_model_path = os.path.join(save_dir, "quantized_3dvae.pt")
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"Quantized model saved at {quantized_model_path}")

def use_quantized_3dvae():
    # Step 1: Load Model Configuration
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Step 2: Load the SAME model architecture used for saving
    vae3d = CVVAEModel.from_config(config).to("cpu")  

    # Step 3: Load Quantized Weights
    quantized_model_path = os.path.join(save_dir, "quantized_3dvae.pt")
    state_dict = torch.load(quantized_model_path, map_location="cpu")
    
    #Ensure the keys match before loading
    vae3d.load_state_dict(state_dict, strict=False)
    vae3d.eval()

    return vae3d


def test(vae3d, video_path, save_path, height=576, width=1024, num_frames=16):

    transform = transforms.Compose([
        transforms.Resize(size=(height,width))
    ])
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    video_reader = VideoReader(video_path,ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    total_frames = len(video_reader)

    # Select exactly `num_frames` frames
    if total_frames >= num_frames:
        frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()  # Sample evenly spaced frames
    else:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)  # Pad with last frame

    video = video_reader.get_batch(frame_indices).asnumpy()

    video = rearrange(torch.tensor(video),'t h w c -> t c h w')

    video = transform(video)

    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).half()

    frame_end = 1 + (len(video_reader) -1) // 4 * 4

    video = video / 127.5 - 1.0

    video= video[:,:,:frame_end,:,:]

    video = video.cuda()

    print(f'Shape of input video: {video.shape}')
    latent = vae3d.encode(video).latent_dist.sample()

    print(f'Shape of video latent: {latent.shape}')

    results = vae3d.decode(latent).sample
    print(f'Shape of video reconstructed: {results.shape}')

    results = rearrange(results.squeeze(0), 'c t h w -> t h w c')

    results = (torch.clamp(results,-1.0,1.0) + 1.0) * 127.5
    results = results.to('cpu', dtype=torch.uint8)

    fps = float(fps)
    fps = Fraction(fps).limit_denominator()
    write_video(save_path, results,fps=fps,options={'crf': '10'})

# Run the saving process
# save_quantized_vae()

# Load and move to CUDA for inference
#vae3d = use_quantized_3dvae()
#vae3d = vae3d.to('cuda').half()

#test(vae3d, "./test_images/videos/April_09_brush_hair_u_nm_np1_ba_goo_0.avi", "test_images/videos/recon.avi", 240, 320, 100)