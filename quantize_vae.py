from optimum.quanto import freeze, quantize, quantization_map, requantize, qint8
from diffusers import AutoencoderKL
from safetensors.torch import save_file, load_file
from accelerate import init_empty_weights
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype, normalize
from torchvision.utils import save_image
import torch
import json
import psutil
import os

VAE_MODEL = "./vae_weights/2d_vae"

def save_quantized_vae():
    """
    Quantize the VAE model and save it to ./quantized_vae/quantized_vae.safetensors.
    
    This function loads the VAE model, quantizes it to qint8, freezes the model, and saves the 
    quantized model to a safetensor file. It also saves the quantization map to 
    ./quantized_vae/quantization_map.json.
    """
    vae = AutoencoderKL.from_pretrained(VAE_MODEL)
    quantize(vae, weights=qint8, activations=None)
    freeze(vae)
    if not os.path.exists('quantized_vae'):
        os.makedirs('quantized_vae')
    save_file(vae.state_dict(), './quantized_vae/quantized_vae.safetensors')
    with open('./quantized_vae/quantization_map.json', 'w') as f:
        json.dump(quantization_map(vae), f)

def use_quantized_vae():
    """
    if (not (os.path.exists('./quantized_vae/quantized_vae.safetensors') and os.path.exists('./quantized_vae/quantization_map.json'))):
        save_quantized_vae()
    if (not (os.path.exists('./original_vae/config.json'))):
        vae = AutoencoderKL.from_pretrained(VAE_MODEL)
        vae.save_pretrained('./original_vae')
    """

    state_dict = load_file('./quantized_vae/quantized_vae.safetensors')
    with open('./quantized_vae/quantization_map.json', 'r') as f:
        quantization_map = json.load(f)
    with open('./quantized_vae/config.json', 'r') as f:
        vae_config = json.load(f)

    # process = psutil.Process()
    # before = process.memory_info().rss
    # start_mem = torch.cuda.memory_allocated()
    print("Loading an empty model")
    # init_empty_weights is used so that loading an initial vae doesn't take up RAM for initialization
    with init_empty_weights():
        new_vae = AutoencoderKL.from_config(vae_config)

    # get_memory_footprint() is a calculated estimate & will usually just be
    # what size the model would be & not how much memory it actually uses
    # print(new_vae.get_memory_footprint())
    # print(f"CUDA memory allocated: {(torch.cuda.memory_allocated() - start_mem)/1e6:.2f} bytes") 
    # print(f"Total memory used: {(process.memory_info().rss - before)/1e6:.2f} MB")

    # process = psutil.Process()
    # before = process.memory_info().rss
    # start_mem = torch.cuda.memory_allocated()
    print("Initializing the new model with the quantized weights")
    requantize(new_vae, state_dict, quantization_map)
    # print(new_vae.get_memory_footprint())
    # print(f"CUDA memory allocated: {(torch.cuda.memory_allocated() - start_mem)/1e6:.2f} bytes") 
    # print(f"Total memory used: {(process.memory_info().rss - before)/1e6:.2f} MB")
    return new_vae

    
def encode_img(input_img, vae):
    # process = psutil.Process()
    # before = process.memory_info().rss
    # Ensures input has a batch dimension
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        # Change image values from [0, 1] to [-1, 1]
        latent = vae.encode(input_img * 2 - 1)
        # print(f"Memory used for encoding: {(process.memory_info().rss - before)/1e6:.2f} MB")
    return latent.latent_dist.sample()

def decode_img(latents, vae):
    # process = psutil.Process()
    # before = process.memory_info().rss
    # bath of latents -> list of images
    with torch.no_grad():
        image = vae.decode(latents).sample
        # print(f"Memory used for decoding: {(process.memory_info().rss - before)/1e6:.2f} MB")
    # Change image values back from [-1, 1] to [0, 1] 
    image = ((image + 1) / 2).clamp(0, 1)
    image = image.detach()
    return image

def test(imgpath):
    img = Image.open(imgpath)
    img = img.resize((112, 112))
    tensor_image = convert_image_dtype(pil_to_tensor(img), torch.float32)

    # Print tensor details
    print(tensor_image.shape)  # (C, H, W) format
    print(tensor_image.dtype)


    # Original vae
    vae = AutoencoderKL.from_pretrained(VAE_MODEL)
    
    # print("I'm not sure how accurate this memory measurement is: I'll try with CUDA later")
    # print("==================================")
    # print("Original VAE Memory Usage")
    # print("==================================")
    latents = encode_img(tensor_image, vae)
    decoded = decode_img(latents, vae)
    img_name = "decoded_" + os.path.basename(imgpath)
    save_image(decoded, os.path.join(os.path.dirname(imgpath), img_name))

    vaeq = use_quantized_vae()
    # print("==================================")
    # print("Quantized VAE Memory Usage")
    # print("==================================")
    qlatents = encode_img(tensor_image, vaeq)
    qdecoded = decode_img(qlatents, vaeq)
    qimg_name = "q_decoded_" + os.path.basename(imgpath)
    save_image(qdecoded, os.path.join(os.path.dirname(imgpath), qimg_name))


#test("test_images/astronaut.png")
#test("test_images/img_00058.jpg")
