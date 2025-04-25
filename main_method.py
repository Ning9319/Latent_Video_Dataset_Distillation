import datetime
import os
import argparse
import numpy as np
import torch
import random
import wandb
import warnings
from einops import rearrange
from tqdm import tqdm, trange
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, TensorDataset, preload_test_data
from quantize_vae import use_quantized_vae
from dquantize_3dvae import use_quantized_3dvae
from sklearn.cluster import KMeans
import tensorly as tl
from tensorly.decomposition import tucker
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from dppy.finite_dpps import FiniteDPP



warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    # Set the seed
    np.random.seed(args.random_state)

    torch.cuda.set_device(0)  # Ensure it uses the correct device
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader= get_dataset(args.dataset, args.data_path)

    if args.preload:
        print("Preloading dataset")
        video_all = []
        label_all = []
        for i in trange(len(dst_train)):
            _ = dst_train[i]
            video_all.append(_[0])
            label_all.append(_[1])
        
        video_all = torch.stack(video_all)
        label_all = torch.tensor(label_all)
        dst_train = torch.utils.data.TensorDataset(video_all, label_all)
    
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    print("Eval mode is, ", args.eval_mode)


    ''' organize the real dataset '''
    labels_all = label_all if args.preload else dst_train.labels
    indices_class = [[] for c in range(num_classes)]

    
    print("BUILDING DATASET")
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    

    def get_images(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if n == 1:
            imgs = video_all[idx_shuffle[0]].unsqueeze(0)
        else:
            imgs = video_all[idx_shuffle]
        return imgs.to(args.device)

    # Getting the all videos into latent space

    if args.latent_file == None:
        if args.vae_model == '2DVAE':
            vae = use_quantized_vae().to(args.device)
            vae.requires_grad_(False)
            N = len(dst_train)
            
            img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(args.device)
            img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(args.device)
            encode_batch_size = 8
            num_batches = N // encode_batch_size + (1 if N % encode_batch_size > 0 else 0)
            video_latent = []
            print("\nEncoding the real videos into the latent space with 2DVAE\n")
            for i in trange(num_batches):
                batch = []
                for j in range(i * encode_batch_size, min((i+1) * encode_batch_size, N)):
                    batch.append(dst_train[j][0].unsqueeze(0))  # Index individually and unsqueeze

                batch = torch.cat(batch, dim=0)

                batch = rearrange(batch, "b t c h w -> (b t) c h w") # Merge batch & frames
                batch = batch.to(args.device)

                batch = batch * img_std + img_mean  # Convert back to [0,1]
                batch = batch * 2 - 1  # Convert to [-1,1]

                latents = vae.encode(batch).latent_dist.sample().cpu()  # Move to CPU
                video_latent.append(latents) 
            
            video_all = torch.cat(video_latent, dim=0)
            video_all = rearrange(video_all, "(b t) c h w -> b t c h w", b=N)
            print("The tensor in the latent space with size:", video_all.shape)  # [N, T, C, H, W]
            

        elif args.vae_model == '3DVAE':
            vae = use_quantized_3dvae().to(args.device)
            vae.requires_grad_(False)
            
            N = len(dst_train)
            print(f"{N} videos")

            
            img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(args.device)
            img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(args.device)
            encode_batch_size = args.encode_batch_size
            num_batches = N // encode_batch_size + (1 if N % encode_batch_size > 0 else 0)
            video_latent = []
            print("\nEncoding the real videos into the latent space with 3DVAE\n")
            for i in trange(num_batches):
                batch = []
                for j in range(i * encode_batch_size, min((i+1) * encode_batch_size, N)):
                    batch.append(dst_train[j][0].unsqueeze(0))  # Index individually and unsqueeze
                
                batch = torch.cat(batch, dim=0)

                batch = rearrange(batch, 'b t c h w -> b c t h w').half() # Rearrange the frames and channels then half
                batch = batch.to(args.device)

                batch = batch * img_std + img_mean  # Convert back to [0,1]
                batch = batch * 2 - 1  # Convert to [-1,1]

                latents = vae.encode(batch).latent_dist.sample().cpu()  # Move to CPU
                video_latent.append(latents)
            video_all = torch.cat(video_latent, dim=0)
            video_all = rearrange(video_all, 'b c t h w -> b t c h w').to(torch.float32)
            print("The tensor in the latent space with size:", video_all.shape)  # [N, T, C, H, W]

            os.makedirs("./SSv2_latent_tensor", exist_ok=True)
            torch.save(video_all, os.path.join("./SSv2_latent_tensor", "video_latent_3d.pt"))

        else:
            pass
    else:
        print(f"Loading video_all from the file {args.latent_file}")

        video_all = torch.load(args.latent_file)
        print("Loaded latent tensor shape:", video_all.shape)

        if args.vae_model == '2DVAE':
            vae = use_quantized_vae().to(args.device)
            img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(args.device)
            img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(args.device)
        else:
            vae = use_quantized_3dvae().to(args.device)
            img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(args.device)
            img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(args.device)

        
        vae.requires_grad_(False)
        encode_batch_size = args.encode_batch_size
        


    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []



    project_name = f"Latent_Video_{args.dataset}_{args.method}_{args.vae_model}"

    
    wandb.init(sync_tensorboard=False,
               project=project_name,
               job_type="CleanRepo",
               config=args,
               name = f'{args.dataset}_ipc{args.ipc}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
               )


    args = type('', (), {})()


    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc
    

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    
    image_syn = torch.randn(size=(num_classes*args.ipc, args.frames, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=False, device=args.device)

    label_syn = torch.tensor(np.stack([np.ones(args.ipc)*i for i in range(0, num_classes)]), dtype=torch.long, requires_grad=False,device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    


    
    if args.select_mode == 'full':
        # Use the entire dataset as the synthetic data
        image_syn = video_all.detach().clone().to(args.device)
        label_syn = labels_all.detach().clone().to(args.device)

    elif args.select_mode == 'kmeans':
        num_samples = args.ipc  # Total number of samples to select per class
        latent_features_np = video_all.reshape(video_all.shape[0], -1).cpu().numpy()  # Flatten features

        selected_indices = []

        for class_id in trange(num_classes):
            class_mask = (labels_all == class_id)
            class_latents = latent_features_np[class_mask.cpu().numpy()]  # Extract class-specific features
            class_indices = np.where(class_mask.cpu().numpy())[0]  # Original indices for this class

            # If not enough samples, take all available
            if len(class_latents) <= num_samples:
                sampled_indices = class_indices  # Take all available
            else:
                # Apply K-Means to cluster the class-specific samples
                num_subclusters = args.num_clusters
                kmeans = KMeans(n_clusters=num_subclusters, random_state=args.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(class_latents)

                # Store cluster indices
                cluster_indices_list = [np.where(cluster_labels == cluster)[0] for cluster in range(num_subclusters)]
                
                # Determine how many samples per cluster
                base_samples_per_cluster = num_samples // num_subclusters
                remainder = num_samples % num_subclusters  # Distribute remainder among some clusters

                cluster_sample_counts = [base_samples_per_cluster] * num_subclusters
                for i in range(remainder):
                    cluster_sample_counts[i] += 1  # Distribute the remainder

                # Select samples from each cluster, ensuring exactly num_samples total
                sampled_indices = []
                deficit = 0  # Track missing samples if some clusters are too small

                for cluster_id, cluster_indices in enumerate(cluster_indices_list):
                    num_to_sample = cluster_sample_counts[cluster_id]

                    if len(cluster_indices) >= num_to_sample:
                        # Sample normally
                        cluster_sample = np.random.choice(cluster_indices, num_to_sample, replace=False)
                    else:
                        # Take all available samples and record the shortfall
                        cluster_sample = cluster_indices
                        deficit += num_to_sample - len(cluster_indices)

                    sampled_indices.extend(cluster_sample)

                # If we are short on samples, get additional samples from larger clusters
                if deficit > 0:
                    extra_needed = deficit
                    for cluster_id, cluster_indices in enumerate(cluster_indices_list):
                        remaining = list(set(cluster_indices) - set(sampled_indices))  # Available for extra sampling
                        if len(remaining) > 0:
                            extra_sample = np.random.choice(remaining, min(len(remaining), extra_needed), replace=False)
                            sampled_indices.extend(extra_sample)
                            extra_needed -= len(extra_sample)
                        if extra_needed == 0:
                            break  # Stop when the shortfall is filled

                # Convert back to original indices
                sampled_indices = class_indices[sampled_indices]

            selected_indices.extend(sampled_indices)

        # Move selected samples to device
        image_syn = video_all[selected_indices].to(args.device)
        label_syn = labels_all[selected_indices].to(args.device)
        print(f"Selected {image_syn.shape} diverse samples using Kmeans.")
          
    elif args.select_mode == 'DAPS':
        num_samples = args.ipc  # Total samples to select per class
        video_all_flat = video_all.reshape(video_all.shape[0], -1).cpu().numpy()


        # Normalize the feature space
        scaler = StandardScaler()
        video_all_flat = scaler.fit_transform(video_all_flat)

        num_classes = len(torch.unique(labels_all))
        selected_indices = []

        print("Starting Diversity-Aware Prototype Selection (DAPS) ....")

        for class_id in tqdm(range(num_classes), desc="Processing Classes"):
            class_mask = labels_all == class_id
            class_videos = video_all_flat[class_mask.cpu().numpy()]
            class_indices = np.where(class_mask.cpu().numpy())[0]

            if len(class_videos) < num_samples:
                selected_indices.extend(class_indices)
                continue

            # Compute pairwise distances (Euclidean or Cosine)
            pairwise_distances = cdist(class_videos, class_videos, metric='euclidean')

            # Convert distances to similarity matrix
            similarity_matrix = np.exp(-pairwise_distances)

            # Apply Determinantal Point Process (DPP) for diversity-aware selection
            dpp = FiniteDPP(kernel_type='likelihood', L=similarity_matrix)

            # Sample from DPP
            dpp.sample_exact_k_dpp(size=num_samples)
            selected = list(dpp.list_of_samples[0])

            selected_indices.extend(class_indices[selected])

        # Final selected dataset
        image_syn = video_all[selected_indices].to(args.device)
        label_syn = labels_all[selected_indices].to(args.device)

        print(f"Selected {image_syn.shape} diverse samples using DAPS.")


    else:
        # Random sampling
        image_syn = torch.randn(size=(num_classes*args.ipc, video_all.shape[-4], video_all.shape[-3], video_all.shape[-2], video_all.shape[-1]), dtype=torch.float, requires_grad=False, device=args.device)

        label_syn = torch.tensor(np.stack([np.ones(args.ipc)*i for i in range(0, num_classes)]), dtype=torch.long, requires_grad=False,device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        if args.init == 'real':
            print('initialize synthetic data from random real images in the latent space')
            for c in range(0, num_classes):
                i = c 
                image_syn.data[i*args.ipc:(i+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

    
    # Different compression techniques
    if args.method == 'PCA':
        # Applying the PCA
        N, T, C, H, W = image_syn.shape
        D = T*C*H*W
        r = int((N * D) / (2 * (N + D)))
        X = image_syn.view(N, -1)

        X_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - X_mean
        
        U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
        PCA_basis = V[:r, :]
        PCA_coeffs = U[:, :r] * S[:r] 

        print("PCA_basis shape:", PCA_basis.shape)
        print("PCA_coeffs shape:", PCA_coeffs.shape)
        pca_dict = {"basis": PCA_basis.cpu(), "coeffs": PCA_coeffs.cpu()}

        save_dir = os.path.join(args.save_path, project_name, wandb.run.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(pca_dict, os.path.join(save_dir, "synthetic_data.pt"))


        X_recon = PCA_coeffs @ PCA_basis 
        X_recon = X_recon + X_mean

        image_syn = X_recon.view(N, T, C, H, W).to(args.device)
        print("Reconstructed latent tensor shape:", image_syn.shape)
    
    elif args.method == 'Tucker':
        # Set tensorly backend to PyTorch
        tl.set_backend('pytorch')

        # Get the shape of image_syn
        N, T, C, H, W = image_syn.shape

        # Compute new ranks based on compression ratio
        new_T = max(1, int(T * args.compress_ratio))
        new_C = C
        new_H = max(1, int(H * args.compress_ratio))
        new_W = max(1, int(W * args.compress_ratio))

        # Define batch size (adjust based on available memory)
        batch_size = 400

        # Storage for decomposed components
        core_list = []
        factors_list = []

        print(f"Applying batch-wise Tucker decomposition with batch size: {batch_size}")

        # Process in batches
        for start in trange(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = image_syn[start:end]  # Extract a batch

            #print(f"Processing batch {start}-{end} with shape {batch.shape}")

            # Apply Tucker decomposition to the batch
            ranks = [end - start, new_T, new_C, new_H, new_W]
            core, factors = tucker(batch, rank=ranks)

            # Store the core and factors
            core_list.append(core.cpu())  # Move to CPU to save memory
            factors_list.append([f.cpu() for f in factors])

        # Save decomposed components
        tucker_dict_cpu = {
            "cores": core_list,
            "factors": factors_list
        }

        save_dir = os.path.join(args.save_path, project_name, wandb.run.name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tucker_dict_cpu, os.path.join(save_dir, "synthetic_data.pt"))

        # Reconstruct the tensor batch-wise
        reconstructed_batches = []
        for i, (core, factors) in enumerate(zip(core_list, factors_list)):
            batch_reconstructed = tl.tucker_to_tensor((core, factors))
            reconstructed_batches.append(batch_reconstructed)

        # Concatenate all reconstructed batches
        image_syn = torch.cat(reconstructed_batches, dim=0).to(args.device)

        print("Reconstructed tensor shape:", image_syn.shape)

        
    else:
        pass





    ''' Evaluate synthetic data '''
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}


    for model_eval in model_eval_pool:
        print('Evaluation\nmodel_eval = %s'%(model_eval))
        accs_test = []
        accs_train = []

        # Loading test data into memory
        test_images, test_labels = preload_test_data(dst_test)
        test_dataset = TensorDataset(test_images, test_labels)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)





        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model

            image_syn_eval, label_syn_eval = image_syn.detach().clone(), label_syn.detach().clone() # avoid any unaware modification

            # Applying the decoder to the image_syn_eval

            if args.vae_model == '2DVAE':
                B = image_syn_eval.shape[0]
                image_syn_eval = rearrange(image_syn_eval, "b t c h w -> (b t) c h w") # Merge batch & frames
            elif args.vae_model == '3DVAE':
                image_syn_eval = rearrange(image_syn_eval, 'b t c h w -> b c t h w')
            else:
                pass


            if args.vae_model == '2DVAE' or args.vae_model == '3DVAE': 
                reconstructed_videos = []
                decode_num_batches = len(image_syn_eval) // encode_batch_size + (1 if len(image_syn_eval) % encode_batch_size > 0 else 0)
                print("Decoding the synthesis video...")
                for i in trange(decode_num_batches):
                    batch = image_syn_eval[i * encode_batch_size : (i + 1) * encode_batch_size]
                    batch = batch.to(args.device)
                    decoded_batch  = vae.decode(batch).sample

                    decoded_batch = ((decoded_batch + 1) / 2).clamp(0, 1) # back in [0,1], still need to be normalized
                    decoded_batch = (decoded_batch - img_mean) / img_std

                    reconstructed_videos.append(decoded_batch)
                image_syn_eval = torch.cat(reconstructed_videos, dim=0)

                if args.vae_model == '2DVAE':
                    image_syn_eval = rearrange(image_syn_eval, "(b f) c h w -> b f c h w", b=B)
                elif args.vae_model == '3DVAE':
                    image_syn_eval = rearrange(image_syn_eval, 'b c t h w -> b t c h w').to(torch.float32)

                    
                    expected_frames = args.frames 

                    if image_syn_eval.shape[1] < expected_frames:  # Check if frames are missing
                        num_missing = expected_frames - image_syn_eval.shape[1]
                        last_frame = image_syn_eval[:, -1:]  # Select the last frame [B, 1, C, H, W]

                        # Repeat the last frame to match expected frames
                        padding_frames = last_frame.repeat(1, num_missing, 1, 1, 1)  # [B, num_missing, C, H, W]
                        image_syn_eval = torch.cat([image_syn_eval, padding_frames], dim=1)  # Concatenate along frame dimension
                    
                else:
                    pass

                del reconstructed_videos, batch, decoded_batch
                torch.cuda.empty_cache()
    
                print("\nThe image_syn_eval has size of", image_syn_eval.data.shape, "\n") # [n, T, C, H, W]
            else:

                print("\nThe image_syn_eval has size of", image_syn_eval.data.shape, "\n")


            _, acc_train, acc_test, acc_per_cls = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, mode='none',test_freq=100)
            accs_test.append(acc_test)
            accs_train.append(acc_train)
            print("acc_per_cls:",acc_per_cls)
            print("acc_test:", acc_test)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
            len(accs_test), model_eval, acc_test_mean, acc_test_std))

        wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean})
        wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]})
        wandb.log({'Std/{}'.format(model_eval): acc_test_std})
        wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]})


    wandb.finish()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')

    parser.add_argument('--method', type=str, default='Kmeans', help='Kmeans')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='use top5 to eval top5 accuracy, use S to eval single accuracy')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')

    parser.add_argument('--lr_net', type=float, default=0.001, help='learning rate for network')
    
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn')

    parser.add_argument('--init', type=str, default='real', choices=['noise', 'real', 'real-all'], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--data_path', type=str, default='distill_utils/data', help='dataset path')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--preload', action='store_true', help='preload dataset')
    parser.add_argument('--save_path',type=str, default='./logged_files', help='path to save')
    parser.add_argument('--frames', type=int, default=16, help='')


    parser.add_argument('--random_state', type=int, default=42, help="Random State")
    parser.add_argument('--vae_model', type=str, default="2DVAE", help="VAE model used for encoding and decoding")
    parser.add_argument("--compress_ratio", type=float, default=0.75, help="The compression ratio used in the HOSVD(Tucker)")
    parser.add_argument('--num_clusters', type=int, default=10, help="Number of clusters used in Kmeans Data Selection")
    parser.add_argument('--select_mode', type=str, default="random", choices=['full', 'kmeans', 'random', 'DAPS'], help="How to sample the videos")
    parser.add_argument('--latent_file', type=str, help='The file path to the saved entire video tensor in the latent space')
    parser.add_argument('--encode_batch_size', type=int, default=4, help="The encoding/decoding batch size")

    

    args = parser.parse_args()

    main(args)

