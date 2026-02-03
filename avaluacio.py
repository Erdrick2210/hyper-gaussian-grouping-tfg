import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from skimage.io import imsave


# dataset_solo i experiment per calcular mètriques o visualització test/train per un dataset i experiment concret
dataset_solo = 'birdhouse' # 'basement' 'birdhouse' 'penguin' 'Spec-NeRF' 'bouquet' 'clock' 'fan' 'forestgang1' 'globe'
#experiment = 'ex5' # 'ex1' 'ex2' 'ex3' 'ex4' 'ex5' 'ex6
experiment = 'embedding_24' # 'embedding_8' 'embedding_16' 'embedding_24' 'embedding_32'

embedding_sizes = [8, 16, 24, 32]
experiments = [f'embedding_{size}' for size in embedding_sizes]
#experiments = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6']

base_output = 'output' # path output folder
base_data = 'data' # path dataset folder

iter = 70000
channel = 0  # 0...8 mms-studio i basement, 0..9 penguin, 0..19 Spec-NeRF

datasets_config = {
    'basement': {
        'frames_test': ['0000', '0016', '0024', '0032'],
        'frames_train': ['0018', '0023', '0042'],
        'data_path': 'basement'
    },
    'birdhouse': {
        'frames_test': ['0000', '0016', '0024', '0032'],
        'frames_train': ['0018', '0023', '0042'],
        'data_path': 'multi-modal-studio/birdhouse'
    },
    'bouquet': {
        'frames_test': ['0000', '0016', '0024', '0032'],
        'frames_train': ['0018', '0023', '0042'],
        'data_path': 'multi-modal-studio/bouquet'
    },
    'clock': {
        'frames_test': ['0000', '0016', '0024', '0032'],
        'frames_train': ['0018', '0023', '0042'],
        'data_path': 'multi-modal-studio/clock'
    },
    'fan': {
        'frames_test': ['0000', '0016', '0024', '0032'],
        'frames_train': ['0018', '0023', '0042'],
        'data_path': 'multi-modal-studio/fan'
    },
    'forestgang1': {
        'frames_test': ['0000', '0016', '0024', '0032'],
        'frames_train': ['0018', '0023', '0042'],
        'data_path': 'multi-modal-studio/forestgang1'
    },
    'globe': {
        'frames_test': ['0000', '0016', '0024', '0032'],
        'frames_train': ['0018', '0023', '0042'],
        'data_path': 'multi-modal-studio/globe'
    },
    'penguin': {
        'frames_test': ['0000', '0008', '0016', '0024'],
        'frames_train': ['0005', '0006', '0013'],
        'data_path': 'xnerf/penguin'
    },
    'Spec-NeRF': {
        'frames_test': ['0000', '0008'],
        'frames_train': ['0001', '0004', '0006', '0007'],
        'data_path': 'Spec-NeRF'
    }
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_fn = lpips.LPIPS(net='vgg').to(device)
lpips_fn.eval()


def to_tensor(img):
    # img: (H, W) numpy en [0,1]
    return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

def compute_lpips(img1, img2):
    t1 = to_tensor(img1).to(device)
    t2 = to_tensor(img2).to(device)

    # LPIPS espera [-1, 1]
    t1 = t1 * 2 - 1
    t2 = t2 * 2 - 1

    with torch.no_grad():
        return lpips_fn(t1, t2).item()

def evaluate_experiment(dataset, experiment, frames_test, data_path):
    path_eval = os.path.join(base_output, f'{dataset}_{experiment}', 'eval')
    path_gt = os.path.join(base_data, data_path, 'images')

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for frame in frames_test:
        pred = np.load(os.path.join(path_eval, 'test', f'{frame}_{iter}.npy'))
        gt   = np.load(os.path.join(path_gt,   f'{frame}.npy'))

        C = gt.shape[0]

        for c in range(C):
            gt_c, pred_c = gt[c], pred[c]
            data_range = gt_c.max() - gt_c.min()

            psnr_vals.append(psnr(gt_c, pred_c, data_range=data_range))
            ssim_vals.append(ssim(gt_c, pred_c, data_range=data_range))
            lpips_vals.append(compute_lpips(gt_c, pred_c))

    print('TEST METRICS')
    print('PSNR :', np.mean(psnr_vals))
    print('SSIM :', np.mean(ssim_vals))
    print('LPIPS:', np.mean(lpips_vals))

    return {
        'psnr': np.mean(psnr_vals) if psnr_vals else 0,
        'ssim': np.mean(ssim_vals) if ssim_vals else 0,
        'lpips': np.mean(lpips_vals) if lpips_vals else 0
    }


# Mètriques per un experiment concret
if False:
    print("Dataset:", dataset_solo)
    print("Iteration:", iter)
    print("Experiment:", experiment)
    evaluate_experiment(dataset_solo, experiment, datasets_config[dataset_solo]['frames_test'], datasets_config[dataset_solo]['data_path'])
    torch.cuda.empty_cache()


# Mostrar només mètriques (sense gràfica) per un conjunt d'experiments d'un dataset concret
if False:
    print("Dataset:", dataset_solo)
    print("Iteration:", iter)
    for experiment in experiments:
        print(f"Experiment: {experiment}")   
        evaluate_experiment(dataset_solo, experiment, datasets_config[dataset_solo]['frames_test'], datasets_config[dataset_solo]['data_path'])
        torch.cuda.empty_cache()


# Gràfiques d'embedding, utilitzar quan es vulgui comparar múltiples experiments d'embedding
if False:
    all_results = {}
    for dataset_name, config in datasets_config.items():
        print(f"\nAvaluant dataset: {dataset_name}")
        
        all_results[dataset_name] = []
        
        for experiment in experiments:
            print(f"Experiment: {experiment}")
            
            metrics = evaluate_experiment(
                dataset_name, 
                experiment, 
                config['frames_test'], 
                config['data_path']
            )
            
            all_results[dataset_name].append(metrics)
            
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors per cada dataset
    colors = {
        'basement15': '#2E86AB',
        'birdhouse8': '#A23B72',
        'bouquet': '#F18F01',
        'clock': '#C73E1D',
        'fan': '#6A994E',
        'forestgang1': '#BC4B51',
        'globe': '#8B5A3C',
        'penguin2': '#5E548E',
        'Spec-NeRF': '#E63946'
    }

    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X']

    # PSNR
    for i, (dataset_name, results) in enumerate(all_results.items()):
        axes[0].plot(embedding_sizes, [r['psnr'] for r in results], 
                    marker=markers[i], linewidth=2, markersize=8, 
                    color=colors[dataset_name], label=dataset_name)

    axes[0].set_xlabel('Mida Embedding d', fontsize=12)
    axes[0].set_ylabel('PSNR', fontsize=12)
    axes[0].set_title('PSNR vs Mida Embedding', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(embedding_sizes)

    '''
    # SSIM
    for i, (dataset_name, results) in enumerate(all_results.items()):
        axes[1].plot(embedding_sizes, [r['ssim'] for r in results], 
                    marker=markers[i], linewidth=2, markersize=8, 
                    color=colors[dataset_name], label=dataset_name)

    axes[1].set_xlabel('Mida Embedding d', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs Mida Embedding', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(embedding_sizes)
    axes[1].legend(loc='best', fontsize=10)
    '''

    # LPIPS
    for i, (dataset_name, results) in enumerate(all_results.items()):
        axes[1].plot(embedding_sizes, [r['lpips'] for r in results], 
                    marker=markers[i], linewidth=2, markersize=8, 
                    color=colors[dataset_name], label=dataset_name)

    axes[1].set_xlabel('Mida Embedding d', fontsize=12)
    axes[1].set_ylabel('LPIPS', fontsize=12)
    axes[1].set_title('LPIPS vs Mida Embedding', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(embedding_sizes)

    # Llegenda
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
           ncol=5, fontsize=14, frameon=True, markerscale=1.2)


    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.savefig('embedding_comparison_all_datasets.png', dpi=300, bbox_inches='tight')
    print("\nGràfica guardada: embedding_comparison_all_datasets.png")


# ===== TEST VISUALIZATION FOR CHANNEL =====

if True:
    num_frames_test = len(datasets_config[dataset_solo]['frames_test'])
    plt.figure('test channel {}'.format(channel)), plt.subplot(2, num_frames_test, 1)

    for i, frame_test in enumerate(datasets_config[dataset_solo]['frames_test']):
        test = np.load(os.path.join(base_output, f'{dataset_solo}_{experiment}/eval', 'test', '{}_{}.npy'.format(frame_test, iter)))
        gt = np.load(os.path.join(base_data, datasets_config[dataset_solo]['data_path'], 'images', '{}.npy'.format(frame_test)))
        
        groundtruth = "groundtruth_test_" + frame_test + ".png"
        pred = "prediction_test_" + frame_test + ".png"
        imsave(groundtruth, (gt[channel]*255).astype(np.uint8), cmap='gray')
        imsave(pred, (test[channel]*255).astype(np.uint8), cmap='gray')
        
        plt.subplot(2, num_frames_test, i+1)
        plt.imshow(gt[channel], cmap='gray')
        plt.axis('off')
        plt.title('gt {}'.format(frame_test))
        plt.subplot(2, num_frames_test, i+1+num_frames_test), plt.imshow(test[channel], cmap='gray')
        plt.axis('off')
        plt.title('pred test {}'.format(frame_test))

    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("test_visualization.png")
    plt.close()
    print("Test visualization saved")


# ===== TRAIN VISUALIZATION FOR CHANNEL =====

if False:
    num_frames_train = len(datasets_config[dataset_solo]['frames_train'])
    plt.figure('train channel {}'.format(channel)), plt.subplot(2, num_frames_train, 1)

    for i, frame_train in enumerate(datasets_config[dataset_solo]['frames_train']):
        train = np.load(os.path.join(base_output, f'{dataset_solo}_{experiment}/eval', 'train', '{}_{}.npy'.format(frame_train, iter)))
        gt = np.load(os.path.join(base_data, datasets_config[dataset_solo]['data_path'], 'images', '{}.npy'.format(frame_train)))

        plt.subplot(2, num_frames_train, i+1)
        plt.imshow(gt[channel], cmap='gray')
        plt.axis('off')
        plt.title('gt {}'.format(frame_train))
        plt.subplot(2, num_frames_train, i+1+num_frames_train)
        plt.imshow(train[channel], cmap='gray')
        plt.axis('off')
        plt.title('pred train {}'.format(frame_train))

    plt.tight_layout()
    plt.show(block=False)
    plt.savefig("train_visualization.png")
    plt.close()
    print("Train visualization saved")

plt.close('all')

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Done.")