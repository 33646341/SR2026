
import torch
import os
from models.metric import AdvancedMetrics

def test_metrics():
    print("=== Testing Advanced Metrics ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    # Check torch hub cache dir
    hub_dir = torch.hub.get_dir()
    print(f"Torch Hub Directory: {hub_dir}")
    checkpoints_dir = os.path.join(hub_dir, 'checkpoints')
    if os.path.exists(checkpoints_dir):
        print(f"Checkpoints found in {checkpoints_dir}:")
        print(os.listdir(checkpoints_dir))
    else:
        print(f"Warning: Checkpoints directory {checkpoints_dir} does not exist.")

    try:
        evaluator = AdvancedMetrics(device=device)
    except ImportError as e:
        print(f"Failed to initialize AdvancedMetrics: {e}")
        return

    # Create dummy images: BxCxHxW, range [-1, 1]
    # Use batch size 2 to test batch processing
    B, C, H, W = 2, 3, 256, 256
    print(f"\nCreating dummy input ({B}x{C}x{H}x{W})...")
    input_img = torch.randn(B, C, H, W).to(device).clamp(-1, 1)
    target_img = input_img.clone() 
    # Add some noise to target to make metrics non-perfect
    target_img = target_img + 0.1 * torch.randn_like(target_img)
    target_img = target_img.clamp(-1, 1)
    
    metrics_to_test = ['psnr', 'ssim', 'fsim', 'lpips', 'dists', 'clipiqa']
    
    for metric in metrics_to_test:
        print(f"\nTesting {metric.upper()}...")
        try:
            if metric == 'clipiqa':
                # CLIPIQA is No-Reference
                val = evaluator.compute(input_img, None, metric)
            else:
                val = evaluator.compute(input_img, target_img, metric)
            
            print(f" -> Success! {metric.upper()} Score: {val:.4f}")
        except Exception as e:
            print(f" -> Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_metrics()
