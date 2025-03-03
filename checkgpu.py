import torch
import subprocess

def check_gpu():
    print("=== Checking PyTorch & GPU Info ===")
    print("PyTorch version:", torch.__version__)
    
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA is NOT available. This means PyTorch only sees the CPU.")
        print("Make sure you installed a CUDA-enabled version of PyTorch, and that your drivers match.\n")
        return
    
    # If we reach here, PyTorch sees at least one GPU
    gpu_count = torch.cuda.device_count()
    print(f"\nCUDA is available. Number of detected GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i} Name: {torch.cuda.get_device_name(i)}")

    # Quick GPU operation test
    try:
        print("\n=== Running a small test on the GPU ===")
        x = torch.randn((1000, 1000), device='cuda')
        y = torch.randn((1000, 1000), device='cuda')
        z = torch.matmul(x, y)  # matrix multiply
        print("Matrix multiplication succeeded on the GPU!")
        print("Result shape:", z.shape)
    except Exception as e:
        print("[ERROR] Could not run a GPU operation. Details:")
        print(e)
        return
    
    # Show nvidia-smi
    print("\n=== nvidia-smi output ===")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except Exception as e:
        print("[WARNING] Could not run nvidia-smi. Details:")
        print(e)

if __name__ == "__main__":
    check_gpu()
