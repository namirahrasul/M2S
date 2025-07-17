real_path = "./results_celebahq_256_thick_100/celeba/thick/gtImg"  # <-- Replace this with the correct path to your real CelebA-HQ images
gen_path = "./results_celebahq_256_thick_100/celeba/thick/outImg"

# Check folders
import os
def count_images(folder):
    return len([f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')])

print(f" Real images found: {count_images(real_path)}")
print(f"Generated images found: {count_images(gen_path)}")

# Run FID
import subprocess
print("Calculating FID...")
result = subprocess.run(
    ["python", "-m", "pytorch_fid", real_path, gen_path],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("FID Score:", result.stdout.strip())
else:
    print("FID Error:", result.stderr)