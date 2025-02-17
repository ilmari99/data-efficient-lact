import argparse
import importlib.util
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from functools import cache
from torchvision.io import read_image, ImageReadMode
import torch as pt
import csv
from utils import calc_phi_coefficient


def get_htc_scan(level = 1, sample = "a"):
    base_path = "./htc2022_data/"
    # htc2022_01a_recon_fbp_seg.png
    htc_file = f"htc2022_0{level}{sample}_recon_fbp_seg.png"
    sinogram_file = f"htc2022_0{level}{sample}_limited_sinogram.csv"
    angle_file = f"htc2022_0{level}{sample}_angles.csv"
    angle_file = os.path.join(base_path, angle_file)
    
    # Load angles
    angles = np.loadtxt(angle_file,dtype=np.str_, delimiter=",")
    angles = np.array([float(angle) for angle in angles])

    # Load the img
    img = read_image(os.path.join(base_path, htc_file))
    img = pt.tensor(img, dtype=pt.float32, device='cuda', requires_grad=False)
    print(f"Sample {level}{sample} rotating to {angles[0]}")
    img = img.squeeze()
    y = img / 255
    
    outer_mask = load_solid_mask()#.cpu().numpy()
    sinogram = np.loadtxt(os.path.join(base_path, sinogram_file), delimiter=',')
    sinogram = pt.tensor(sinogram, dtype=pt.float32, device='cuda') * 255
    
    
    if level == 8:
        sample_to_degrees = {
            "a": 30,
            "b": 30,
            "c": 30,
            "d": 30,
        }
        # Only take a part of the sinogram
        angles = np.arange(0,sample_to_degrees[sample] + 0.5,0.5)
        sinogram = sinogram[0:len(angles),:]
    return y, outer_mask, sinogram, angles

@cache
def load_solid_mask(buffer = 1):
    img = read_image("./solid_disk.png", ImageReadMode.GRAY)
    img = img / 255
    img = img.squeeze()
    #return img.to("cuda")
    buffered_mask = pt.zeros_like(img)
    
    # Use dilation to add buffer around the disk
    kernel = pt.ones((2 * buffer + 1, 2 * buffer + 1), device=img.device)
    buffered_mask = pt.nn.functional.conv2d(img.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=buffer)
    buffered_mask = pt.clamp(buffered_mask, 0, 1).squeeze()
    buffered_mask = buffered_mask.to("cuda")
    return buffered_mask

def main():
    parser = argparse.ArgumentParser(
        description="Call a user-provided 'reconstruct(sinogram, angles)' function on levels 1-8."
    )
    parser.add_argument("file", help="Path to the Python file defining 'reconstruct(sinogram, angles)'.")
    parser.add_argument("--data_dir", default="./htc2022_test_data",
                        help="Directory containing the sinogram and angles CSV files.")
    parser.add_argument("--output_dir", default="./reconstructions",
                        help="Directory to save the reconstruction images.")
    args = parser.parse_args()

    # Dynamically import the module from the provided file path.
    spec = importlib.util.spec_from_file_location("reconstruct_module", args.file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "reconstruct"):
        raise AttributeError("The provided file does not define a function named 'reconstruct'.")

    reconstruct = module.reconstruct

    os.makedirs(args.output_dir, exist_ok=True)

    for level in range(7,0,-1):
        level_sum = 0
        samples = ["a", "b", "c"] if level < 8 else ["a", "b", "c", "d"]
        for sample in samples:
            y, _, sinogram, angles = get_htc_scan(level, sample)
            print(f"Reconstructing {level}{sample}...")
            # Assume reconstruct returns a numpy array.
            #sinogram = sinogram.cpu().numpy()
            
            x = reconstruct(sinogram, angles)
            # Convert outer_mask to a numpy array for element-wise multiplication.
            #outer_mask_np = outer_mask.cpu().numpy() if hasattr(outer_mask, "cpu") else outer_mask
            #x = np.clip(x, 0, 1)
            recon_path = os.path.join(args.output_dir, f"htc2022_0{level}{sample}_recon.png")
            plt.imsave(recon_path, x, cmap="gray")
            # Compute the binary segmentation by thresholding at 0.5.
            y_np = y.cpu().numpy() if hasattr(y, "cpu") else y
            x_bin = (x >= 0.5).astype(np.uint8)
            y_bin = (y_np >= 0.5).astype(np.uint8)
            # Binarize
            mcc_score = calc_phi_coefficient(x_bin, y_bin)
            print(f"MCC score for {level}{sample}: {mcc_score}")
            level_sum += mcc_score
            # Save the MCC score and reconstruction info to a CSV file.
            results_csv = os.path.join(args.output_dir, "results.csv")
            file_exists = os.path.isfile(results_csv)
            with open(results_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(["level", "sample", "MCC"])
                writer.writerow([level, sample, mcc_score])
        print(f"Total MCC score for level {level}: {level_sum}")


if __name__ == "__main__":
    main()
    