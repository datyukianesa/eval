import cv2
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips

# ==========================================
# BLOCK 1: METRIC CALCULATORS (Keep these exactly the same)
# ==========================================
def calculate_psnr(gt_img, pred_img):
    return compare_psnr(gt_img, pred_img, data_range=255)

def calculate_ssim(gt_img, pred_img):
    return compare_ssim(gt_img, pred_img, channel_axis=2, data_range=255)

def calculate_lpips(gt_img, pred_img, lpips_model, device):
    gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    gt_tensor = (torch.from_numpy(gt_rgb.transpose(2, 0, 1)).float() / 255.0) * 2.0 - 1.0
    pred_tensor = (torch.from_numpy(pred_rgb.transpose(2, 0, 1)).float() / 255.0) * 2.0 - 1.0
    gt_tensor = gt_tensor.unsqueeze(0).to(device)
    pred_tensor = pred_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        distance = lpips_model(gt_tensor, pred_tensor)
    return distance.item()

# ==========================================
# BLOCK 2: MASTER PIPELINE (Now accepts arguments)
# ==========================================
def run_evaluation(dataset_path, annotations_path, output_csv_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading LPIPS VGG model...")
    lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    lpips_vgg.eval()

    # Convert the string arguments into Path objects
    dataset_dir = Path(dataset_path)
    annotations_dir = Path(annotations_path)
    
    results = []
    count = 0

    print(f"Starting master evaluation on: {dataset_dir}")

    for folder in dataset_dir.iterdir():
        if folder.is_dir():
            folder_name = folder.name
            
            gt_path = folder / "frame2.png"
            anime_path = folder / "frame2_animeinterp.png"
            rife_path = folder / "frame2_rife.png"
            film_path = folder / "frame2_film.png"
            json_path = annotations_dir / folder_name / f"{folder_name}.json"

            if gt_path.exists() and json_path.exists():
                with open(json_path, 'r') as f:
                    ann_data = json.load(f)
                
                difficulty = ann_data.get("level", -1) 
                motion_type = ann_data.get("general_motion_type", "Unknown")
                behavior = ann_data.get("behavior", "Unknown")

                gt_img = cv2.imread(str(gt_path))

                row_data = {
                    "triplet_name": folder_name,
                    "difficulty_level": difficulty,
                    "motion_type": motion_type,
                    "behavior": behavior
                }

                if anime_path.exists():
                    anime_img = cv2.imread(str(anime_path))
                    row_data["AnimeInterp_PSNR"] = calculate_psnr(gt_img, anime_img)
                    row_data["AnimeInterp_SSIM"] = calculate_ssim(gt_img, anime_img)
                    row_data["AnimeInterp_LPIPS"] = calculate_lpips(gt_img, anime_img, lpips_vgg, device)

                if rife_path.exists():
                    rife_img = cv2.imread(str(rife_path))
                    row_data["RIFE_PSNR"] = calculate_psnr(gt_img, rife_img)
                    row_data["RIFE_SSIM"] = calculate_ssim(gt_img, rife_img)
                    row_data["RIFE_LPIPS"] = calculate_lpips(gt_img, rife_img, lpips_vgg, device)

                if film_path.exists():
                    film_img = cv2.imread(str(film_path))
                    row_data["FILM_PSNR"] = calculate_psnr(gt_img, film_img)
                    row_data["FILM_SSIM"] = calculate_ssim(gt_img, film_img)
                    row_data["FILM_LPIPS"] = calculate_lpips(gt_img, film_img, lpips_vgg, device)

                results.append(row_data)
                
                count += 1
                if count % 50 == 0:
                    print(f"Evaluated {count} triplets...")

    print(f"Evaluation complete. Converting {len(results)} rows to CSV...")
    df = pd.DataFrame(results)
    df.to_csv(output_csv_name, index=False)
    print(f"Successfully saved all metrics to {output_csv_name}!")

# ==========================================
# BLOCK 3: ARGUMENT PARSER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic VFI Evaluation Script")
    
    # Define the arguments. We set defaults so it won't crash if you forget to pass them.
    parser.add_argument('--dataset_dir', type=str, default="/content/atd_12k_trim/test_2k_540p", help="Path to the inferenced dataset folder")
    parser.add_argument('--annotations_dir', type=str, default="/content/atd_12k_trim/test_2k_annotations", help="Path to the JSON annotations folder")
    parser.add_argument('--output_csv', type=str, default="Master_VFI_Evaluation_Results.csv", help="Name of the output CSV file")
    
    args = parser.parse_args()
    
    # Pass the parsed arguments directly into the master pipeline
    run_evaluation(args.dataset_dir, args.annotations_dir, args.output_csv)
