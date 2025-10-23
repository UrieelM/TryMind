
# region Imports and Setup
import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
# endregion

# region Image Loading and Preprocessing
# Image path to analyze
img_path = "dataset/000000007459.jpg"

# Read image and convert to grayscale
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Could not load the image.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Normalize for entropy calculation
gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# endregion

# region Metric Calculation Functions
# Function to calculate entropy


def calculate_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    return entropy(hist + 1e-7)

# Function to calculate Laplacian of Gaussian variance


def calculate_log_variance(img):
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)
    return laplacian.var()
# endregion


# region Original Image Analysis
# Calculate original entropy
orig_entropy = calculate_entropy(gray_norm)
orig_log_var = calculate_log_variance(gray)

print(f"\nOriginal image values:")
print(f"Original entropy: {orig_entropy:.4f}")
print(f"Original LoG variance: {orig_log_var:.4f}")
# endregion

# region STAGE 1: COARSE OPTIMIZATION
print("\n" + "="*50)
print("STAGE 1: COARSE PARAMETER SEARCH")
print("="*50)

# Define initial coarse arrays
window_sizes_coarse = [1, 2, 4, 8, 16, 32, 64]
clip_limits_coarse = [1, 5, 10, 15]

# Store results for stage 1
results_stage1 = []

print("\nStage 1: Testing coarse parameter combinations...")
for ws in window_sizes_coarse:
    for cl in clip_limits_coarse:
        # Calculate the formula first: (clip_limit * (window_size^2)) / 256
        formula_result = int((cl * (ws * ws)) / 256)

        # Skip if formula result is below 1 (ineffective clipping threshold)
        if formula_result < 1:
            print(
                f"Skipping ws={ws}, cl={cl} - Formula result ({formula_result}) below threshold")
            continue

        # Skip if formula result is higher than clip limit (no actual clipping occurs)
        if formula_result > cl:
            print(
                f"Skipping ws={ws}, cl={cl} - Formula result ({formula_result}) exceeds clip limit ({cl})")
            continue

        clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(ws, ws))
        img_clahe = clahe.apply(gray)
        ent = calculate_entropy(cv2.normalize(
            img_clahe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        log_var = calculate_log_variance(img_clahe)

        results_stage1.append({
            "Window Size": ws,
            "Clip Limit": cl,
            "Entropy": ent,
            "LoG Variance": log_var,
            "Formula Result": formula_result,
            "Image CLAHE": img_clahe
        })

# Create DataFrame for stage 1
df_stage1 = pd.DataFrame(results_stage1)
print("\nStage 1 Results:")
print(df_stage1[["Window Size", "Clip Limit", "Entropy",
      "LoG Variance", "Formula Result"]].to_string(index=False))

# Stage 1: Find optimal parameters using Euclidean distance
Me_stage1 = df_stage1["Entropy"].mean()
Msigma_stage1 = df_stage1["LoG Variance"].mean()
df_stage1["Distance_to_Avg"] = np.sqrt(
    (df_stage1["Entropy"] - Me_stage1)**2 + (df_stage1["LoG Variance"] - Msigma_stage1)**2)
opt_stage1 = df_stage1.loc[df_stage1["Distance_to_Avg"].idxmin()]

print(f"\nStage 1 Optimal Parameters:")
print(
    f"Window Size: {opt_stage1['Window Size']}, Clip Limit: {opt_stage1['Clip Limit']}")
print(
    f"Entropy: {opt_stage1['Entropy']:.4f}, LoG Variance: {opt_stage1['LoG Variance']:.2f}")

# Determine optimal ranges based on stage 1 results
# Use a more flexible approach: select top 50% of results or expand around the optimal
# More flexible threshold
threshold_distance = opt_stage1['Distance_to_Avg'] * 2.0
good_params = df_stage1[df_stage1['Distance_to_Avg'] <= threshold_distance]

# If still too restrictive, take top 50% of all results
if len(good_params) < 3:
    df_sorted = df_stage1.sort_values('Distance_to_Avg')
    top_half = max(3, len(df_stage1) // 2)  # At least 3 or half of results
    good_params = df_sorted.head(top_half)

cl_min = good_params['Clip Limit'].min()
cl_max = good_params['Clip Limit'].max()
ws_min = good_params['Window Size'].min()
ws_max = good_params['Window Size'].max()

# Ensure minimum range width for meaningful Stage 2 exploration
if cl_max == cl_min:
    # Expand clip limit range by at least ±2 around the optimal
    opt_cl = opt_stage1['Clip Limit']
    cl_min = max(1, opt_cl - 2)
    cl_max = min(15, opt_cl + 2)

if ws_max == ws_min:
    # Add neighboring window sizes from the original array
    opt_ws = opt_stage1['Window Size']
    ws_candidates = [ws for ws in window_sizes_coarse]
    opt_idx = ws_candidates.index(opt_ws) if opt_ws in ws_candidates else 0

    # Include current and neighboring window sizes
    ws_indices = []
    for i in range(max(0, opt_idx-1), min(len(ws_candidates), opt_idx+2)):
        ws_indices.append(i)

    expanded_ws = [ws_candidates[i] for i in ws_indices]
    ws_min = min(expanded_ws)
    ws_max = max(expanded_ws)

print(f"\nStage 1 Optimal Ranges:")
print(f"Clip Limit Range: {cl_min} - {cl_max}")
print(f"Window Size Range: {ws_min} - {ws_max}")
# endregion

# region STAGE 2: FINE-TUNED OPTIMIZATION
print("\n" + "="*50)
print("STAGE 2: REFINED PARAMETER SEARCH")
print("="*50)

# Create refined arrays within the optimal ranges
# For clip limits: create integer array with step of 1 within the range
clip_limits_fine = []
for cl in range(int(cl_min), int(cl_max) + 1):
    clip_limits_fine.append(cl)

# For window sizes: stick to the original array, filter by the optimal range
window_sizes_fine = [
    ws for ws in window_sizes_coarse if ws_min <= ws <= ws_max]

# If the range is too narrow, add neighboring values from original array
if len(window_sizes_fine) < 3:
    # Add some values around the range from the original array
    extended_window_sizes = []
    for ws in window_sizes_coarse:
        if ws_min//2 <= ws <= ws_max*2:
            extended_window_sizes.append(ws)
    window_sizes_fine = extended_window_sizes[:8]  # Limit to reasonable number

print(f"Refined Clip Limits: {clip_limits_fine}")
print(f"Refined Window Sizes: {window_sizes_fine}")

# Store results for stage 2
results_stage2 = []

print(f"\nStage 2: Testing refined parameter combinations...")
for ws in window_sizes_fine:
    for cl in clip_limits_fine:
        # Calculate the formula first: (clip_limit * (window_size^2)) / 256
        formula_result = int((cl * (ws * ws)) / 256)

        # Skip if formula result is below 1 (ineffective clipping threshold)
        if formula_result < 1:
            print(
                f"Skipping ws={ws}, cl={cl} - Formula result ({formula_result}) below threshold")
            continue

        # Skip if formula result is higher than clip limit (no actual clipping occurs)
        if formula_result > cl:
            print(
                f"Skipping ws={ws}, cl={cl} - Formula result ({formula_result}) exceeds clip limit ({cl})")
            continue

        clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(ws, ws))
        img_clahe = clahe.apply(gray)
        ent = calculate_entropy(cv2.normalize(
            img_clahe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        log_var = calculate_log_variance(img_clahe)

        results_stage2.append({
            "Window Size": ws,
            "Clip Limit": cl,
            "Entropy": ent,
            "LoG Variance": log_var,
            "Formula Result": formula_result,
            "Image CLAHE": img_clahe
        })

# Create DataFrame for stage 2
df = pd.DataFrame(results_stage2)
print(f"\nStage 2 Results ({len(results_stage2)} combinations):")
print(df[["Window Size", "Clip Limit", "Entropy",
      "LoG Variance", "Formula Result"]].to_string(index=False))
# endregion

# region FINAL OPTIMIZATION (Euclidean Distance)
print(f"\n" + "="*50)
print("FINAL OPTIMAL PARAMETERS")
print("="*50)

# Combined Metric Optimization (Euclidean Distance) - Same logic as testit.py
# Calculate average entropy and Laplacian variance across all combinations from both stages
all_results = results_stage1 + results_stage2
df_all = pd.DataFrame(all_results)
Me = df_all["Entropy"].mean()
Msigma = df_all["LoG Variance"].mean()

print(f"Average vector of all combinations:")
print(f"├─ Average Entropy (Me): {Me:.4f}")
print(f"└─ Average LoG Variance (Msigma): {Msigma:.2f}")

print(f"\nCalculating Euclidean distances to average vector:")

# Calculate distances for Stage 1: distance to average vector
df_stage1["Distance_to_Avg"] = np.sqrt(
    (df_stage1["Entropy"] - Me)**2 + (df_stage1["LoG Variance"] - Msigma)**2)

# Calculate distances for Stage 2: distance to average vector
df["Distance_to_Avg"] = np.sqrt(
    (df["Entropy"] - Me)**2 + (df["LoG Variance"] - Msigma)**2)

# Find optimal parameters from BOTH stages using distance to average vector
opt_stage1_combined = df_stage1.loc[df_stage1["Distance_to_Avg"].idxmin()]
opt_stage2_combined = df.loc[df["Distance_to_Avg"].idxmin()]

# Compare both stages to find the TRUE optimal (closest to average vector)
if opt_stage1_combined['Distance_to_Avg'] <= opt_stage2_combined['Distance_to_Avg']:
    opt_combined = opt_stage1_combined
    opt_combined_source = "Stage 1"
    opt_combined_distance = opt_stage1_combined['Distance_to_Avg']
else:
    opt_combined = opt_stage2_combined
    opt_combined_source = "Stage 2"
    opt_combined_distance = opt_stage2_combined['Distance_to_Avg']

print(f"\nFinal Optimal Parameters (minimum distance to average vector):")
print(f"Source: {opt_combined_source}")
print(f"Window Size: {opt_combined['Window Size']}")
print(f"Clip Limit: {opt_combined['Clip Limit']}")
print(f"Entropy: {opt_combined['Entropy']:.4f}")
print(f"LoG Variance: {opt_combined['LoG Variance']:.2f}")
print(f"Distance to Average: {opt_combined_distance:.4f}")
print(f"Formula Result: {opt_combined['Formula Result']}")

print(f"\nStage Results Comparison:")
print(
    f"Stage 1 Best: WS={opt_stage1_combined['Window Size']}, CL={opt_stage1_combined['Clip Limit']}, Distance={opt_stage1_combined['Distance_to_Avg']:.4f}")
print(
    f"Stage 2 Best: WS={opt_stage2_combined['Window Size']}, CL={opt_stage2_combined['Clip Limit']}, Distance={opt_stage2_combined['Distance_to_Avg']:.4f}")

if opt_stage1_combined['Distance_to_Avg'] <= opt_stage2_combined['Distance_to_Avg']:
    print(f"✓ Stage 1 provides better optimization (closer to average vector)")
    improvement_pct = 0  # No improvement, Stage 1 is better
else:
    improvement_pct = ((opt_stage1_combined['Distance_to_Avg'] -
                       opt_stage2_combined['Distance_to_Avg']) / opt_stage1_combined['Distance_to_Avg']) * 100
    print(
        f"✓ Stage 2 provides better optimization: {improvement_pct:.2f}% closer to average vector")
# endregion

# region Image Comparison Visualizations
# Display comparison: Original, Stage 1 optimal, and Stage 2 optimal
img_opt_stage1 = opt_stage1["Image CLAHE"]
img_opt_combined = opt_combined["Image CLAHE"]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(gray, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(img_opt_stage1, cmap='gray')
axs[1].set_title(
    f"Stage 1 Optimal CLAHE\nWS={opt_stage1['Window Size']} CL={opt_stage1['Clip Limit']}\nDistance={opt_stage1['Distance_to_Avg']:.4f}")
axs[1].axis('off')

axs[2].imshow(img_opt_combined, cmap='gray')
axs[2].set_title(
    f"Final Optimal CLAHE Stage 2\nWS={opt_combined['Window Size']} CL={opt_combined['Clip Limit']}\nDistance={opt_combined['Distance_to_Avg']:.4f}")
axs[2].axis('off')

plt.tight_layout()
plt.show()
