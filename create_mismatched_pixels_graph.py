import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = "output/image_comparison_results.csv"
comparison_df = pd.read_csv(
    file_path, header=None, names=["image_size", "sphere_count", "mismatched_pixels"]
)

# Display the content of the CSV to understand its structure
comparison_df.head(), comparison_df["image_size"].unique(), comparison_df[
    "sphere_count"
].unique()

# Mismatched Pixels vs Image Size
plt.figure(figsize=(12, 7))
image_size_groups = comparison_df.groupby("image_size", as_index=False)[
    "mismatched_pixels"
].mean()
plt.bar(
    image_size_groups["image_size"],
    image_size_groups["mismatched_pixels"],
    color="blue",
)
plt.xlabel("Output Image Size")
plt.ylabel("Average Mismatched Pixels")
plt.title("Mismatched Pixels vs. Image Size")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("mismatched_pixels_vs_image_size.png")
plt.show()

# Mismatched Pixels vs Sphere Count
plt.figure(figsize=(12, 7))
sphere_count_groups = comparison_df.groupby("sphere_count", as_index=False)[
    "mismatched_pixels"
].mean()
plt.bar(
    sphere_count_groups["sphere_count"].astype(str),
    sphere_count_groups["mismatched_pixels"],
    color="green",
)
plt.xlabel("Sphere Count")
plt.ylabel("Average Mismatched Pixels")
plt.title("Mismatched Pixels vs. Sphere Count")
plt.tight_layout()
plt.savefig("mismatched_pixels_vs_sphere_count.png")
plt.show()
