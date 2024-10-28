import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_file = "output/raytracer_results.csv"
column_names = ["trial_time", "image_size", "scene", "sphere_count", "type"]
df = pd.read_csv(csv_file, header=None, names=column_names)

df["sphere_count"] = df["sphere_count"].astype(int)

grouped_df = df.groupby(["image_size", "type", "sphere_count"], as_index=False).agg(
    {"trial_time": "mean"}
)

plt.figure(figsize=(12, 7))
types = grouped_df["type"].unique()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # colors for different types

image_sizes = sorted(grouped_df["image_size"].unique())
x = np.arange(len(image_sizes))  # the label locations
width = 0.25  # the width of the bars

for i, t in enumerate(types):
    subset = grouped_df[grouped_df["type"] == t].sort_values("image_size")
    plt.bar(
        x + i * width,
        subset["trial_time"].values[: len(x)],
        width,
        label=t,
        color=colors[i],
    )

plt.xlabel("Output Image Size")
plt.ylabel("Average Execution Time (s)")
plt.yscale("log")
plt.title("Raytracer Execution Time by Image Size and Type (Log Scale)")
plt.xticks(x + width / 2, image_sizes, rotation=45)
plt.legend(title="Executable Type")

plt.tight_layout()
plt.savefig("execution_timings_by_image_size.png")

plt.figure(figsize=(12, 7))

sphere_counts = sorted(grouped_df["sphere_count"].unique())
x = np.arange(len(sphere_counts))

for i, t in enumerate(types):
    subset = grouped_df[grouped_df["type"] == t].sort_values("sphere_count")
    plt.bar(
        x + i * width,
        subset["trial_time"].values[: len(x)],
        width,
        label=t,
        color=colors[i],
    )

plt.xlabel("Number of Spheres")
plt.ylabel("Average Execution Time (s)")
plt.yscale("log")
plt.title("Raytracer Execution Time by Sphere Count and Type (Log Scale)")
plt.xticks(x + width / 2, sphere_counts, rotation=45)
plt.legend(title="Executable Type")

plt.tight_layout()
plt.savefig("execution_timings_by_sphere_count.png")

print(
    "Graphs saved as execution_timings_by_image_size.png and execution_timings_by_sphere_count.png"
)
