import os

labels_folder = r"D:\microplastic_project\microplastic-detection\microplastic-detection\data\raw\microplastic_dataset\train\labels"

for file in os.listdir(labels_folder):
    if file.endswith(".txt"):
        path = os.path.join(labels_folder, file)
        with open(path, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = "0"  # set class to 0
            new_lines.append(" ".join(parts))
        with open(path, "w") as f:
            f.write("\n".join(new_lines))

print("✅ Macroplastic labels updated to class 0")
