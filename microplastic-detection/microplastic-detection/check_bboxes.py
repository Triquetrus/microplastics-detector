import cv2
import os

# Paths
images_folder = r"D:\microplastic_project\microplastic-detection\microplastic-detection\data\raw\microplastic_dataset\train\images"
labels_folder = r"D:\microplastic_project\microplastic-detection\microplastic-detection\data\raw\microplastic_dataset\train\labels"

# Check images img_0.jpg to img_10.jpg
for i in range(11):
    img_filename = f"img_{i}.jpg"
    label_filename = f"img_{i}.txt"

    img_path = os.path.join(images_folder, img_filename)
    label_path = os.path.join(labels_folder, label_filename)

    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print(f"Skipping {img_filename} — file missing")
        continue

    # Load image
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Draw bounding boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id, x_center, y_center, bw, bh = map(float, parts)

            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow(f"{img_filename}", img)
    print(f"Showing {img_filename}. Press any key to continue...")
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()
