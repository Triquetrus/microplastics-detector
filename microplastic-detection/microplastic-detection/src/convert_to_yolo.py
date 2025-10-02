import pandas as pd
import os

# Paths to CSV files
train_csv = "D:/microplastic-detection/data/raw/microplastic_dataset/train/train_annotations.csv"
valid_csv = "D:/microplastic-detection/data/raw/microplastic_dataset/valid/valid_annotations.csv"


# Paths to save YOLO labels
train_label_folder = "D:/microplastic-detection/data/raw/microplastic_dataset/train/labels"
valid_label_folder = "D:/microplastic-detection/data/raw/microplastic_dataset/valid/labels"

# Make sure labels folders exist
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(valid_label_folder, exist_ok=True)

# Define class mapping
classes = ["Microplastic"]  # Add more if needed

def convert(df, label_folder, image_folder):
    for filename in df['filename'].unique():
        records = df[df['filename'] == filename]
        txt_file = os.path.join(label_folder, os.path.splitext(filename)[0] + ".txt")

        with open(txt_file, 'w') as f:
            for _, row in records.iterrows():
                class_id = classes.index(row['class'])
                
                # Get image size
                img_path = os.path.join(image_folder, row['filename'])
                from PIL import Image
                im = Image.open(img_path)
                w, h = im.size
                
                # YOLO normalized coordinates
                x_center = ((row['xmin'] + row['xmax']) / 2) / w
                y_center = ((row['ymin'] + row['ymax']) / 2) / h
                width = (row['xmax'] - row['xmin']) / w
                height = (row['ymax'] - row['ymin']) / h
                
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Convert train labels
train_images_folder = "D:/microplastic-detection/data/raw/microplastic_dataset/train/images"
train_df = pd.read_csv(train_csv)
convert(train_df, train_label_folder, train_images_folder)

# Convert validation labels
valid_images_folder = "D:/microplastic-detection/data/raw/microplastic_dataset/valid/images"
valid_df = pd.read_csv(valid_csv)
convert(valid_df, valid_label_folder, valid_images_folder)

print("YOLO labels created successfully!")
