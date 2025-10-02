import os

train_path = "D:/microplastic-detection/data/raw/microplastic_dataset/train/images"
val_path = "D:/microplastic-detection/data/raw/microplastic_dataset/valid/images"

print("Train exists?", os.path.exists(train_path))
print("Val exists?", os.path.exists(val_path))
print("Train files:", os.listdir(train_path)[:5])
print("Val files:", os.listdir(val_path)[:5])
