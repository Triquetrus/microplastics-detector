import yaml
import os

print("Current directory:", os.getcwd())

yaml_path = "microplastic_dataset.yaml"
print("Exists?", os.path.exists(yaml_path))

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)
    print("YAML content:", data)
