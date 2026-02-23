import os
import json

API_DIR = "./APIs/services"
SRC_DIR = "./src"


def create_file_if_not_exists(filepath, content):
    if os.path.exists(filepath):
        print(f"Skipping existing file: {filepath}")
        return
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Created: {filepath}")


def generate_config(name):
    class_name = name.capitalize()
    return f"""from dataclasses import dataclass


@dataclass
class {class_name}Config:
    name: str = "{name}"
    version: str = "v1"
    batch_size: int = 32
    learning_rate: float = 0.001
"""


def generate_dataloader(name):
    class_name = name.capitalize()
    return f"""class {class_name}DataLoader:

    def __init__(self, config):
        self.config = config

    def fetch_data(self):
        \"\"\"Fetch raw data\"\"\"
        pass

    def fetch_training_dataloader(self):
        \"\"\"Return training dataloader\"\"\"
        pass

    def fetch_test_dataset(self):
        \"\"\"Return test dataset\"\"\"
        pass
"""


def generate_model(name):
    class_name = name.capitalize()
    return f"""class {class_name}ModelV1:

    def __init__(self, config):
        self.config = config

    def fit(self, data):
        \"\"\"Train the model\"\"\"
        pass

    def inference(self, input_data):
        \"\"\"Run inference\"\"\"
        pass


class {class_name}ModelV2:

    def __init__(self, config):
        self.config = config

    def fit(self, data):
        pass

    def inference(self, input_data):
        pass
"""


def generate_verification(name):
    class_name = name.capitalize()
    return f"""class {class_name}Verification:

    def __init__(self, model):
        self.model = model

    def verify_inference(self, input_data):
        \"\"\"Verify model inference\"\"\"
        pass

    def create_api_request(self, payload):
        \"\"\"Create API request payload\"\"\"
        pass
"""


def generate_worker(name):
    class_name = name.capitalize()
    return f"""from .config import {class_name}Config
from .dl import {class_name}DataLoader
from .model import {class_name}ModelV1
from .verification import {class_name}Verification


class {class_name}Worker:

    def __init__(self):
        self.config = {class_name}Config()
        self.dataloader = {class_name}DataLoader(self.config)
        self.model = {class_name}ModelV1(self.config)
        self.verification = {class_name}Verification(self.model)

    def run(self):
        data = self.dataloader.fetch_training_dataloader()
        self.model.fit(data)
"""


def main():
    if not os.path.exists(API_DIR):
        print("API directory not found.")
        return

    if not os.path.exists(SRC_DIR):
        os.makedirs(SRC_DIR)
    for filename in os.listdir(API_DIR):
        if not filename.endswith(".json"):
            continue

        name = filename[:-5]  # remove .json
        module_dir = os.path.join(SRC_DIR, name)

        if os.path.exists(module_dir):
            print(f"Skipping existing module folder: {module_dir}")
            continue

        os.makedirs(module_dir)

        create_file_if_not_exists(os.path.join(module_dir, "__init__.py"), "")
        create_file_if_not_exists(os.path.join(module_dir, "config.py"), generate_config(name))
        create_file_if_not_exists(os.path.join(module_dir, "dl.py"), generate_dataloader(name))
        create_file_if_not_exists(os.path.join(module_dir, "model.py"), generate_model(name))
        create_file_if_not_exists(os.path.join(module_dir, "verification.py"), generate_verification(name))
        create_file_if_not_exists(os.path.join(module_dir, "worker.py"), generate_worker(name))


if __name__ == "__main__":
    main()