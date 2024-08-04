import random
from pathlib import Path
from unittest import TestCase

import torch
import yaml

from sub_ocr.losses import build_loss
from sub_ocr.metrics import build_metric
from sub_ocr.modeling.architectures import build_model
from train import build_optimizer, build_datasets

all_configs = list(Path("../configs").glob("**/*.yml"))
det_configs_dir = list(Path("../configs/det").glob("**/*.yml"))
rec_configs_dir = list(Path("../configs/rec").glob("**/*.yml"))


class TestBuildModel(TestCase):

    def test_det_build_model(self) -> None:
        test_name = "det build model"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(det_configs_dir):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                _ = build_model(config | {"lang": "en"})
                print(f"Config: {config_file}, passed {test_name} test.")

    def test_det_model_train_forward_pass(self) -> None:
        test_name = "det model train forward pass"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(det_configs_dir):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                model = build_model(config | {"lang": "en"}).train()
                _ = model(torch.rand([2, 3, config["Dataset"]["image_height"], config["Dataset"]["image_width"]]))
                print(f"Config: {config_file}, passed {test_name} test.")

    def test_det_model_eval_forward_pass(self) -> None:
        test_name = "det model eval forward pass"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(det_configs_dir):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                model = build_model(config | {"lang": "en"}).eval()
                _ = model(torch.rand([2, 3, config["Dataset"]["image_height"], config["Dataset"]["image_width"]]))
                print(f"Config: {config_file}, passed {test_name} test.")

    def test_rec_build_model(self) -> None:
        test_name = "rec build model"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(rec_configs_dir):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                _ = build_model(config | {"lang": "en"})
                print(f"Config: {config_file}, passed {test_name} test.")

    def test_rec_model_train_forward_pass(self) -> None:
        test_name = "rec model train forward pass"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(rec_configs_dir):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                model = build_model(config | {"lang": "en"}).train()
                _ = model(torch.rand([4, 3, config["Dataset"]["image_height"], config["Dataset"]["image_width"]]))
                print(f"Config: {config_file}, passed {test_name} test.")

    def test_rec_model_eval_forward_pass(self) -> None:
        test_name = "rec model eval forward pass"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(rec_configs_dir):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                model = build_model(config | {"lang": "en"}).eval()
                _ = model(torch.rand([4, 3, config["Dataset"]["image_height"], config["Dataset"]["image_width"]]))
                print(f"Config: {config_file}, passed {test_name} test.")


class TestBuildLoss(TestCase):

    def test_build_loss(self) -> None:
        test_name = "build loss"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(all_configs):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                _ = build_loss(config["Loss"])
                print(f"Config: {config_file}, passed {test_name} test.")


class TestBuildMetric(TestCase):
    def test_build_metric(self) -> None:
        test_name = "build metric"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(all_configs):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                _ = build_metric(config | {"lang": "en"})
                print(f"Config: {config_file}, passed {test_name} test.")


class TestOptimizer(TestCase):
    def test_build_optimizer(self) -> None:
        test_name = "build optimizer"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(all_configs):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                model = build_model(config | {"lang": "en"})
                _ = build_optimizer(model, config["Optimizer"])
                print(f"Config: {config_file}, passed {test_name} test.")


class TestDetectionDataset(TestCase):
    def test_dataset(self) -> None:
        test_name = "build dataset"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(det_configs_dir):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                train_ds, val_ds = build_datasets("test", config["Dataset"])
                _ = train_ds[random.randint(0, len(train_ds))], val_ds[random.randint(0, len(val_ds))]
                print(f"Config: {config_file}, passed {test_name} test.")


class TestRecognitionDataset(TestCase):
    def test_dataset(self) -> None:
        test_name = "build dataset"
        print(f"\nTesting {test_name} with configs...")
        for i, config_file in enumerate(rec_configs_dir):
            with self.subTest(f"File: {config_file}", i=i):
                config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
                train_ds, val_ds = build_datasets("test", config["Dataset"])
                _ = train_ds[random.randint(0, len(train_ds))], val_ds[random.randint(0, len(val_ds))]
                print(f"Config: {config_file}, passed {test_name} test.")
