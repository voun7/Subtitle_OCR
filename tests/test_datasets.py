from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from os import cpu_count
from pathlib import Path
from time import perf_counter

import yaml

from train import build_datasets


def dur_calc(start_time: float) -> timedelta:
    return timedelta(seconds=round(perf_counter() - start_time))


class TestDataset:

    def __init__(self, config_file: Path) -> None:
        """
        This class uses a config file to test all data in the dataset for errors that may occur during postprocessing
        or augmentation. This test will take long on large datasets.
        """
        self.config = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    @staticmethod
    def _mini_batch(dataset: list, start_idx: int, stop_idx: int) -> None:
        for idx in range(start_idx, stop_idx):
            try:
                _ = dataset[idx]
                print(f"\r{idx=} passed test", end="", flush=True)
            except Exception as error:
                print(end="\r", flush=True)
                print(f"{idx=} failed test. {error=}")

    def batch_tester(self, ds_name: str, dataset: list, ds_len: int) -> None:
        start_time, no_processes = perf_counter(), cpu_count() // 4
        batch_size = int(ds_len / (no_processes * no_processes))
        print(f"Running Tests for {ds_name} dataset... {no_processes=}, {batch_size=:,}")
        ds_chunks = [[i, i + batch_size] for i in range(0, ds_len, batch_size)]
        ds_chunks[-1][-1] = ds_len  # make sure last batch has correct end
        with ProcessPoolExecutor(no_processes) as executor:
            futures = [executor.submit(self._mini_batch, dataset, idx[0], idx[1]) for idx in ds_chunks]
            for f in futures:
                f.result()  # Prevents silent bugs. Exceptions raised will be displayed.
        print(f"\n{ds_name} dataset testing completed... Duration: {dur_calc(start_time)}\n")

    def run(self, lang: str) -> None:
        start_time = perf_counter()
        print(f"Loading Dataset lang: {lang}...")
        train_ds, val_ds = build_datasets(lang, self.config["Dataset"])
        train_ds_len, val_ds_len = len(train_ds), len(val_ds)
        print(f"Loading Completed... Dataset Size Train: {train_ds_len:,}, Val: {val_ds_len:,}, "
              f"Duration: {dur_calc(start_time)}\n")

        self.batch_tester("Training", train_ds, train_ds_len)
        self.batch_tester("Validation", val_ds, val_ds_len)


def main():
    config_file = Path("../configs/det/det_ppocr_v3.yml")
    tester = TestDataset(config_file)
    tester.run("ch")
    tester.run("en")


if __name__ == '__main__':
    main()
