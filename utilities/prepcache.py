import argparse
import logging
import sys

from torch.utils.data import DataLoader

from data.build_dataset import LunaDataset
from utilities.logconf import setup_logging
from utilities.utils import enumerate_with_estimate

logger = logging.getLogger(__name__)


class LunaPrepCacheApp:
    @classmethod
    def __init__(cls, sys_argv=None):
        cls.prep_dl = None
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=1024,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )

        cls.cli_args = parser.parse_args(sys_argv)

    def main(self):
        logger.info(f"Starting {type(self).__name__}, {self.cli_args}")

        self.prep_dl = DataLoader(
            LunaDataset(sort_by='series_uid'),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers
        )

        batch_iter = enumerate_with_estimate(self.prep_dl, "Stuffing cache", start_ndx=self.prep_dl.num_workers)
        for _ in batch_iter:
            pass


if __name__ == '__main__':
    setup_logging()
    logger.debug("\n\nLogging Started")
    LunaPrepCacheApp().main()
    logger.debug("Logging Ended\n\n")
