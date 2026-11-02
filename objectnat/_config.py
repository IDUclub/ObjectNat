import sys
from typing import Literal

from loguru import logger


class Config:
    """
    A configuration class to manage global settings for the application, such as Overpass API URL,
    timeouts, and logging options.

    Attributes
    ----------
    enable_tqdm_bar : bool
        Enables or disables progress bars (via tqdm). Defaults to True.
    logger : Logger
        Logging instance to handle application logging.

    Methods
    -------
    change_logger_lvl(lvl: Literal["TRACE", "DEBUG", "INFO", "WARN", "ERROR"])
        Changes the logging level to the specified value.
    set_enable_tqdm(enable: bool)
        Enables or disables progress bars in the application.
    """

    def __init__(
        self,
        enable_tqdm_bar=True,
    ):
        self.enable_tqdm_bar = enable_tqdm_bar
        self.logger = logger

    def change_logger_lvl(self, lvl: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]):
        self.logger.remove()
        self.logger.add(sys.stderr, level=lvl)

    def set_enable_tqdm(self, enable: bool):
        self.enable_tqdm_bar = enable


config = Config()
config.change_logger_lvl("WARNING")
