import logging

logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
# TODO: Add


def create_logger(
    name: str,
    level: str,
    format: str = "%(asctime)s | %(levelname)s | %(filename)s:%(funcName)s | %(message)s",
):
    assert level in ["INFO", "DEBUG", "WARNING", "ERROR"], "Illegal log level"
    root_logger = logging.getLogger(name)
    if level == "DEBUG":
        root_logger.setLevel(logging.DEBUG)
    elif level == "INFO":
        root_logger.setLevel(logging.INFO)
    elif level == "WARNING":
        root_logger.setLevel(logging.WARNING)
    elif level == "ERROR":
        root_logger.setLevel(logging.ERROR)

    ch = logging.StreamHandler()
    root_logger.addHandler(ch)
    formatter = logging.Formatter(format)
    ch.setFormatter(formatter)

    return logging.getLogger(name)


logger = create_logger(__name__, "INFO")
