import logging

#DEBUG
# 1. Base Configuration
# Here setup trace level (WARNING) and text message format
logging.basicConfig(
    level=logging.WARNING,
    # level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)

# Here Create Object logger with trace level INFO
LOGGER = logging.getLogger("AiAppLogger")
LOGGER.setLevel(logging.INFO)