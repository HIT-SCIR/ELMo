#!/usr/bin/env python
from .elmo import Embedder


import logging
logger = logging.getLogger('elmoformanylangs')

# if the client application hasn't set the log level, we set it
# ourselves to INFO
if logger.level == 0:
    logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="%(asctime)-15s %(levelname)s: %(message)s")
log_handler.setFormatter(log_formatter)

# also, if the client hasn't added any handlers for this logger
# (or a default handler), we add a handler of our own
#
# client can later do
#   logger.removeHandler(stanza.log_handler)
if not logger.hasHandlers():
    logger.addHandler(log_handler)
