import logging
import sys


def setup_loggers(logfile):

    logFormatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s]  %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')

    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(logfile, mode='w')
    fileHandler.setFormatter(logFormatter)

    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)

    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)
    return rootLogger


def close_loggers(rootLogger):

    x = list(rootLogger.handlers)
    for i in x:
        rootLogger.removeHandler(i)
        i.flush()
        i.close()


for ii in range(5):

    rootLogger = setup_loggers(f'log_test_{ii}.log')

    for jj in range(10):
        logging.info(jj)

    close_loggers(rootLogger)
