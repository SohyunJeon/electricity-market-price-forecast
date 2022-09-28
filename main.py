import importlib
import logging
import sys
from concurrent import futures
from optparse import OptionParser
import pandas as pd

import grpc
from plus_ba import plus_ba_pb2_grpc as pb2_grpc
from fps.server import PlusBAServer
import config
from config import Config


def serve():
    # if module is None:
    #     logging.info("module name cannot be empty")
    #     parser.print_help()
    #     sys.exit(1)

    cfg = Config()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg.get_max_workers()))

    # mod = importlib.import_module(module)
    # plus_ba_server = getattr(mod, 'PlusBAServer')

    pb2_grpc.add_PlusBAServicer_to_server(PlusBAServer(), server)
    server.add_insecure_port(f'[::]:{cfg.get_port()}')
    server.start()

    logging.info(f'server started, port: {cfg.get_port()}')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("server will be shutdown")
        server.stop(10).wait()


if __name__ == '__main__':
    log_format = '%(asctime)s %(filename)s:%(lineno)d %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=log_format, datefmt='%Y/%m/%d %H:%M:%S')

    parser = OptionParser()
    parser.add_option("--config", dest="config_file", default="config.yml", help="config file path",
                      metavar="PLUS_BA_CONFIG_PATH")
    parser.add_option("--module", dest="module", help="module for loading", metavar="PLUS_BA_MODULE")
    (options, args) = parser.parse_args()

    config.load_config(options.config_file)
    # serve(options.module)
    serve()
