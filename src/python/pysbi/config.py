from ConfigParser import ConfigParser
import os

def get_conf_dir():
    fpath = os.path.abspath(__file__)
    (cwdir, fname) = os.path.split(fpath)
    cdir = os.path.abspath(os.path.join(cwdir, '..', '..', '..', 'conf'))
    return cdir


CONF_DIR = get_conf_dir()
#DATA_DIR = '../../data'

config = ConfigParser()
config.read(os.path.join(CONF_DIR, 'config.properties'))
SRC_DIR=config.get('dir','src_dir')
DATA_DIR=config.get('dir','data_dir')
LOG_DIR=config.get('dir','log_dir')
TEMPLATE_DIR=config.get('dir','template_dir')