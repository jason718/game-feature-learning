# Create a user_config.py file and set the variables there
# See user_config.example.py (do not commit this user_config.py to git)

def tryLoad(name, default=None):
    try:
        import user_config
    except:
        return None
    if hasattr(user_config, name):
        return getattr(user_config, name)
    return default

from os import path

THIS_DIR = path.dirname(path.realpath(__file__))
CAFFE_DIR = tryLoad('CAFFE_DIR', path.abspath(THIS_DIR+'../caffe/'))
VOC_DIR = tryLoad('VOC_DIR')
