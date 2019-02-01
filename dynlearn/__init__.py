import os

__version__ = '0.1.0'


def get_file_name(name):
    """If not running in a script join the current file with the path, otherwise
    join 'dynlearn' with name."""
    if '__file__' in globals():
        dir_name = os.path.dirname(__file__)
    else:
        dir_name = 'dynlearn'
    file_name = os.path.join(dir_name, name)
    return file_name
