import os

def get_file_name(name):
    if '__file__' in globals():
        dir_name = os.path.dirname(__file__)
    else:
        dir_name = 'dynlearn'
    file_name = os.path.join(dir_name,name)
    return file_name
