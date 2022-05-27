import numpy as np
import numpy.linalg as linalg
import torch


def flatten_dict(dd, separator ='*', prefix =''): 
    """
    https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/
    """
    return { str(prefix) + separator + str(k) if prefix != '' else str(k) : v 
                for kk, vv in dd.items() 
                for k, v in flatten_dict(vv, separator, kk).items() 
                } if isinstance(dd, dict) else { prefix : dd } 

def set_in_nested_dict(nested_dict, keys, new_val):
    """
        Sets a value in a nested dictionary (or ml_collections config)
        e.g.
        nested_dict = \
        {
            'outer1': {
                'inner1': 4,
                'inner2': 5
            },
            'outer2': {
                'inner3': 314,
                'inner4': 654
            }
        } 
        keys = ['outer2', 'inner3']
        new_val = 315
    """
    if len(keys) == 1:
        nested_dict[keys[-1]] = new_val
        return
    return set_in_nested_dict(nested_dict[keys[0]], keys[1:], new_val)

def is_model_state_DDP(dict):
    for key in dict.keys():
        if '.module.' in key:
            return True
    return False

def remove_module_from_keys(dict):
    # dict has keys of the form a.b.module.c.d
    # changes to a.b.c.d
    new_dict = {}
    for key in dict.keys():
        if '.module.' in key:
            new_key = key.replace('.module.', '.')
            new_dict[new_key] = dict[key]
        else:
            new_dict[key] = dict[key]

    return new_dict