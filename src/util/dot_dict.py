"""
Script of simple and recursive dotdict classes. 
- These are subclasses of dict to make a dictonary elements accessed as 
  attributes using dot. 
- Done by aliasing __getattr__ method as get method in dict class
"""

class dotdict(dict):
    """ To Convert normal dict to dot dict. Keys can then be accessed as 
    attributes using dot `.` notation 

    Example:
        >>> my_dict = {key1: value1, key2:value2}
        >>> my_dict = dotdict(my_dict)
        >>> print(my_dict.key1)
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class DotDict(dict):     
    """ DocDict is a recusrsive class that converts nested dict to nested dot dict. 
    Keys and sub dicts can then be accessed as attributes using dot `.` notation
    Example:
        >>> my_dict = {key1: {"name": {"first": "abc", "last": "xyz"}, age: 23},
        >>>            key2: value2}
        >>> my_dict = DotDict(my_dict)
        >>> print(my_dict.key1.name)
        >>> print(my_dict.key1.name.first) 
     """      
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val      
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__