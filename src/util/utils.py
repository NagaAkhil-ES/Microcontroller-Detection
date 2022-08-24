import os
import re

# get in-digits in input string(ips)
get_digits = lambda ips: int(re.sub('\D', '', ips))
has_number = lambda ips: bool(re.search(r'\d', ips))
all_has_number = lambda l_strings: all(map(has_number, l_strings))

def listdir(directory, file_ext):
    # list files ending with {file_ext}
    files = [f for f in os.listdir(directory) if f.endswith(file_ext)]
    # sort files as per in-digits
    if all_has_number(files): # if all file names have numbers only then sort them
        files.sort(key=get_digits)
    return files

def setup_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# unti testing
if __name__ == "__main__":
    # import pdb;pdb.set_trace()
    # print(listdir("models/exp3_3", ".pth"))
    t1 = ["hello1", "hello"];
    print(t1, all_has_number(t1))
    t1 = ["hello1", "hello2"];
    print(t1, all_has_number(t1))