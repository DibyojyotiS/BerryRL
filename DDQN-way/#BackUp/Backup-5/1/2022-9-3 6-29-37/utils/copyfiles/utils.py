import shutil
import os
import re

# regerx pattern help--------------------------------------

# \   Used to drop the special meaning of character
#     following it (discussed below)
# []  Represent a character class
# ^   Matches the beginning
# $   Matches the end
# .   Matches any character except newline
# ?   Matches zero or one occurrence.
# |   Means OR (Matches with any of the characters
#     separated by it.
# *   Any number of occurrences (including 0 occurrences)
# +   One ore more occurrences
# {}  Indicate number of occurrences of a preceding RE 
#     to match.
# ()  Enclose a group of REs

# --------------------------------------------------------

def copy_files(frmDir,dest, ignore=['\.', '__pycache__','\#']):
    def should_ignore(str_):
        for k in ignore:
            if re.match(k,str_): return True
        return False
    for stuff in os.listdir(frmDir):
        if should_ignore(stuff): continue
        if os.path.isfile(f"{frmDir}/{stuff}"):
            if not os.path.exists(dest): os.makedirs(dest)
            shutil.copy2(f"{frmDir}/{stuff}", dest)
            print('copied', stuff, 'to', dest)
        if os.path.isdir(f"{frmDir}/{stuff}"):
            copy_files(f"{frmDir}/{stuff}", f"{dest}/{stuff}", ignore)
    print()
    # for file in [f for f in os.listdir(frmDirs) \
    #     if f.endswith('.py')]: shutil.copy2(file, dest)
    