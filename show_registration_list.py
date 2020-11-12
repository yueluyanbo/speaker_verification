import os

path = "~/speaker_verification"
dirName = "registered"

dir_withpath = path + "/" + dirName

if not os.path.exists(dir_withpath):
    print("No registered user!")
else:
    #show a list of all subdirectories in the current directory
    names = next(os.walk(dir_withpath))[1]
    thing = "test"
    if thing in names: 
        names.remove(thing)
    print(*names, sep = "\n")  