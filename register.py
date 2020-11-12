import os
import random
import time
from sys import stdin
from Record import Record

path = "~/voxceleb_trainer"

dirName = 'registered'

if not os.path.exists(dirName):
    os.mkdir(dirName)

def process(path, dirName):
    # Use interactive user input
    # Input the user name in termial
    user_name = input("Enter your name: ")
    user_file_withpath = path + "/" + dirName + "/" + user_name.lower()    
    
    # Record an wav file to the registered dir
    if not os.path.exists(user_file_withpath):
        os.mkdir(user_file_withpath)   
 
    num_files = 0
    for path in os.listdir(user_file_withpath):
        if os.path.isfile(os.path.join(user_file_withpath, path)):
            num_files += 1

    newfile = num_files + 1 

    newfile_withpath = user_file_withpath + "/" + str(newfile) + ".wav"
    
    
    res = [random.randrange(0, 9, 1) for i in range(10)] 
    print("Please read the following numbers when you see" + ' "Recording..."' + ": ")
    print(str(res))
    
    Record.record(newfile_withpath)
    
while True:
    try:
        process(path, dirName) # perform some operation(s) on given string  
        time.sleep(1)
    except KeyboardInterrupt:
        print("\nRegister process is terminated!")
        break