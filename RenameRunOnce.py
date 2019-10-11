#Warning : run once only

import numpy as np
import os.path

import os

# Function to rename multiple files
def main():

    for f in range(1,6):
        for g in range(2):
            types=["ham","spam"]
            x = types[g]
            y = str(f)
            data_dir = r"/home/ahmed/Desktop/python/Assignment2/Data/enron"+ y+ "/"+x+"/"

            i = 1
            for filename in os.listdir(data_dir):

                dst = str(i) + ".txt"

                src = os.path.join(data_dir + filename)
                dst = os.path.join(data_dir + dst)

                # rename() function will
                # rename all the files
                os.rename(src, dst)
                i += 1

main()
