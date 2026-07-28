#import pysftp
import sys, os
import time
import logging
import glob
import shutil
import pysftp

from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import multiprocessing

logging.raiseExceptions=False
def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]

def copy_func(payloads):
    # with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts) as sftp:
    #     print("Connection succesfully stablished ... ")

    cpu_amount = float(cpu_count())
    cpu_id = float(multiprocessing.current_process().name.split("-")[1])

    outputPath = payloads["output_path"]
    k=0
    time1 = time.time()
    for input_path in payloads["input_paths"]:
        with sftp.cd(outputPath+input_path.split("/")[-2]+"/"):
            sftp.put(input_path)
        # shutil.copy2(input_path,outputPath+input_path.split("/")[-2]+"/"+input_path.split("/")[-1])
        
        if k%1000==0:
            print("LEFT {}".format(len(payloads["input_paths"])-k))
            print(time.time()-time1)
        k+=1

#ip adress computera
myHostname = "172.30.10.117"
#login computer
myUsername = "svision"
#pswd computer
myPassword = "1q2w3e"

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

#path of 13mln photo
path = '/data/datasets/recognition/merged_ms1m_glint/'  # /mnt/ssd1/merged_ms1m_glint_copy/
#list_of_folders = next(os.walk(path))[1]

#pictures = []
#i=0
#for folder in list_of_folders:
#   pictures = pictures + glob.glob(path+folder+'/*')
#   print(len(pictures),pictures[-1])
    #i = i+1
    #if i==5:
    #   break
#type of photo
file_type = '*'
#end path of photo where it will be
remote_path = '/home/svision/datasets/merged_ms1m_glint_copy/'

pictures = sorted(glob.glob(path + file_type))
print(len(pictures))

procs = cpu_count()
procIDs = list(range(0, procs))

PicturesPerProc = len(pictures) / float(procs)
PicturesPerProc = int(np.ceil(PicturesPerProc))

chunkedPaths = list(chunk(pictures, PicturesPerProc))


payloads = []
for (i, imagePaths) in enumerate(chunkedPaths):
    data = {
        "input_paths": imagePaths,
        "output_path": remote_path
    }
    payloads.append(data)
#print(payloads)

start = time.time()

with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts) as sftp:
    print("Connection succesfully stablished ... ")

    pool = Pool(processes=procs)
    pool.map(copy_func, payloads)

    print("[INFO] waiting for processes to finish...")

    pool.close()
    pool.join()

    print(time.time()-start)
