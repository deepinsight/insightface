import pysftp
import sys, os
import time
import logging
import glob

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
	with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, cnopts=cnopts) as sftp:
		print("Connection succesfully stablished ... ")

		cpu_amount = float(cpu_count())
		cpu_id = float(multiprocessing.current_process().name.split("-")[1])

		outputPath = payloads["output_path"]
		k=0
		for input_path in payloads["input_paths"]:
			sftp.put(input_path,outputPath+input_path.split("/")[-1])
			
			if k%1000==0:
				print("LEFT {}".format(len(payloads["input_paths"])-k))
			k+=1

#ip adress computera
myHostname = "10.16.107.15"
#login computer
myUsername = "umai"
#pswd computer
myPassword = "passw0rd13!"

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None

folders = '/home/umai/'
remote_folder = '/photo/'

folders_lst = ['ud_gr_photos']

print('Amount of folders:', len(folders_lst))

for folder_name in folders_lst:
	#path of 13mln photo
	path = folders + folder_name + '/'
	print('Source:', path)
	#type of photo
	file_type = '*.ldr'
	#end path of photo where it will be
	remote_path = remote_folder + folder_name + '/'
	if not os.path.exists(remote_path):
        	os.makedirs(remote_path)
	print('Destination:', remote_path)

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

	pool = Pool(processes=procs)
	pool.map(copy_func,	payloads)

	print("[INFO] waiting for processes to finish...")

	pool.close()
	pool.join()

	print(time.time()-start)
