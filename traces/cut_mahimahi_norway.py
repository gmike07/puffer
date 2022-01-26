import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

DATA_PATH = './mahimahi/'
OUTPUT_PATH = './mahimahi_chunks/'
BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
BITS_IN_BYTE = 8.0

CHUNK_DURATION = 5 * 60.0  # duration in seconds
CHUNK_JUMP = 60.0  # shift in seconds


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx



def recreated_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)



def main():
	files = os.listdir(DATA_PATH)
	recreated_dir(OUTPUT_PATH)
	for file in files:
		file_path = DATA_PATH +  file
		output_path = OUTPUT_PATH + file

		print(file_path)

		mahimahi_win = []
		with open(file_path, 'r') as f:
			for line in f:
				mahimahi_win.append(float(line.split()[0]))

		mahimahi_win = np.array(mahimahi_win)
		chunk = 0
		start_time = 0
		while True:
			end_time = start_time + CHUNK_DURATION

			if len(mahimahi_win) == 0 or end_time * MILLISEC_IN_SEC > np.max(mahimahi_win): 
				break

			print("start time", start_time)

			start_ptr = find_nearest(mahimahi_win, start_time * MILLISEC_IN_SEC)
			end_ptr = find_nearest(mahimahi_win, end_time * MILLISEC_IN_SEC)

			with open(output_path + '_' + str(int(start_time)), 'w') as f:
				for i in range(start_ptr, end_ptr + 1):
					towrite = mahimahi_win[i] - mahimahi_win[start_ptr]
					f.write(str(int(towrite)) + '\n')

			start_time += CHUNK_JUMP
	

if __name__ == '__main__':
	main()