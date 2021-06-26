import numpy as np
import datetime
import os

FILE_PATH = '/home/ubuntu/Downloads/data-raw-2016-jan/201601/curr_webget_2016_01.csv'
OUTPUT_PATH = './cooked/'
NUM_LINES = 20000
TIME_ORIGIN = datetime.datetime.utcfromtimestamp(0)


bw_measurements = {}
def main():
	line_counter = 0
	with open(FILE_PATH, 'rb') as f:
		for line in f:
			parse = line.split(',')
			uid = parse[0]
			dtime = (datetime.datetime.strptime(parse[1],'%Y-%m-%d %H:%M:%S') 
				- TIME_ORIGIN).total_seconds()
			target = parse[2]
			address = parse[3]
			throughput = parse[6]  # bytes per second

			k = (uid, target)

			if k in bw_measurements:
				bw_measurements[k].append(throughput)
			else:
				bw_measurements[k] = [throughput]

			line_counter += 1
			if line_counter >= NUM_LINES:
				break

	for k in bw_measurements:
		out_file = 'trace_' + '_'.join(k)
		out_file = out_file.replace(':', '-')
		out_file = out_file.replace('/', '-')

		if not os.path.exists(OUTPUT_PATH):
			os.mkdir(OUTPUT_PATH)

		out_file = OUTPUT_PATH + out_file
		with open(out_file, 'wb') as f:
			for i in bw_measurements[k]:
				f.write(str(i) + '\n')

if __name__ == '__main__':
	main()

