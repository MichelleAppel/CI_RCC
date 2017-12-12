#! /usr/bin/env python3
from network import Network
from pytocl.main import main
from my_driver import MyDriver
import multiprocessing
import psutil
import subprocess
import pickle


path = "ea_output/output_2017-12-12 16:42:55.407269/outfile_gen10"


# Initialize socket process
def init_process():
	bashCommand = "torcs" # The command to be executed to start a quickrace without visualisation
	return subprocess.Popen(bashCommand.split())								# Return subprocess so that it can be killed

# Kill socket process
def kill_process(server_proc):
	process = psutil.Process(server_proc.pid)						# Get process
	for proc in process.children(recursive=True):					# For all processes
		proc.kill()													# Kill them all
	process.kill()													# Kill it

def select_best(population):
	pop_list = [(member.fitness, member) for member in population]	# Make 2 dimensional list with the fitness of a individual as first argument
																	# and the individual itself as second argument
	pop_list = sorted(pop_list, key=lambda t: t[0], reverse=True)	# Sort according to fitness and reverse so that highest fitness is first in the list
	return pop_list[0][1]

def start_weights_from_file(two_cars=False):
	networks = pickle.load( open(path, "rb" ))
	network = select_best(networks)

	server_proc = init_process()

	que = multiprocessing.Queue()

	p1 = multiprocessing.Process(
		target=main, 
		args = (MyDriver(network=network), 
		999999999999, 
		3001, 
		que))

	p1.start()

	if two_cars == True:
		p2 = multiprocessing.Process(target=main, args = (MyDriver(network=network), 999999999999, 3002, que))
		p2.start()

	p1.join()
	if two_cars == True:
		p2.join()

	kill_process(server_proc)


start_weights_from_file(two_cars=False)