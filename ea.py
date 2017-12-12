from network import Network
from pytocl.main import main
from my_driver import MyDriver

import xml.etree.ElementTree as ET
import numpy as np

import os
import multiprocessing
import subprocess
import psutil
import pickle

import random
import datetime


class EA(): # Evolutionary Algorithm

	# Initialize EA object
	def __init__(self, NO_GENERATIONS, POPULATION_SIZE, INIT_NO_FRAMES, FINAL_NO_FRAMES, FRACTION_MUTATE, NUMBER_OF_PARTNERS):
		self.NO_GENERATIONS = NO_GENERATIONS		# The number of generations to be evolved
		self.POPULATION_SIZE = POPULATION_SIZE		# The population size per generation
		self.INIT_NO_FRAMES = INIT_NO_FRAMES		# The initial number of frames until the process is killed
		self.FINAL_NO_FRAMES = FINAL_NO_FRAMES		# The number of frames for the last generation until the process is killed
		self.FRACTION_BEST = 1./(2*NUMBER_OF_PARTNERS)# The fraction of best individuals of a population that is going to produce offspring
		self.FRACTION_MUTATE = FRACTION_MUTATE		# The fraction of the generation that is mutated
		self.NUMBER_OF_PARTNERS = NUMBER_OF_PARTNERS# The number of partners to make offspring with

		self.no_frames_list = np.linspace(INIT_NO_FRAMES, FINAL_NO_FRAMES, num=NO_GENERATIONS)

		self.fitness1 = 0
		self.fitness2 = 0

		if float(self.POPULATION_SIZE)*self.FRACTION_BEST != int(float(self.POPULATION_SIZE)*self.FRACTION_BEST):
			print("Warning! Population size will change.")	# Print warning when population size will not be the same as initial

	# --------------------------------------------------------------------- Socket process --------------------------------------------------------------------- # 

	# Initialize socket process
	def init_process(self):
		bashCommand = "torcs -r /home/student/Documents/torcs-server/quickrace.xml" # The command to be executed to start a quickrace without visualisation
		return subprocess.Popen(bashCommand.split())								# Return subprocess so that it can be killed

	# Kill socket process
	def kill_process(self, server_proc):
		process = psutil.Process(server_proc.pid)						# Get process
		for proc in process.children(recursive=True):					# For all processes
			proc.kill()													# Kill them all
		process.kill()													# Kill it

	# ------------------------------------------------------------------------ Evolution ----------------------------------------------------------------------- #

	# Initialize population with individuals with random weights
	def init_population(self):
		population = []													# Initial list that is going to contain the population
		for i in range(self.POPULATION_SIZE):							# For all indiviuals to be created
			network = Network()											# Make network with random weights
			population.append(network)									# Append to population list
		return population 												# Return population list

	# Select best fraction of the population to make offspring
	def select_best(self, population):
		pop_list = [(member.fitness, member) for member in population]	# Make 2 dimensional list with the fitness of a individual as first argument
																		# and the individual itself as second argument
		pop_list = sorted(pop_list, key=lambda t: t[0], reverse=True)	# Sort according to fitness and reverse so that highest fitness is first in the list
		pop_list = [p[1] for i, p in enumerate(pop_list) \
			if i+1 <= len(pop_list)*self.FRACTION_BEST] 				# Select the first FRACTION_BEST of the list
		return pop_list													# Return the best fraction of the population

	# Generate 2 complentary children as result of mating of two parents
	def generate_child(self, parent1, parent2):
		child1 = []														# The initial list that is going to contain the weights of the first child		
		child2 = []														# The initial list that is going to contain the weights of the second child

		no_weights = sum([len(x) for y in parent1.weights_matrix for x in y])
		cutoff = random.randrange(no_weights)

		# The weight matrix of a network consist of 3 dimensions, so we have 3 nested loops to go though each weight value of both parents and
		# pass the weight along to one of the children.
		for i, (p1_dim1, p2_dim1) in \
			enumerate(zip(parent1.weights_matrix, parent2.weights_matrix)):		# Loop through first dimension

			c1_dim1, c2_dim1 = [], []									# Initial list that is going to contain the first dimensional weight vectors
			for j, (p1_dim2, p2_dim2) in enumerate(zip(p1_dim1, p2_dim1)):				# Loop though second dimension

				c1_dim2, c2_dim2 = [], []								# Initial list that is going to contain the second dimensional weight vectors
				for k, (p1_dim3, p2_dim3) in enumerate(zip(p1_dim2, p2_dim2)):			# Loop though third dimension

					# coinflip = random.randint(0,1)						# Simulated a coinflip, generates 1 or 0 with 50% chance each
					# if coinflip == 0:									# If 0 is drawn
					# 	c1_dim3, c2_dim3 = p1_dim3, p2_dim3				# Give weight of parent1 to child1 and weight of parent2 to child2
					# else:												# If 1 is drawn
					# 	c1_dim3, c2_dim3 = p2_dim3, p1_dim3				# Give weight of parent2 to child1 and weight of parent1 to child2

					ind = (i * len(p1_dim1) * len(p1_dim2)) + (j * len(p1_dim2)) + k

					if k < cutoff:
						c1_dim3, c2_dim3 = p1_dim3, p2_dim3
					else:
						c1_dim3, c2_dim3 = p2_dim3, p1_dim3

					# c1_dim3 = np.mean([p1_dim3, p2_dim3])
					# c2_dim3 = np.mean([p1_dim3, p2_dim3])

					c1_dim2.append(c1_dim3)								# Append
					c2_dim2.append(c2_dim3)								# Append
				c1_dim1.append(c1_dim2)									# Append
				c2_dim1.append(c2_dim2)									# Append
			child1.append(c1_dim1)										# Append
			child2.append(c2_dim1)										# Append

		c1 = Network(weights_matrix=child1)								# Make network object of weights for child1
		c2 = Network(weights_matrix=child2)								# Make network object of weights for child2
		return [c1, c2] 													# The children are born

	# Generate a new population given a old (best part of a) population
	def generate_children(self, population):
		children = []													# The initial list that is going to contain all children
		for i in range(len(population)):								# Loop through all parents
			parent1 = population[i]										# Parent1 is chosen by index

			numbers = list(range(0, i))									# Create a list of numbers that contains all indices, except for the one of parent1
			if i <= len(population):									# If the parent1 is not the last one
				numbers.extend(list(range(i+1, len(population))))		# Extend list with rest of the numbers

			for i in range(self.NUMBER_OF_PARTNERS):					# For the number of partners to be
				parent2 = population[random.choice(numbers)]			# Randomly choose partner for parent1
				children.extend(self.generate_child(parent1, parent2))	# Let the mating process begin		

		return children 												# Return entire new population

	# Mutates an individual
	def mutate(self, network):
		weights = network.weights_matrix								# Get the weight matrix of an individual
																		# Loop through all dimensions of the weight matrix
		for i in range(len(weights)):									# Loop though first dimension
			for j in range(len(weights[i])):							# Loop though second dimension
				for k in range(len(weights[i][j])):						# Loop through third dimension
					if random.uniform(0,1) <= 0.05:
						sig = 0.2
						weights[i][j][k] += random.gauss(0, sig)

						if weights[i][j][k] > 1:
							weights[i][j][k] = 1
						elif weights[i][j][k] < -1:
							weights[i][j][k] = -1
					# if random.random() <= 0.1:							# With a chance of 0.2
					# 	weights[i][j][k] += random.uniform(-.2, .2)		# Add or substract noise to weight

					# if random.random() <= 0.05:							# With 0.1 chance
					# 	weights[i][j][k] *= random.uniform(0.1, 2)		# Reduce or enlarge weight

					# if random.random() <= 0.02:							# With 0.05 chance
					# 	weights[i][j][k] = random.uniform(-1, 1)		# Replace weight with random value between -1 and 1

		return network 													# Return mutated individual

	# Mutates a part of an entire population
	def mutate_population(self, population):
		new_population = []												# List that is going to contain the partly mutated population	

		no_mut_individuals = int(self.FRACTION_MUTATE*len(population))	# The boundary of the population that is going to be mutated

		random.shuffle(population)										# Randomly shuffle population

		for i in range(no_mut_individuals):								# For index in range of number of individuals to be
			new_population.append(self.mutate(population[i]))			# Append mutated individual to new population list

		new_population.extend(population[no_mut_individuals:])			# Append rest of the unmutated population

		return new_population											# Return partly mutated population

	def adapt_xml(
		self, 
		random_track=True, 
		two_cars=False, 
		no_opponents=0
		):
		if random_track == True:
			tracks = [
				["road", 'aalborg'],
				["road", 'alpine-2'],
				["road", 'brondehach'],
				["road", 'corkscrew'],
				["road", 'e-track-1'],
				["road", 'e-track-3'],
				["road", 'e-track-6'],
				["road", 'eroad'],
				["road", 'g-track-1'],
				["road", 'g-track-2'],
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'g-track-3'],    # SUPER!!!!!!!!!!!
				["road", 'ruudskogen'],
				["road", 'spring'],
				["road", 'street-1'],
				["road", 'wheel-2']
			]

			track_type, track_name = random.choice(tracks)
			path = '/home/student/Documents/torcs-server/quickrace.xml'
			tree = ET.parse(path)
			root = tree.getroot()
					
			attr1 = root[1][1][0]
			attr2 = root[1][1][1]

			attr1.set('val', track_name)
			attr2.set('val', track_type)

			print("T R A C K 			", track_type, track_name)

		if no_opponents > 0:
			opponents_list = [
				'damned',
				'olethros',
				'bt',
				'tita',
				'inferno',
				'damned',
				'berniw',
				'lliaw'
			]

			opponents = [
				['scr_server', '0']
			]

			if two_cars == True:
				opponents.append(['scr_server', '1'])

			for i in range(no_opponents-len(opponents)):
				name = random.choice(opponents_list)

				opp_lst = []
				for opps in opponents:
					opp_lst.append(opps[0])
					
				opponents.append([name, str(opp_lst.count(name)+1)])
				
			for i in range(no_opponents):
				sublst = random.choice(opponents)
				module, idx = sublst
				opponents.remove(sublst)
				attr1 = root[4][3+i][0]
				attr2 = root[4][3+i][1]
				attr1.set('val', idx)
				attr2.set('val', module)

			print("O P P O N E N T S 	", opponents)		

		if random_track == True or no_opponents > 0:
			tree.write(path)

	# Makes an unique directory to store the weights
	def make_directory(self):
		dir_name = 'ea_output/output_' + str(datetime.datetime.now())	# Unique dir name generated using the current date and time
		if not os.path.exists(dir_name):								# If it not already exists
			os.makedirs(dir_name)										# Make directory
		return dir_name													# Return dir name

	# Saves weights of best network of a generation as pickle file
	def save_networks(self, dir_name, networks, gen_no):
		with open(dir_name+'/outfile_gen'+str(gen_no), 'wb') as fp:		# Open file
			pickle.dump(networks, fp)			# Write to file

	# The main loop of the evolution algorithm
	def evolve_population(
		self, 
		population, 
		two_cars=False, 
		random_track=True, 
		no_opponents=0, 
		print_ea_info=True, 
		print_car_values=False
		):
		dir_name = self.make_directory()								# Make directory to store networks

		all_networks = []												# Initialize list with all networks											

		for gen_no in range(self.NO_GENERATIONS):						# For all generations

			self.adapt_xml(
				random_track=random_track, 
				two_cars=two_cars, 
				no_opponents=no_opponents)								# Set random track and opponent

			individual_counter = 0
			for network in population:									# For every individual in a population
				individual_counter += 1

				server_proc = self.init_process()						# Initialize socket process

				result_queue = multiprocessing.Queue()					# Queue to store fitness values

				p1 = multiprocessing.Process(
					target=main, 
					args = (MyDriver(network=network), 
					self.no_frames_list[gen_no], 
					3001, 
					result_queue,
					print_car_values))
				p1.start()

				if two_cars == True:
					p2 = multiprocessing.Process(
						target=main, 
						args = (MyDriver(network=network), 
						self.no_frames_list[gen_no], 
						3002, 
						result_queue,
						print_car_values))
					p2.start()

				p1.join()

				if two_cars == True:	
					p2.join()

				self.fitness1 = result_queue.get()						# Get fitness from queue
				if two_cars == True:
					self.fitness2 = result_queue.get()				

				network.fitness = self.fitness1 + self.fitness2 		# Assign fitness to network object

				if print_ea_info == True:
					print()
					print("-------------")
					print()
					print("No frames 	", self.no_frames_list[gen_no])
					print()
					print("generation 	", gen_no)
					print("individual 	", individual_counter)
					print()
					if two_cars == True:
						print("Fitness1 	", self.fitness1)
						print("Fitness2 	", self.fitness2)
					print()
					print("Fitness 	", network.fitness)
					print()
					print("-------------")
					print()

				all_networks.append(network)							# Append network to all_network object
				self.kill_process(server_proc)							# Kill process

			self.save_networks(dir_name, all_networks, '_all')		# Save file with all networks
			self.save_networks(dir_name, population, gen_no)		# Save all networks
			print()
			print('__________________________________')
			print('__________________________________')
			print()

			best_networks = self.select_best(population)				# Select best fraction of the population to make offspring

			print()
			print("Fitness of best network from last generation", best_networks[0].fitness)

			children = self.generate_children(best_networks)			# Generate children population 

			population = self.mutate_population(children)				# Mutate part of the population

			print("Population size ", len(population))

			print()
			print('__________________________________')
			print('__________________________________')
			print()

			
		print()
		print("Training finished ( ͡° ͜ʖ ͡°).	")
		print("The dir name is 				", dir_name)
		print("The amount of generations is ", self.NO_GENERATIONS)
		return best_networks[0]											# Return final best network with highest fitness

	# Initialize population with random weights and evolve
	def evolve_best_individual(
		self, 
		init_population=False, 
		two_cars=False, 
		random_track=True, 
		no_opponents=0, 
		print_ea_info=True,
		print_car_values=False
		):
		if init_population == None:										# If no start population is given
			init_population = self.init_population()					# Initialize population with random weights

		return self.evolve_population(
			init_population, 
			two_cars=two_cars, 
			random_track=random_track, 
			no_opponents=no_opponents, 
			print_ea_info=print_ea_info,
			print_car_values=print_car_values
			)															# Return best individual after evolution