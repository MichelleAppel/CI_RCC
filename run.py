#! /usr/bin/env python3

from ea import EA
import pickle


def init_population():
	path = "ea_output/output_2017-12-08 23:00:02.194085/outfile_gen2"

	allnetworks = pickle.load( open(path, "rb" ))

	networklist = []
	for network in allnetworks:
		networklist.append([network.fitness, network])

	networklist = sorted(networklist, key=lambda t: t[0], reverse=True)
	networklist = [p[1] for i, p in enumerate(networklist) \
			if i+1 <= POPULATION_SIZE] 

	return networklist


NO_GENERATIONS = 3
POPULATION_SIZE = 18
INIT_NO_FRAMES = 200
FINAL_NO_FRAMES = 500
FRACTION_MUTATE = 0.3
NUMBER_OF_PARTNERS = 3

EA = EA(
	NO_GENERATIONS = NO_GENERATIONS, 
	POPULATION_SIZE = POPULATION_SIZE, 
	INIT_NO_FRAMES = INIT_NO_FRAMES, 
	FINAL_NO_FRAMES = FINAL_NO_FRAMES, 
	FRACTION_MUTATE = FRACTION_MUTATE, 
	NUMBER_OF_PARTNERS = NUMBER_OF_PARTNERS
	)

best_network = EA.evolve_best_individual(
	init_population=None, 
	two_cars=False, 
	random_track=True, 
	no_opponents=0, 
	print_ea_info=True, 
	print_car_values=False
	)