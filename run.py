#! /usr/bin/env python3

from ea import EA

EA = EA(NO_GENERATIONS = 60, POPULATION_SIZE = 300, INIT_NO_FRAMES = 300, FINAL_NO_FRAMES = 4000, FRACTION_MUTATE = 0.3, NUMBER_OF_PARTNERS = 300)

best_network = EA.evolve_best_individual()