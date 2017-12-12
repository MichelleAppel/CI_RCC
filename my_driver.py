from pytocl.driver import Driver
from pytocl.car import State, Command
from network import Network
import numpy as np
import math

def fitness_function(distance_from_start, distance_raced, damage, offroad_count, race_position, last_lap_time, turn_around_count, negative_speed_count):
	if (turn_around_count > 0 or negative_speed_count > 0) and (distance_from_start > distance_raced + 100):
		return -(turn_around_count + negative_speed_count)*(1+damage)*(1+offroad_count)
	else:
		return (distance_from_start) / ((1+damage)*(1+offroad_count)*race_position)

class MyDriver(Driver):
	# Override the `drive` method to create your own driver

	def __init__(self, logdata=True, network=Network()):
		self.network = network
		super().__init__(logdata)

	def drive(self, carstate: State, offroad_count, turn_around_count, negative_speed_count) -> Command:
		command = Command()

		max_dist = 200.0
		react_dist = 200.0
		max_angle = 180.0
		max_speed = 200.0 / 3.6
		max_rpm = 10000

		# left_side = list(carstate.distances_from_edge[:9])[::-1]
		# right_side = list(carstate.distances_from_edge[10:])
		middle_sens = carstate.distances_from_edge[9]

		opps = list(carstate.opponents)
		for i, dist in enumerate(opps):
			if dist == 200.:
				opps[i] = 0.

		# back_opp = opponents[0]
		# left_opp = opponents[1:18][::-1]
		# forward_opp = opponents[18]
		# right_opp = opponents[19:]

		# even_sens = [x + y for x,y in zip(left_side, right_side)]
		# uneven_sens = [x - y for x,y in zip(left_side, right_side)]
		
		# edges = even_sens + uneven_sens + [middle_sens]
		edges = [np.exp(-x/react_dist) for x in carstate.distances_from_edge] 

		# even_opp = even_sens = [x + y for x,y in zip(left_opp, right_opp)]
		# uneven_opp = even_sens = [x - y for x,y in zip(left_opp, right_opp)]

		# opps = even_opp + uneven_opp + [forward_opp] + [back_opp]

		offroad = False

		if middle_sens == -1.:
			offroad_count += 1
			offroad = True
			edges = 19 * [0]
		else:
			for i, dist in enumerate(edges):
				if dist > 0:
					edges[i] = np.exp(-dist/react_dist)
				else:
					edges[i] = -np.exp(dist/react_dist)

		turn_around = False
		if carstate.angle > 100 or carstate.angle < -100:
			turn_around_count += 1
			turn_around = True

		negative_speed = False
		if carstate.speed_x < 0:
			negative_speed_count += 1
			negative_speed = True

		for i, dist in enumerate(opps):
			if dist != 0.:
				opps[i] = np.exp(-dist/react_dist)

		input_vector = edges + [carstate.speed_x/max_speed] + [carstate.speed_y/max_speed] + [carstate.speed_z/max_speed] + [carstate.angle/max_angle] + \
		[carstate.distance_from_center] + [carstate.rpm/max_rpm] #+ opps

		# print("LEN", len(input_vector))

		output_vector = self.network.NN(input_vector)

		# speed_value = 200. * (output_vector[0] + 1.)/2.
		speed_value = output_vector[0]

		# Combination of acceleration and brake
		acceleration_value = 0.
		brake_value = 0.

		if speed_value > 0:
			acceleration_value = speed_value
		elif speed_value < 0:
			brake_value = -speed_value

		steer_value = output_vector[1]

		if not math.isnan(output_vector[2]):
			gear_value = int(((output_vector[2] + 1) / 2) * 5 + 1)
		else:
			gear_value = 1

		command.gear = gear_value
		command.accelerator = acceleration_value
		command.brake = brake_value		
		command.steering = steer_value

		# self.accelerate(carstate, speed_value, command)
		# self.steer(carstate, steer_value, command)


		fitness = fitness_function(carstate.distance_from_start, carstate.distance_raced, carstate.damage, offroad_count, carstate.race_position, carstate.last_lap_time, turn_around_count, negative_speed_count)


		# print('------')
		# # print()
		# # print("input_vector        ", input_vector)
		# # print()
		# # print("weights_matrix      ", np.array(self.network.weights_matrix))
		# # print()

		# # print("output vector       ", output_vector)

		# print()
		# print("acceleration_value  ", acceleration_value)
		# print("brake_value         ", brake_value)
		# print("steer_value         ", steer_value)
		# print("gear_value          ", gear_value)
		# print()

		# # print("input vector        ", input_vector)

		# print("turn_around_count				" , turn_around_count)
		# print("negative speed count 			" , negative_speed_count)
		# print("carstate.speed_x					" , carstate.speed_x )
		# print("carstate.speed_y					" , carstate.speed_y )
		# print("carstate.speed_z					" , carstate.speed_z )

		# print()
		# print("distance_from_start ", carstate.distance_from_start)
		# print("distance_raced      ", carstate.distance_raced)
		# print("damage              ", carstate.damage)	
		# print("offroad             ", offroad)	
		# print("offroad count       ", offroad_count)
		# print("race_position       ", carstate.race_position)
		# print()
		# # print(list(carstate.opponents))
		# # print()
		# print("fitness             ", fitness)
		# # print()

		return command, fitness, offroad_count, turn_around_count, negative_speed_count
