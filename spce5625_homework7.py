import numpy as np
from pprint import pprint
import math


def compute_receiver_pos_estimate(gps: list, user_pos: list):
    x = math.pow(gps[0] - user_pos[0], 2)
    y = math.pow(gps[1] - user_pos[1], 2)
    z = math.pow(gps[2] - user_pos[2], 2)

    return math.sqrt(x + y + z)


def compute_directional_derivatives(gps: list, user_pos: list, receiver_pos_estimate):
    x = (gps[0] - user_pos[0])/receiver_pos_estimate
    y = (gps[1] - user_pos[1])/receiver_pos_estimate
    z = (gps[2] - user_pos[2])/receiver_pos_estimate
    t = -1

    return [x, y, z, t]


# We have the X, Y, Z coordinates for all the GPS vehicles, and the user position
gps1 = [15524471.175, -16649826.222, 13512272.387]
gps2 = [-2304058.534, -23287906.465, 11917038.105]
gps3 = [16680243.357, -3069625.561, 20378551.047]
gps4 = [-14799931.395, -21425358.24, 6069947.224]
user_pos = [-730000, -5440000, 3230000]


print(f'SV 15: {gps1}')
print(f'SV 27: {gps2}')
print(f'SV 31: {gps3}')
print(f'SV 7 : {gps4}')
print(f'User : {user_pos}')

gps1_receiver_pos_esimate = compute_receiver_pos_estimate(gps1, user_pos)
gps2_receiver_pos_esimate = compute_receiver_pos_estimate(gps2, user_pos)
gps3_receiver_pos_esimate = compute_receiver_pos_estimate(gps3, user_pos)
gps4_receiver_pos_esimate = compute_receiver_pos_estimate(gps4, user_pos)

print(f'R_0: {gps1_receiver_pos_esimate}')
print(f'R_1: {gps2_receiver_pos_esimate}')
print(f'R_2: {gps3_receiver_pos_esimate}')
print(f'R_3: {gps4_receiver_pos_esimate}')

gps1_directional_derivative = compute_directional_derivatives(gps1, user_pos, gps1_receiver_pos_esimate)
gps2_directional_derivative = compute_directional_derivatives(gps2, user_pos, gps2_receiver_pos_esimate)
gps3_directional_derivative = compute_directional_derivatives(gps3, user_pos, gps3_receiver_pos_esimate)
gps4_directional_derivative = compute_directional_derivatives(gps4, user_pos, gps4_receiver_pos_esimate)

print(f'Direction Derivative 1: {gps1_directional_derivative}')
print(f'Direction Derivative 2: {gps2_directional_derivative}')
print(f'Direction Derivative 3: {gps3_directional_derivative}')
print(f'Direction Derivative 4: {gps4_directional_derivative}')

a_matrix = [gps1_directional_derivative,
            gps2_directional_derivative,
            gps3_directional_derivative,
            gps4_directional_derivative]

a_matrix_transpose = np.transpose(a_matrix)

halfway = np.dot(a_matrix_transpose, a_matrix)

result = np.linalg.inv(halfway)

pprint(result)

p00 = result[0][0]
p11 = result[1][1]
p22 = result[2][2]
p33 = result[3][3]

print(f'P00 : {p00}')
print(f'P11 : {p11}')
print(f'P22 : {p22}')
print(f'P33 : {p33}')

gdop = math.sqrt(p00+p11+p22+p33)
pdop = math.sqrt(p00+p11+p22)
tdop = math.sqrt(p33)
hdop = math.sqrt(p00+p11)
vdop = math.sqrt(p22)

print(f'GDOP: {gdop}')
print(f'PDOP: {pdop}')
print(f'TDOP: {tdop}')
print(f'HDOP: {hdop}')
print(f'VDOP: {vdop}')
