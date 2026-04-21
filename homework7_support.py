import math

import keHelperFunctions
from groundSite import GroundSite

gps1_ecef = [15524471.175, -16649826.222, 13512272.387]
gps2_ecef = [-2304058.534, -23287906.465, 11917038.105]
gps3_ecef = [16680243.357, -3069625.561, 20378551.047]
gps4_ecef = [-14799931.395, -21425358.24, 6069947.224]
user_pos_ecef = [-730000, -5440000, 3230000]

lat, lon, alt = keHelperFunctions.compute_lat_lon_alt(user_pos_ecef)

user_gs = GroundSite(lat, lon, alt)

gps1_topo = keHelperFunctions.convert_ecef_topocentric(user_gs.lat, user_gs.lon, gps1_ecef)
gps2_topo = keHelperFunctions.convert_ecef_topocentric(user_gs.lat, user_gs.lon, gps2_ecef)
gps3_topo = keHelperFunctions.convert_ecef_topocentric(user_gs.lat, user_gs.lon, gps3_ecef)
gps4_topo = keHelperFunctions.convert_ecef_topocentric(user_gs.lat, user_gs.lon, gps4_ecef)

az, el = user_gs.determine_az_el_to_sv(gps1_topo)
print(f'AZ to GPS1: {math.degrees(az)}')
print(f'EL to GPS1: {math.degrees(el)}')
print()
az, el = user_gs.determine_az_el_to_sv(gps2_topo)
print(f'AZ to GPS2: {math.degrees(az)}')
print(f'EL to GPS2: {math.degrees(el)}')
print()
az, el = user_gs.determine_az_el_to_sv(gps3_topo)
print(f'AZ to GPS3: {math.degrees(az)}')
print(f'EL to GPS3: {math.degrees(el)}')
print()
az, el = user_gs.determine_az_el_to_sv(gps4_topo)
print(f'AZ to GPS4: {math.degrees(az)}')
print(f'EL to GPS4: {math.degrees(el)}')
print()


