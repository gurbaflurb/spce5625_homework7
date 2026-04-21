import math

import numpy as np

from keplarianElements import KeplerianElements
import keHelperFunctions

class GroundSite():
    '''Class GroundSite takes in a geodetic latitude, longitude, height, and the radius from the center of the earth to the site.'''
    def __init__(self, lat, lon, height):
        self.lat = lat
        self.lon = lon
        self.height = height
        # self.r_vector = [X, Y, Z] # AKA ECEF coords
        # self.radius = radius
        self.ecef_coords = self.compute_ecef_coords()
        self.r_vector = self.ecef_coords
        self.radius = np.linalg.norm(self.r_vector)


    def compute_relative_pos(self, sv_pos: list):
        '''Takes in the ECEF coordinates for a Space Vehicle (SV) and compute the relative position between the defined ground site and the SV'''
        return np.subtract(sv_pos, self.r_vector)
    

    def compute_ecef_coords(self):
        # From WGS-84
        earth_radius = 6378137
        flattening = 1/298.257223563
        e2 = flattening * (2-flattening)

        lat_radians = math.radians(self.lat)
        lon_radians = math.radians(self.lon)

        Lamba = lon_radians
        Phi = lat_radians

        divsor = math.sqrt(1-e2*math.pow(math.sin(Phi), 2))

        x = ((earth_radius/divsor)+self.height) * math.cos(Phi)*math.cos(Lamba)
        y = ((earth_radius/divsor)+self.height) * math.cos(Phi)*math.sin(Lamba)
        z = (((earth_radius*(1-e2))/divsor)+self.height) * math.sin(Phi)

        return [x, y, z]

    def determine_azimuth_to_sv(self, sv_pos: list):
        '''Takes in the position of the space vehicle after its been converted to ECEF and then transformed to TOPO'''
        x = sv_pos[0]
        y = sv_pos[1]

        # Formula given in class
        azimuth = math.atan2(x, y)

        if azimuth < 0:
            azimuth = azimuth + (2*math.pi)

        if azimuth >= (2*math.pi):
            azimuth = azimuth % (2*math.pi)

        return azimuth
    

    def determine_elevation_to_sv(self, sv_pos: list):
        '''Takes in the position of the space vehicle after its been converted to ECEF and then transformed to TOPO'''

        x = sv_pos[0]
        y = sv_pos[1]
        z = sv_pos[2]

        # Formula given in class
        elevation = math.atan(z/math.sqrt(math.pow(x, 2)+math.pow(y, 2)))

        if elevation > (math.pi/2):
            elevation = elevation % (2*math.pi)

        while elevation < (-math.pi/2):
            elevation = elevation + (2*math.pi)
        

        return elevation


    def determine_az_el_to_sv(self, relative_pos) -> tuple:
        '''Takes in the position of the space vehicle after its been converted to ECEF and then transformed to TOPO'''

        azimuth = self.determine_azimuth_to_sv(relative_pos)
        elevation = self.determine_elevation_to_sv(relative_pos)

        return (azimuth, elevation)


    def estimate_orbit(self, observations: list) -> tuple:
        '''Estimates the orbit of a space vehicle based on a set of measurements from a defined ground site'''

        

        # Helps to keep track of which is which
        Lamba = math.radians(self.lon)
        Phi = math.radians(self.lat)
     
        transformation_matrix = [[-math.sin(Lamba), math.cos(Lamba), 0],
                                 [-math.sin(Phi)*math.cos(Lamba), -math.sin(Phi)*math.sin(Lamba), math.cos(Phi)],
                                 [math.cos(Phi)*math.cos(Lamba), math.cos(Phi)*math.sin(Lamba), math.sin(Phi)]]

        sv_tod_coord_list = []

        for observation in observations:
            # Compute the Topocentric range vector
            current_range = observations[observation]['range'] * 1852 # Convert to meters
            current_elevation = math.radians(observations[observation]['el']) # Convert to radians
            current_azimuth = math.radians(observations[observation]['az']) # Convert to radians

            new_coordinates = [current_range * math.cos(current_elevation) * math.sin(current_azimuth),
                               current_range * math.cos(current_elevation) * math.cos(current_azimuth),
                               current_range * math.sin(current_elevation)]
            
            # New Coordinates are rhoTOPO

            # Apply transformation matrix to range vector
            sv_ecef_range_vector =  np.dot(np.transpose(transformation_matrix), new_coordinates)

            # Compute site ECEF coords
            current_site_coords = keHelperFunctions.convert_eci_ecef(self.r_vector, observation*60)

            # Compute satellite ECEF vector
            sv_ecef_coords = current_site_coords + sv_ecef_range_vector

            # Compute GHA at the epoch of the measurement and convert the ECEF vector to TOD
            year = observations[observation]['time'].year
            month = observations[observation]['time'].month
            day = observations[observation]['time'].day
            hour = observations[observation]['time'].hour
            minute = observations[observation]['time'].minute
            second = observations[observation]['time'].second

            jd = keHelperFunctions.convert_date_to_jd(year, month, day, hour, minute, second)
            sv_tod_coord_list.append(keHelperFunctions.convert_ecef_tod(sv_ecef_coords, jd))



        # Estimate Velocity
        estimated_velocity_list = []

        for i in range(0, len(sv_tod_coord_list)-1):
            # This might be wrong, sub matrix from matrix ?!?!?!?!?!!?!?!
            estimated_velocity_list.append((sv_tod_coord_list[i+1]-sv_tod_coord_list[i])/60) # Hard coded to 60 since we know from hw10 it will always be 6

        tolerance = 1
        iterations = 0

        # Initialize the estimate matrix to zeros
        sata_matrix = [[0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]]

        satr_matrix = [[0],
                       [0],
                       [0],
                       [0],
                       [0],
                       [0]]
        
        A = [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]

        last_s = [sv_tod_coord_list[0][0], sv_tod_coord_list[0][1], sv_tod_coord_list[0][2], estimated_velocity_list[0][0], estimated_velocity_list[0][1], estimated_velocity_list[0][2]] # First three are initial position vector, last three are velocity vector

        while iterations < 10:

            delta_v = []

            # Compute the angular difference between this vector and the vectors cooresponding to all subsequent measurements
            for i in sv_tod_coord_list:
                if i is sv_tod_coord_list[0]:
                    continue # Skip the first iteration
                delta_v.append(math.acos(np.dot(sv_tod_coord_list[0], i)/(np.linalg.norm(sv_tod_coord_list[0])*np.linalg.norm(i))))

            # Compute f & g for each pair
            for i in range(0, len(estimated_velocity_list)):
                # Compute KE from position and velocity vectors
                temp_ke = KeplerianElements(sv_tod_coord_list[i][0], sv_tod_coord_list[i][1], sv_tod_coord_list[i][2], estimated_velocity_list[i][0], estimated_velocity_list[i][1], estimated_velocity_list[i][2])

                f = 1 - (np.linalg.norm(temp_ke.r_vector)/temp_ke.determine_p())*(1-math.cos(delta_v[i]))
                g = ((np.linalg.norm(temp_ke.r_vector)*np.linalg.norm(temp_ke.r_vector))/math.sqrt(temp_ke.mu * temp_ke.determine_p()))*math.sin(delta_v[i])


                # Compute ATA and ATR
                ak_matrix = [[f, 0, 0, g, 0, 0],
                            [0, f, 0, 0, g, 0],
                            [0, 0, f, 0, 0, g]]
                
                ata_matrix = np.dot(np.transpose(ak_matrix), ak_matrix)
                atr_matrix = np.dot(np.transpose(ak_matrix), temp_ke.r_vector)

                # Add matricies to sum matricies
                sata_matrix = sata_matrix + ata_matrix
                satr_matrix = satr_matrix + atr_matrix

            # Compute inverse of the ATA matrix
            inverse_sata_matrix = np.linalg.inv(sata_matrix)

            # Compute updated state estimate
            state_estimate_matrix = np.dot(inverse_sata_matrix, satr_matrix)


            for i in state_estimate_matrix:
                new_x = last_s[0] - i[0]
                new_y = last_s[1] - i[1]
                new_z = last_s[2] - i[2]
                new_xd = last_s[3] - i[3]
                new_yd = last_s[4] - i[4]
                new_zd = last_s[5] - i[5]

                new_pos = [new_x, new_y, new_z]
                new_vel = [new_xd, new_yd, new_zd]


            break

            if (np.linalg.norm([last_s[0], last_s[1], last_s[2]]) - np.linalg.norm(new_pos) < tolerance) and (np.linalg.norm([last_s[3], last_s[4], last_s[5]]) - np.linalg.norm(new_vel) < tolerance):
                break

            
        # Return position and velocity vector
        return ([last_s[0], last_s[1], last_s[2]], [last_s[3], last_s[4], last_s[5]])


    def determine_angle_between_two_sv(self, sv1_pos: list, sv2_pos: list):
        '''Given two position vectors for a space vehicle in TOPO, this returns the angle (in degrees) between the two vehicles from the reference frame of the ground site'''
        r3_vector = np.add(sv1_pos, sv2_pos)

        a = np.linalg.norm(sv1_pos)
        b = np.linalg.norm(sv2_pos)
        c = np.linalg.norm(r3_vector)

        theta = math.acos((math.pow(c, 2)- math.pow(a, 2) - math.pow(b, 2))/(2*a*b))

        return theta
