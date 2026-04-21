import math
import numpy as np
import tabulate


class KeplerianElements():
    '''
    Generates the Keplerian Elements given the 6 required parameters:
    X-Position, X-Velocity
    Y-Position, Y-Velocity
    Z-Position, Z-Velocity

    Depends on numpy for finding dot and cross products 
    '''
    def __init__(self, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel):
        self.initial_x_pos = x_pos
        self.initial_x_vel = x_vel
        self.initial_y_pos = y_pos
        self.initial_y_vel = y_vel
        self.initial_z_pos = z_pos
        self.initial_z_vel = z_vel

        # WGS84 values
        self.mu = 398600441800000
        self.wgs84_earth_eccentricity = 0.0818191908
        self.wgs84_earth_flattening = 1/298.257223563
        self.wgs84_rotation_of_earth = 0.000072921151467 # Radians per second
        self.wgs84_rotation_of_earth_vector = [0, 0, self.wgs84_rotation_of_earth]

        z_hat = [0, 0, 1]

        self.r_vector = np.array([self.initial_x_pos, self.initial_y_pos, self.initial_z_pos])
        self.r_dot_vector = np.array([self.initial_x_vel, self.initial_y_vel, self.initial_z_vel])

        self.h_vector = self.determine_h()
        self.angular_momentum_vector = self.h_vector

        self.inclination = self.determine_inclination(z_hat)

        self.n_hat = self.determine_n_hat(z_hat)
        
        self.raan = self.determine_right_ascension_of_ascending_node()

        self.b_vector = self.determine_b()
        
        self.eccentricity = self.determine_eccentricity()

        self.energy = self.determing_energy()
        
        self.semi_major_axis = self.determine_semi_major_axis()
        
        self.orbital_period = self.determine_orbital_period()
        self.tp = self.orbital_period

        self.apogee_radii = self.determine_apogee_radii()

        self.perigee_radii = self.determine_perigee_radii()

        self.aop = self.determine_argument_of_periapsis()

        self.nu = self.determine_true_anomaly()
        self.true_anomaly = self.nu

        self.eccentricity_anomaly = self.determine_eccentricity_anomaly()

        self.mean_anomaly = self.determine_mean_anomaly()
        
        self.mean_motion = self.determine_mean_motion()

        self.v0 = self.determine_v_0()

        self.E0 = self.determine_E_0(self.eccentricity_anomaly)

        self.perifocal_positions = self.determine_perifocal_position()
        
        self.velocity_components = self.determine_velocity_components() 


    def determine_semi_major_axis(self):
        return -(self.mu/(2 * self.energy))

    def determine_eccentricity(self):
        return np.linalg.norm(self.b_vector / self.mu)

    def determine_eccentricity_vector(self):
        return (((np.cross(self.r_dot_vector, self.h_vector))/self.mu)-self.r_vector/np.linalg.norm(self.r_vector))

    def determine_eccentricity_anomaly(self):
        '''Returns in radians'''
        # return math.asin((math.sin(self.nu)*math.sqrt(1-math.pow(self.eccentricity, 2)))/(1+self.eccentricity*math.cos(self.nu)))
    
        n_e = np.dot(self.r_vector, self.r_dot_vector)/math.sqrt(self.mu*self.semi_major_axis)
        d_e = 1 - (np.linalg.norm(self.r_vector)/self.semi_major_axis)
        r = math.atan2(n_e, d_e)

        if np.dot(self.r_vector, self.r_dot_vector) < 0:
            r = (2 * math.pi) + r

        return r
    
    def determine_arbitrary_eccentricity_anomaly(self, nu):
        return math.asin((math.sin(nu)*math.sqrt(1-math.pow(self.eccentricity, 2)))/(1+self.eccentricity*math.cos(nu)))


        n_e = np.dot(r_vector, r_dot_vector)/math.sqrt(self.mu*self.semi_major_axis)
        d_e = 1 - (np.linalg.norm(r_vector)/self.semi_major_axis)
        r = math.atan2(n_e, d_e)

        if np.dot(r_vector, r_dot_vector) < 0:
            r = (2 * math.pi) + r

        return r
    
    def determine_arbitrary_eccentric_anomaly(self, angle):
        '''Takes in an angle in degrees'''
        radian_angle = math.radians(angle)

        return math.acos((self.eccentricity + math.cos(radian_angle))/(1 + self.eccentricity * math.cos(radian_angle)))

    def determine_inclination(self, z_hat: list):
        '''Returns in radians, convert to degrees if you need'''
        return math.acos(np.dot(self.h_vector, z_hat)/(np.linalg.norm(self.h_vector)))

    def determine_right_ascension_of_ascending_node(self):
        '''Returns in radians, convert to degrees if you need'''
        r = math.atan2(self.n_hat[1], self.n_hat[0])

        # Correct for if we are in quadrant 3 or 4 
        if r < 0:
            r = r + (2 * math.pi)

        return r

    def determine_true_anomaly(self):
        '''Returns in radians, convert to degrees if you need'''
        r = math.acos((np.dot(self.r_vector, self.b_vector))/(np.linalg.norm(self.r_vector) * np.linalg.norm(self.b_vector)))

        # Correct for if we are in quadrant 3 or 4
        if np.dot(self.r_vector, self.r_dot_vector) < 0:
            r = (2 * math.pi) - r
        return r

    def determine_argument_of_periapsis(self):
        '''Returns in radians, convert to degrees if you need'''
        h_hat = self.h_vector/np.linalg.norm(self.h_vector)
        b_hat = self.b_vector/np.linalg.norm(self.b_vector)

        return math.atan2(np.dot(h_hat, np.cross(self.n_hat, b_hat)), np.dot(self.n_hat, b_hat))

    def determine_orbital_period(self):
        return 2 * math.pi * math.sqrt((math.pow(self.semi_major_axis, 3))/self.mu)

    def determine_apogee_radii(self):
        return self.semi_major_axis * (1 + self.energy)

    def determine_perigee_radii(self):
        return self.semi_major_axis * (1 - self.energy)

    def determing_energy(self):
        return (math.pow(np.linalg.norm(self.r_dot_vector), 2)/2) - (self.mu/np.linalg.norm(self.r_vector))

    def determine_h(self):
        '''Returns the H-Hat, the cross product of the position vector (r) and the velocity vector (r-dot)'''
        return np.cross(self.r_vector, self.r_dot_vector)

    def determine_n_hat(self, z_hat: list):
        '''Returns the N-Hat'''
        return np.cross(z_hat, self.h_vector)/np.linalg.norm(np.cross(z_hat, self.h_vector))

    def determine_b(self):
        return np.cross(self.r_dot_vector, self.h_vector) - (self.mu * (self.r_vector/np.linalg.norm(self.r_vector))) 

    def print_ke(self):
        # print(tabulate.tabulate([[self.r_vector, self.r_dot_vector, self.semi_major_axis, 
        #                          self.eccentricity, math.degrees(self.inclination), math.degrees(self.raan),
        #                          math.degrees(self.aop), math.degrees(self.true_anomaly), self.tp, self.apogee_radii, self.perigee_radii]],
        #                          headers=['Position Vector', 'Velocity Vector', 'Semi-Major Axis', 'Eccentricity', 'Inclination', 'RAAN', 'AoP', 'Nu', 'Orbit Period', 'Apogee Radii', 'Perigee Radii']))
        print(f'Position Vector       : {self.r_vector}')
        print(f'Velocity Vector       : {self.r_dot_vector}')
        print(f'Semi-major Axis       : {self.semi_major_axis} meters')
        print(f'Eccentricity          : {self.eccentricity}')
        print(f'Inclination           : {math.degrees(self.inclination)} Degrees')
        print(f'RAAN                  : {math.degrees(self.raan)} Degrees')
        print(f'Argument of Periapsis : {math.degrees(self.aop)} Degrees')
        print(f'Nu                    : {math.degrees(self.nu)} Degrees')
        print(f'Nu                    : {self.nu} radians')
        print(f'Orbit Period          : {self.tp} seconds')
        print(f'Apogee Radii          : {self.apogee_radii} meters')
        print(f'Perigee Radii         : {self.perigee_radii} meters')

    def determine_mean_motion(self):
        return math.sqrt(self.mu/math.pow(self.semi_major_axis, 3))

    def determine_mean_anomaly(self):
        return self.eccentricity_anomaly - self.eccentricity * math.sin(self.eccentricity_anomaly)
    
    def determine_arbitrary_mean_anomaly(self, eccentricity_anomaly):
        return eccentricity_anomaly - self.eccentricity * math.sin(eccentricity_anomaly)

    def determine_v_0(self):
        return self.nu

    def determine_E_0(self, eccentric_anomaly):
        return math.atan2(math.sin(eccentric_anomaly), math.cos(eccentric_anomaly))
    
    def determine_time_of_flight(self, mean_anomaly):
        return mean_anomaly/self.mean_motion
    
    def determine_arbitrary_time_of_flight(self, mean_anomaly, mean_motion):
        #return math.sqrt(math.pow(self.semi_major_axis,3)/self.mu)*mean_anomaly
        return mean_anomaly/mean_motion
    
    def determine_time_to_angle(self, angle, perigee_passes=0):
        '''Provided an angle from 0-360, returns the seconds to reach that angle from Nu, only works for 0-180'''
        E_1 = self.determine_arbitrary_eccentric_anomaly(angle)

        pt_1 = E_1 - self.eccentricity * math.sin(E_1)
        pt_2 = self.mean_anomaly

        time_to_angle = math.sqrt(math.pow(self.semi_major_axis, 3)/self.mu) * ((2 * math.pi * perigee_passes) + pt_1 - pt_2)

        if time_to_angle < 0:
            time_to_angle = time_to_angle + 2 * math.pi
        
        return time_to_angle

    def determine_location_after_n_seconds(self, seconds, nu):
        '''Implemented with Keplers Equation, Nu is expected in degrees'''
        cur_E0 = math.radians(nu)
        #cur_E0 = self.mean_anomaly

        for i in range(0,10000):
            cur_E0 = cur_E0 + (self.mean_motion * seconds + self.mean_anomaly - (cur_E0 - self.eccentricity * math.sin(cur_E0)))/(1 - self.eccentricity * math.cos(cur_E0))

        perigee_passes = math.trunc((cur_E0 - nu)/(2*math.pi))

        # Correct if we perform multiple orbits
        cur_E0 = cur_E0 % (2*math.pi)

        return (cur_E0, perigee_passes)
    
    def determine_true_anomaly_from_eccentric_anomaly(self, eccentric_anomaly):
        nu = math.atan2((math.sin(eccentric_anomaly)*math.sqrt(1-math.pow(self.eccentricity, 2)))/(1-self.eccentricity*math.cos(eccentric_anomaly)), (math.cos(eccentric_anomaly)-self.eccentricity)/(1-self.eccentricity*math.cos(eccentric_anomaly)))
        
        if nu < 0:
            nu = nu + 2*math.pi
        
        return nu

    def determine_p(self):
        return math.pow(np.linalg.norm(self.h_vector), 2)/self.mu
        #return self.semi_major_axis*(1-math.pow(self.eccentricity, 2))

    # Returns an x and y
    def determine_perifocal_position(self) -> list:
        '''Returns the perifocal position vector for the initially provided position and velocity vectors'''
        x = (np.linalg.norm(self.r_vector) * math.cos(self.nu)).item()
        y = (np.linalg.norm(self.r_vector) * math.sin(self.nu)).item()

        return [x, y, 0]

    # Returns an x and y
    def determine_arbitrary_perifocal_position(self, nu, position: np.array) -> list:
        '''Returns the perifocal position vector for the initially provided position and velocity vectors'''
        x = (np.linalg.norm(position) * math.cos(nu)).item()
        y = (np.linalg.norm(position) * math.sin(nu)).item()

        return [x, y, 0]

    # Returns x_dot, and y_dot
    def determine_velocity_components(self) -> list:
        '''Returns the perifocal velocity components for the initially provided position and velocity vectors'''
        x_dot = -math.sqrt(self.mu/self.determine_p())*math.sin(self.nu)
        y_dot = (math.sqrt(self.mu/self.determine_p())*(self.eccentricity+math.cos(self.nu))).item()

        return [x_dot, y_dot, 0]
    
    def determine_arbitrary_velocity_components(self, nu) -> list:
        '''Returns the perifocal position and velocity components for a provided Nu value based on the initially provided orbital parameters'''
        x_dot = -(math.sqrt(self.mu/self.determine_p())*math.sin(nu))
        y_dot = (math.sqrt(self.mu/self.determine_p())*(self.eccentricity+math.cos(nu))).item()

        return [x_dot, y_dot, 0]

    def determine_r(self):
        '''Returns the value of r for the initial orbit provided to the class'''
        return self.determine_p()/(1+self.eccentricity*math.cos(self.nu))
    
    def determine_new_r_dot(self, nu):
        '''Takes in a Nu in radians, not degrees'''
        return math.sqrt(self.mu/self.determine_p())*self.eccentricity*math.sin(nu)

    def determine_arbitrary_r(self, nu):
        '''Returns a value of r at a specified nu within the initial orbit provided by the class'''
        return self.determine_p()/(1+self.eccentricity*math.cos(nu))

    def determine_f(self, nu, delta_nu):
        '''Takes in two angles, nu, and the delta nu as degrees'''
        radian_nu = math.radians(nu)
        radian_delta_nu = math.radians(delta_nu)

        r = self.determine_p()/(1+self.eccentricity*math.cos(radian_delta_nu + radian_nu))
        #r = self.determine_p()/(1+self.eccentricity*math.cos(nu))

        return 1 - (r/self.determine_p())*(1 - math.cos(radian_delta_nu))

    def determine_f_delta_e(self, delta_e):
        return 1 - (self.semi_major_axis/np.linalg.norm(self.r_vector))*(1-math.cos(delta_e))

    def determine_g(self, nu, delta_nu):
        '''Takes in two angles, nu, and the delta nu as degrees'''
        radian_nu = math.radians(nu)
        radian_delta_nu = math.radians(delta_nu)

        r = self.determine_p()/(1+self.eccentricity*math.cos(radian_delta_nu + radian_nu))

        return ((r*np.linalg.norm(self.r_vector))/(math.sqrt(self.mu * self.determine_p()))) * math.sin(radian_delta_nu)
    
    def determine_g_delta_e(self, t_initial, t_final, delta_e):
        '''Takes in two angles, nu, and the delta nu as degrees'''

        return (t_final - t_initial) - math.sqrt(math.pow(self.semi_major_axis, 3)/self.mu)*(delta_e-math.sin(delta_e))

    def determine_g_dot(self, delta_nu, nu=0):
        '''Takes in the angle delta_nu as degrees'''
        #perifocal_positions = self.determine_arbitrary_perifocal_position(math.radians(nu))
        #perifocal_velocitys = self.determine_arbitrary_velocity_components(math.radians(nu))
        
        # return (perifocal_positions[0]*perifocal_velocitys[1]-perifocal_velocitys[0]*perifocal_positions[1])/np.linalg.norm(np.dot(self.determine_perifocal_position(), self.determine_velocity_components()))



        radian_delta_nu = math.radians(delta_nu)

        return 1 - (np.linalg.norm(self.r_vector)/self.determine_p()) * (1 - math.cos(radian_delta_nu))

    def determine_arbitrary_g_dot(self, initial_r_vector, delta_nu):
        '''Takes in the angle delta_nu as degrees'''
        radian_delta_nu = math.radians(delta_nu)

        p1 = np.linalg.norm(initial_r_vector)/self.determine_p()
        p2 = 1 - math.cos(radian_delta_nu)

        r = 1 - ((p1) * (p2))

        return r

    def determine_f_dot(self, nu, delta_nu):
        '''Takes in two angles, nu, and the delta nu as degrees'''
        radian_nu = math.radians(nu)
        radian_delta_nu = math.radians(delta_nu)

        r = self.determine_p()/(1+self.eccentricity*math.cos(radian_nu + radian_delta_nu))

        return math.sqrt(self.mu/self.determine_p()) * (math.tan(radian_delta_nu/2)) * (((1-math.cos(radian_delta_nu))/self.determine_p()) - (1/r) - (1/np.linalg.norm(self.r_vector)))

    def determine_arbitrary_f_dot(self, nu, delta_nu, r_vector):
        '''Takes in two angles, nu, and the delta nu as degrees'''
        radian_nu = math.radians(nu)
        radian_delta_nu = math.radians(delta_nu)

        r = self.determine_p()/(1+self.eccentricity*math.cos(radian_nu)) #+ radian_delta_nu))

        return math.sqrt(self.mu/self.determine_p()) * (math.tan(radian_delta_nu/2)) * (((1-math.cos(radian_delta_nu))/self.determine_p()) - (1/r) - (1/np.linalg.norm(r_vector)))
    
    def determine_f_dot_new(self, f, g, g_dot):
        return (f * g_dot -1)/g

    def determine_new_velocity(self, f_dot, g_dot):
        '''Determine the new velocity vector at a new delta-v'''
        return np.array(self.perifocal_positions) * f_dot + np.array(self.velocity_components) * g_dot
    
    def determine_arbitrary_new_velocity(self, f_dot, g_dot, current_position, velocity_components):
        '''Determine the new velocity vector at a new delta-v'''
        return np.array(current_position) * f_dot + np.array(velocity_components) * g_dot

    def determine_new_position(self, f, g):
        '''Determine the new position vector at a new delta-v'''
        return np.array(self.perifocal_positions) * f + np.array(self.velocity_components) * g

    def determine_arbitrary_new_position(self, f, g, current_position, velocity_components):
        '''Determine the new position vector at a new delta-v'''
        return np.array(current_position) * f + np.array(velocity_components) * g
    
    def convert_perifocal_to_equinoctial(self):
        '''Converts the Keplarian Elements for a given orbit and returns a tuple with all the Equinoctial values'''
        # Direct Set: 0 <= i < 180
        if self.inclination >= 0 and self.inclination < 180:
            semi_major_axis = self.semi_major_axis
            h = self.eccentricity * math.sin(self.aop + self.nu)
            k = self.eccentricity * math.cos(self.aop + self.nu)
            p = math.tan(self.inclination/2) * math.sin(self.nu)
            q = math.tan(self.inclination/2) * math.cos(self.nu)

            # Lambda, but not a python lambda function
            l = self.mean_anomaly + self.aop + self.nu



        # Retrograde Set: 0 < i <= 180
        elif self.inclination > 0 and self.inclination <= 180:
            semi_major_axis = self.semi_major_axis
            h = self.eccentricity * math.sin(self.aop - self.nu)
            k = self.eccentricity * math.cos(self.aop - self.nu)
            p = (1/math.tan(self.inclination/2)) * math.sin(self.nu)
            q = (1/math.tan(self.inclination/2)) * math.cos(self.nu)

            # Lambda, but not a python lambda function
            l = self.mean_anomaly + self.aop - self.nu

        return (semi_major_axis, h, k, p, q, l)

    def convert_coordinates_to_uvw(self) -> tuple:
        '''Convert given ECI coordinates (r_vector) and its cooresponding velocities (r_dot_vector) to UVW coordinates.'''
        u = self.r_vector/np.linalg.norm(self.r_vector)
        w = (np.cross(self.r_vector, self.r_dot_vector))/np.linalg.norm(np.linalg.cross(self.r_vector, self.r_dot_vector))
        v = np.cross(w, u)

        return (u, w, v)

    def convert_coordinates_to_lvlh(self) -> tuple:
        '''Convert given ECI coordinates (r_vector) and its cooresponding velocities (r_dot_vector) to LVLH coordinates.'''
        z = (- self.r_vector)/np.linalg.norm(self.r_vector)
        y = np.cross(self.r_dot_vector, self.r_vector)/np.linalg.norm(np.cross(self.r_dot_vector, self.r_vector))
        x = np.cross(y, z)

        return (x, y, z)

    def convert_perifocal_to_eci(self) -> tuple:
        '''Converts manually provided perifocal values to ECI coordinates. Uses radians and not degrees. Passing in degrees will mess up the calculation'''

        # Hard coded because a nice solution for this exists (Pulled from slide 74)
        x = [ math.cos(self.raan)*math.cos(self.aop) - math.sin(self.raan)*math.sin(self.aop)*math.cos(self.inclination), -math.cos(self.raan)*math.sin(self.aop) - math.sin(self.raan)*math.cos(self.aop)*math.cos(self.inclination), math.sin(self.raan)*math.sin(self.inclination)]
        y = [ math.sin(self.raan)*math.cos(self.aop) + math.cos(self.raan)*math.sin(self.aop)*math.cos(self.inclination), -math.sin(self.raan)*math.sin(self.aop)+math.cos(self.raan)*math.cos(self.aop)*math.cos(self.inclination), -math.cos(self.raan)*math.sin(self.inclination)]
        z = [ math.sin(self.inclination)*math.sin(self.aop), math.sin(self.inclination)*math.cos(self.aop), math.cos(self.inclination)]

        return (x, y, z)

    def rotate_uvw_about_x(self, uvw, angle):
        '''Pass in the angle in degrees you would like to rotate your plane '''
        radian_angle = math.radians(angle)

        rotation_matrix = np.matrix([[1.0, 0.0, 0.0],
                                     [0.0, math.cos(radian_angle), math.sin(radian_angle)],
                                     [0.0, -math.sin(radian_angle), math.cos(radian_angle)]])

        return np.dot(rotation_matrix, uvw)
    
    def rotate_uvw_about_y(self, uvw, angle):
        '''Pass in the angle in degrees you would like to rotate your plane '''
        radian_angle = math.radians(angle)

        rotation_matrix = np.matrix([[math.cos(radian_angle), 0, -math.sin(radian_angle)],
                                     [0, 1, 0],
                                     [math.sin(radian_angle), 0, math.cos(radian_angle)]])

        return np.dot(rotation_matrix, uvw)
    
    def rotate_uvw_about_z(self, uvw, angle):
        '''Pass in the angle in degrees you would like to rotate your plane '''
        radian_angle = math.radians(angle)

        rotation_matrix = np.matrix([[math.cos(radian_angle), math.sin(radian_angle), 0],
                                     [-math.sin(radian_angle), math.cos(radian_angle), 0],
                                     [0, 0, 1]])

        return np.dot(rotation_matrix, uvw)

    def convert_eci_to_wgs84ECEF(self, cur_pos, cur_vel):
        # Position vector is identical to ECI
        # Velocity needs to be adjusted thanks to the rotation of the planet

        ecef_position = self.convert_perifocal_to_eci()[2]
        ecef_velocity = self.r_dot_vector - np.cross(self.wgs84_rotation_of_earth_vector, ecef_position)
        
        return (ecef_position, ecef_velocity)
    
    def compute_argument_of_latitude(self):
        '''Compute the argument of latitude. Returns in Radians'''
        arglat = self.determine_argument_of_periapsis()+ self.determine_true_anomaly()

        return arglat
