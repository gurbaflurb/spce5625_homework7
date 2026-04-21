import yaml
import math
import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt

from keplarianElements import KeplerianElements


# Read in a yaml that has all the initial vectors for position and velocity
def read_in_yaml(file_name):
    with open(file_name, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return data

def read_in_csv(file_name: str, headers=True):
     '''Reads in a CSV file, if headers is set to True, then it skips the first row (default option)'''
     return_data = []

     with open(file_name, 'r') as f:
          data = csv.reader(f)

          if headers:
               next(data)
          for row in data:
               return_data.append(row)
     return return_data

def print_data_to_csv(file_name, headers: list, data: list):
     '''Takes in a file name, the headers for your file, and a list of lists for your data. Built initially for Exam 2'''
     with open(f'{file_name}', 'w') as csv_out:
        csvwriter = csv.writer(csv_out, delimiter=',')
        csvwriter.writerow(headers)
        for line in data:
            csvwriter.writerow(line)



def convert_arbitrary_perifocal_to_eci(a, e, inclination, raan, aop, nu) -> tuple:
        '''Converts manually provided perifocal values to ECI coordinates. Uses radians and not degrees. Passing in degrees will mess up the calculation'''

        # Hard coded because a nice solution for this exists (Pulled from slide 74)
        x = [ math.cos(raan)*math.cos(aop) - math.sin(raan)*math.sin(aop)*math.cos(inclination), -math.cos(raan)*math.sin(aop) - math.sin(raan)*math.cos(aop)*math.cos(inclination), math.sin(raan)*math.sin(inclination)]
        y = [ math.sin(raan)*math.cos(aop)+math.cos(raan)*math.sin(aop)*math.cos(inclination), -math.sin(raan)*math.sin(aop)+math.cos(raan)*math.cos(aop)*math.cos(inclination), -math.cos(raan)*math.sin(inclination)]
        z = [ math.sin(inclination)*math.sin(aop), math.sin(inclination)*math.cos(aop), math.cos(inclination)]

        return (x, y, z)



def find_arbitrary_position_and_velocity_vector(a: float, eccentricity: float, nu: float) -> tuple:
    '''Returns a tuple in the form position_vector, velocity_vector'''
    mu = 398600441800000.0 # From WGS84
    
    perifocal = a*(1.0-math.pow(eccentricity, 2))
    radius = perifocal/(1.0 + eccentricity*math.cos(nu))

    return ([radius*math.cos(nu), radius*math.sin(nu), 0.0],
            [-math.sqrt(mu/perifocal)*math.sin(nu), math.sqrt(mu/perifocal)*(eccentricity+math.cos(nu)), 0.0])



def keplarian_rk4(r_vector, r_dot_vector, step, mu):
     '''Takes in the position vector (r_vector), velocity vector (r_dot_vector), r (norm of r_vector), step (Size of step between each round), and mu (Probably from WGS 84).
     Mu is typically provided as meters cubed over seconds squared.'''

     r0norm = np.linalg.norm(r_vector)
     rd_a_pt = (-mu/math.pow(r0norm, 3))*r_vector

     r_a = r_vector + (step/2)*r_dot_vector
     rd_a = r_dot_vector + (step/2)*rd_a_pt
     k1 = [r_dot_vector, rd_a_pt]

     ranorm = np.linalg.norm(r_a)
     rd_b_pt = (-mu/math.pow(ranorm, 3))*r_a

     r_b = r_vector+(step/2)*rd_a
     rd_b = r_dot_vector+(step/2)*rd_b_pt
     k2 = [rd_a, rd_b_pt]

     rbnorm = np.linalg.norm(r_b)
     rd_c_pt = (-mu/math.pow(rbnorm, 3))*r_b

     r_c = r_vector+(step)*rd_b
     rd_c = r_dot_vector+(step)*rd_c_pt
     k3 = [rd_b, rd_c_pt]

     k4 = [rd_c, (-mu/math.pow(np.linalg.norm(r_c), 3))*r_c]

     step_position_solution = r_vector + step*( ((k1[0])/6) + ((k2[0])/3) + ((k3[0])/3) + ((k4[0])/6) )

     step_velocity_solution = r_dot_vector + step*( ((k1[1])/6) + ((k2[1])/3) + ((k3[1])/3) + ((k4[1])/6) )

     return (step_position_solution, step_velocity_solution)



def keplarian_rk4_oblate_earth(r_vector, r_dot_vector, step):
     '''Takes into account the Earth is Oblate. Takes in the position vector (r_vector), velocity vector (r_dot_vector), r (norm of r_vector), and step (Size of step between each round)
     Mu in this case is not provided by the user but instead defined by the WGS 84 standard for this specific implementation.'''

     # Defined by WGS 84
     earth_radius = 6378137 # This value is in meters
     earth_j2 = 0.00108262998905194
     mu = 398600441800000 # This value is in meters cubed per second squared

     # K1
     r = math.sqrt(math.pow(r_vector[0], 2) + math.pow(r_vector[1], 2) + math.pow(r_vector[2], 2))

     a_pt_s = r_vector[2]/r
     a_pt_1 = -(mu/math.pow(r, 3))
     a_pt_2 = (1+((3*earth_j2)/2)*math.pow(earth_radius/r, 2)*(1-5*math.pow(a_pt_s, 2)))*r_vector
     
     a = a_pt_1 * a_pt_2

     r_a = r_vector + (step/2)*r_dot_vector
     rd_a = r_dot_vector + (step/2)*a
     k1 = [r_dot_vector, a]

     # K2
     r = math.sqrt(math.pow(r_a[0], 2) + math.pow(r_a[1], 2) + math.pow(r_a[2], 2))

     b_pt_s = r_a[2]/r
     b_pt_1 = -(mu/math.pow(r, 3))
     b_pt_2 = (1+((3*earth_j2)/2)*math.pow(earth_radius/r, 2)*(1-5*math.pow(b_pt_s, 2)))*r_a
     
     a = b_pt_1 * b_pt_2

     r_b = r_vector+(step/2)*rd_a
     rd_b = r_dot_vector+(step/2)*a
     k2 = [rd_a, a]

     # K3
     r = math.sqrt(math.pow(r_b[0], 2) + math.pow(r_b[1], 2) + math.pow(r_b[2], 2))

     c_pt_s = r_b[2]/r
     c_pt_1 = -(mu/math.pow(r, 3))
     c_pt_2 = (1+((3*earth_j2)/2)*math.pow(earth_radius/r, 2)*(1-5*math.pow(c_pt_s, 2)))*r_b
     
     a = c_pt_1 * c_pt_2

     r_c = r_vector+(step)*rd_b
     rd_c = r_dot_vector+(step)*a
     k3 = [rd_b, a]

     # K4
     r = math.sqrt(math.pow(r_c[0], 2) + math.pow(r_c[1], 2) + math.pow(r_c[2], 2))

     d_pt_s = r_c[2]/r
     d_pt_1 = -(mu/math.pow(r, 3))
     d_pt_2 = (1+((3*earth_j2)/2)*math.pow(earth_radius/r, 2)*(1-5*math.pow(d_pt_s, 2)))*r_c
     
     a = d_pt_1 * d_pt_2

     k4 = [rd_c, a]

     step_position_solution = r_vector + step*( ((k1[0])/6) + ((k2[0])/3) + ((k3[0])/3) + ((k4[0])/6) )

     step_velocity_solution = r_dot_vector + step*( ((k1[1])/6) + ((k2[1])/3) + ((k3[1])/3) + ((k4[1])/6) )

     return (step_position_solution, step_velocity_solution)



def keplarian_rk4_perturbations(r_vector, r_dot_vector, step, mu, current_date: datetime.datetime, model):
     '''Uses RK4 to predict the position and velocity of a SV given a position and velocity vectors, step, and a total computed perturbations'''

     jd = convert_date_to_jd(current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute, current_date.second)

     # rd_a_pt = ((-mu/math.pow(r0norm, 3))*r_vector)
     rd_a_pt = perturbation_acceleration(r_vector, r_dot_vector, mu, jd, model)

     r_a = r_vector + (step/2)*r_dot_vector
     rd_a = r_dot_vector + (step/2)*rd_a_pt
     jda = jd + (step/2)/86400
     k1 = [r_dot_vector, rd_a_pt]

     # rd_b_pt = ((-mu/math.pow(ranorm, 3))*r_a)
     rd_b_pt = perturbation_acceleration(r_a, rd_a, mu, jda, model)

     r_b = r_vector+(step/2)*rd_a
     rd_b = r_dot_vector+(step/2)*rd_b_pt
     jdb = jd + (step/2)/86400
     k2 = [rd_a, rd_b_pt]

     # rd_c_pt = ((-mu/math.pow(rbnorm, 3))*r_b)
     rd_c_pt = perturbation_acceleration(r_b, rd_b, mu, jdb, model)

     r_c = r_vector+(step)*rd_b
     rd_c = r_dot_vector+(step)*rd_c_pt
     jdc = jd + (step)/86400
     k3 = [rd_b, rd_c_pt]

     k4 = [rd_c, perturbation_acceleration(r_c, rd_c, mu, jdc, model)]

     step_position_solution = r_vector + step*( (k1[0]/6) + (k2[0]/3) + (k3[0]/3) + (k4[0]/6) )

     step_velocity_solution = r_dot_vector + step*( (k1[1]/6) + (k2[1]/3) + (k3[1]/3) + (k4[1]/6) )

     return (step_position_solution, step_velocity_solution)


def central_body_acceleration(mu, position_vector):
     r0norm = np.linalg.norm(position_vector)
     return (-(mu/math.pow(r0norm, 3))*position_vector)


def perturbation_acceleration(position_vector, velocity_vector, mu, jd, model):
     ''''''
     # Hardcoded from exam 2
     # Vehicle Characteristics
     sv_mass = 1000 # kg
     drag_area = 10 # m^2
     drag_coefficient = 2.0
     f10 = 100
     f10_scaled = f10/100
 
     # Earth Model
     omega_cross = 72.921151467 * math.pow(10, -6) # Radians per second
     omega_cross_vector = [0, 0, omega_cross] # Format my implementation expects
     earth_radius = 6378137.0 # meters

     sun_mu = mu * 332946.09358859973
     moon_mu = mu / 81.3005764441083

     sun_vec = determine_sun_vector_lf(jd)
     moon_vec = determine_moon_vector_lf(jd)
     
     cba = central_body_acceleration(mu, position_vector)
     
     if model == 1:
          # Just central body
          return cba
     elif model == 2:
          # Just central body and atmospheric drag
          lat, lon, alt = compute_lat_lon_alt(position_vector)
          alt = alt * .001
          cur_atmospheric_density = compute_atmospheric_density(jd, alt, position_vector, sun_vec)
          cur_atmospheric_drag = compute_atmospheric_drag(drag_coefficient, drag_area, sv_mass, cur_atmospheric_density, position_vector, velocity_vector, omega_cross_vector)
          return cba + cur_atmospheric_drag
     elif model == 3:
          # Just central body and sun and moon
          sun_perturbation = compute_third_body_acceleration(sun_mu, position_vector, sun_vec)
          moon_perturbation = compute_third_body_acceleration(moon_mu, position_vector, moon_mu)
          return cba + sun_perturbation + moon_perturbation
     elif model == 4:
          # Just central body and geopotential acceleration
          return cba + keplarian_rk4_geopotential(position_vector)
     elif model == 5:
          lat, lon, alt = compute_lat_lon_alt(position_vector)
          alt = alt * .001
          cur_atmospheric_density = compute_atmospheric_density(jd, alt, position_vector, sun_vec)
          cur_atmospheric_drag = compute_atmospheric_drag(drag_coefficient, drag_area, sv_mass, cur_atmospheric_density, position_vector, velocity_vector, omega_cross_vector)
          sun_perturbation = compute_third_body_acceleration(sun_mu, position_vector, sun_vec)
          moon_perturbation = compute_third_body_acceleration(moon_mu, position_vector, moon_mu)
          geopotential_perturbation = keplarian_rk4_geopotential(position_vector)

          return cba + cur_atmospheric_drag + sun_perturbation + moon_perturbation + geopotential_perturbation

     raise ValueError('Invalid Model requested! Must be 1-5')
     



def keplarian_rk4_geopotential(r_vector):
     '''Takes in a position vector, and applies Vallado 8-51 to compute a simplified geo-potential acceleration model'''

     # Frequently used values
     r = np.linalg.norm(r_vector)
     
     # Defined by WGS 84
     earth_radius = np.float128(6378137) # This value is in meters
     earth_j2 = np.float128(0.00108262998905194)
     mu = np.longlong(398600441800000) # This value is in meters cubed per second squared

     # Vallado 8-51
     numerator = 3*earth_j2*mu*math.pow(earth_radius, 2)
     divisor = 2 * math.pow(r, 5)
     r_squared = math.pow(r, 2)

     a_x = -1 * ((numerator*r_vector[0])/divisor)*(1-((5*math.pow(r_vector[2], 2))/r_squared))
     
     a_y = -1 * ((numerator*r_vector[1])/divisor)*(1-((5*math.pow(r_vector[2], 2))/r_squared))

     a_z = -1 * ((numerator*r_vector[2])/divisor)*(3-((5*math.pow(r_vector[2], 2))/r_squared))

     return [a_x, a_y, a_z]



def compute_f_g_f_dot_g_dot(nu, delta_nu, p, e, pos_vec, mu) -> tuple:
     '''Takes in the current True Anomaly (Nu), and the delta True Anomaly (delta_nu) in degrees.
     Also takes in p, and the eccentricity (e).
     Also takes in the position vector (Generally the r_vector but can be any given position vector).
     Also takes in a value of Mu (Generally from WGS 84).'''
     radian_nu = math.radians(nu)
     radian_delta_nu = math.radians(delta_nu)

     r = p/(1+e*math.cos(radian_delta_nu + radian_nu))

     # Compute f
     f = 1 - (r/p)*(1 - math.cos(radian_delta_nu))
     
     # Compute g
     g = ((r*np.linalg.norm(pos_vec))/(math.sqrt(mu * p))) * math.sin(radian_delta_nu)

     # Compute g_dot
     g_dot = 1 - (np.linalg.norm(pos_vec)/p) * (1 - math.cos(radian_delta_nu))

     # Compute f_dot based on f, g, and g_dot
     f_dot = (f * g_dot - 1)/g

     return (f, g, f_dot, g_dot)



def compute_arbitrary_new_position(f, g, current_position, velocity_components):
     '''Determine the new position vector from a provided f and g function'''
     return np.array(current_position) * f + np.array(velocity_components) * g



def compute_arbitrary_new_velocity(f_dot, g_dot, current_position, velocity_components):
     '''Determine the new velocity vector from a provided f_dot and g_dot function'''
     return np.array(current_position) * f_dot + np.array(velocity_components) * g_dot



def convert_date_to_jd(year: int, month: int, day: int, hour: int, min: int, sec: int) -> float:
     '''Converts a provided date to Julian Date (JD). Returns a float with the JD in days'''
     timeut = hour + ( min / 60 ) + ( sec / 3600 )

     jd = (367 * year) - np.floor( 7 * ( year + np.floor( ( month + 9 ) / 12 ) ) / 4 ) -  np.floor( 3 * ( np.floor( ( year + ( month - 9 ) / 7 ) / 100 ) + 1 ) / 4 ) + np.floor( ( 275 * month ) / 9 ) + day + 1721028.5

     return jd + ( timeut / 24 )

def convert_jd_to_mjd(jd: float) -> float:
     '''Converts a provided julian date to Modified Julian Date (MJD). Returns a float with the MJD in days'''
     return jd - 2400000.5

def convert_mjd_to_besselian_year(mjd: float):
     '''Converts a Modified Julian Date (MJD) to Besselian Year'''
     return 2000.0 + (mjd - 51544.03)/365.242199

def convert_besselian_to_ut2_ut1(besselian_years: float):
     return 0.022 * math.sin(2*math.pi*besselian_years) - 0.012 * math.cos(2*math.pi*besselian_years) - 0.006 * math.sin(4*math.pi*besselian_years) + 0.007 * math.cos(4*math.pi*besselian_years)

def convert_ut1_utc(mjd, ut2_ut1):
     '''Takes in the Modified Julian Date and the UT2-UT1 date and returns the UT1-UTC date'''

     # From https://datacenter.iers.org/data/latestVersion/bulletinA.txt
     # Dated: 26 February 2026                                    Vol. XXXIX No. 009
     return 0.0640 + 0.00003 * (mjd - 61091) - (ut2_ut1)

def get_tai_utc() -> float:
     '''Taken from: https://datacenter.iers.org/data/latestVersion/bulletinA.txt  which at the time of writing (26 February 2026) defines TAI-UTC as 37.000000 seconds'''
     return 37.000000

def get_tt_utc() -> float:
     '''Returns the terrestrial time '''
     return get_tai_utc() + 32.184

def get_j2000_jd_epoch() -> float:
     '''Returns the Julian Date epoch for the j2000 specified epoch: 1 Jan 2000 12:00:00'''
     return convert_date_to_jd(2000, 1, 1, 12, 0, 0)

def determine_seconds_since_j2000_epoch(jd) -> float:
     '''Takes in a Julian Date and converts it to seconds since the J2000 epoch'''
     return (jd - get_j2000_jd_epoch()) * 86400

def determine_tai_s(seconds):
     '''Takes in a time in seconds since the j2000 epoch and converts to the TAI seconds'''
     return seconds + get_tai_utc()

def determine_tai(seconds):
     '''Takes in the TAI seconds and returns the TAI'''
     return get_j2000_jd_epoch() + seconds/86400

def determine_tbd(tai):
     '''Takes in the TAI and computes the Terrestrial Barycentric Dynamic (TBD) time'''
     return tai + 32.184/86400

def determine_tt(jd):
     '''Not to be confused with get_tt_utc(). This will calculate the Terrestrial Time for a given Julian Date'''
     return jd + get_tt_utc()/86400

def determine_sun_vector_lf(jd):
     '''Uses a low fidelity model from the Astronomical Almanac for 2017 for the apparent right ascension and declination of the sun'''
     au = 149597870700 # Definition of a single AU in meters
     n = jd - 2451545

     # L
     mean_longitude_of_sun = math.radians(280.460+0.9856474*n)
     mean_longitude_of_sun = mean_longitude_of_sun % (2*math.pi)
     if mean_longitude_of_sun < 0:
          mean_longitude_of_sun = mean_longitude_of_sun + 2*math.pi

     # g
     mean_anomaly_of_sun = math.radians(357.528+0.9856003*n)
     mean_anomaly_of_sun = mean_anomaly_of_sun % (2*math.pi)
     if mean_anomaly_of_sun < 0:
          mean_anomaly_of_sun = mean_anomaly_of_sun + 2*math.pi

     # Lambda
     ecliptic_longitude = mean_longitude_of_sun + math.radians(1.915)*math.sin(mean_anomaly_of_sun) + math.radians(0.020)*math.sin(2*mean_anomaly_of_sun)
     # Beta
     ecliptic_latitude = 0

     #epsilon
     obliquity_of_ecliptic = math.radians(23.439-0.0000004*n)

     # Alpha
     f = 180/math.pi
     t = math.pow(math.tan(obliquity_of_ecliptic/2), 2)
     right_ascension = ecliptic_longitude - f*t*math.sin(2*ecliptic_longitude)+(f/2)*math.pow(t, 2)*math.sin(4*ecliptic_longitude)
     # right_ascension = math.atan(math.cos(obliquity_of_ecliptic)*math.tan(ecliptic_longitude))

     distance_from_sun = 1.00014 - 0.01671*math.cos(mean_anomaly_of_sun) - 0.00014*math.cos(2*mean_anomaly_of_sun)

     # Sun Coordinates
     x = (distance_from_sun * math.cos(ecliptic_longitude))*au
     y = (distance_from_sun * math.cos(obliquity_of_ecliptic)*math.sin(ecliptic_longitude))*au
     z = (distance_from_sun * math.sin(obliquity_of_ecliptic)*math.sin(ecliptic_longitude))*au

     return [x, y, z]

def determine_moon_vector_lf(jd):
     '''Uses a low fidelity model from the Astronomical Almanac for 2017 for the apparent right ascension and declination of the sun'''
     er = 6378137 # meters
     t = (jd-2451545)/36525


     lambdalambda = math.radians(218.32+481267.881*t) + math.radians(6.29) * math.sin(math.radians(135.0 + 477198.87*t)) - math.radians(1.27) * math.sin(math.radians(259.3 - 413335.36*t))+ math.radians(0.66) * math.sin(math.radians(235.7 + 890534.22*t)) + math.radians(0.21) * math.sin(math.radians(269.9 + 954397.74*t)) - math.radians(0.19) * math.sin(math.radians(357.5 + 35999.05*t)) -  math.radians(0.11) * math.sin(math.radians(186.5 + 966404.03*t))

     beta = math.radians(5.13) * math.sin(math.radians(93.3 + 483202.02*t)) + math.radians(0.28)*math.sin(math.radians(228.2+960400.89*t))-math.radians(0.28) * math.sin(math.radians(318.3 + 6003.15*t))-   math.radians(0.17)*math.sin(math.radians(217.6 - 407332.21*t))

     new_pi = math.radians(0.9508)+math.radians(0.0518)*math.cos(math.radians(135.0+477198.87*t))+math.radians(0.0095)*math.cos(math.radians(259.3-413335.36*t))+math.radians(0.0078)*math.cos(math.radians(235.7+890534.22*t))+math.radians(0.0028)*math.cos(math.radians(269.9+954397.74*t))


     sd = 0.2724*new_pi
     r = 1/math.sin(new_pi)


     l = math.cos(beta)*math.cos(lambdalambda)
     
     m = 0.9175*math.cos(beta)*math.sin(lambdalambda)-0.3978*math.sin(beta)
     
     n = 0.3978*math.cos(beta)*math.sin(lambdalambda)+0.9175*math.sin(beta)

     
     alpha = math.atan(m/l)

     delta = math.asin(n)

     x = r*l*er
     y = r*m*er
     z = r*n*er

     return [x, y, z]

def compute_third_body_acceleration(third_body_mu, sv_eci_position: list, third_body_eci_position: list):
     '''Determines the acceleration force from a third body on a given SV. Units must be given in an ECI reference frame'''
     # Relative position vector is given by the equation: relative_position = third_body_ECI_position - SV_ECI_position
     # Acceleration of the third body is then calculated with the following:
     # a_third_body = mu_third_body * ((relative_position/norm(relative_position)^3) - (third_body_ECI_position/norm(third_body_ECI_position)^3))
     relative_pos = third_body_eci_position - sv_eci_position

     pt1 = np.divide(relative_pos, math.pow(np.linalg.norm(relative_pos), 3))

     pt2 = np.divide(third_body_eci_position, math.pow(np.linalg.norm(third_body_eci_position), 3))

     return third_body_mu * (pt1 - pt2)

def compute_f10_scaled(days):
     '''Computes the value of F10_scaled, takes in days since the epoch: 00:00:00 31 Dec 1957. If you need F10, multiply the returned value by 100'''
     return 1.5 + 0.8 * math.cos((2 * math.pi * days)/4020)

def compute_lat_lon_alt(position_vector) -> tuple:
     '''Computes the longitude, geocentric latitude, and altitude given a position vector'''
     
     # WGS84 defined values from slide 40
     earth_radius = 6378137 # Meters, aka a_cross
     flattening = 1/298.257223563
     polar_semi_minor_axis = 6356752.314 # Meters, aka b_cross
     e_cross = 0.0818191908
     e_squared = math.pow(e_cross, 2)

     cur_lat = math.asin(position_vector[2]/np.linalg.norm(position_vector))
     tolerance = 0.000000000000001
     update = 10

     while update > tolerance:
          tanlat = (position_vector[2]+(earth_radius*e_squared)/math.sqrt(1+math.pow(flattening/earth_radius, 2))*(math.pow(math.tan(cur_lat), 2)))/math.sqrt(math.pow(position_vector[0], 2)+math.pow(position_vector[1], 2))
          lat = math.atan(tanlat)
          update = lat - cur_lat
          cur_lat = lat

     xy = (math.pow(position_vector[0], 2)+math.pow(position_vector[1], 2))/math.pow(earth_radius, 2)
     z = math.pow(position_vector[2], 2)/math.pow(polar_semi_minor_axis, 2)

     altitude = np.linalg.norm(position_vector) * (1 - 1/math.sqrt(xy + z))

     longitude = math.atan2(position_vector[1], position_vector[0])
     if longitude < 0:
          longitude = longitude + 2*math.pi

     return (cur_lat, longitude, altitude)


def compute_atmospheric_density(jd, altitude, position_vector, sun_vector, print_angle_sv2bulge=False):
     '''Implements Jacchia 1960 (AKA J60) density model. Returns atmospheric pressure p. Takes in altitude in km, position_vector, and the sun_vector
     For this homework we are given that F10 = 100 so F10 Scaled would be 1.'''
     
     bulge_lag_angle = 0.55 # radians. Lag of bulge from sun
     cos_b = math.cos(bulge_lag_angle)
     sin_b = math.sin(bulge_lag_angle)

     naut_alt = altitude / 1.852 # Convert km to nautical miles

     # Find days since epoch: 00:00:00 31 Dec 1957
     epoch = datetime.datetime(1957, 12, 31, 0, 0, 0)

     # Commented out for now
     # T = date - epoch
     # f10_scaled = compute_f10_scaled(T.days)
     f10 = 100 # Given on exam

     f10_actual_scaled = f10/100

     sun_vector_unit = sun_vector/np.linalg.norm(sun_vector)

     # Slide 34
     diurnal_bulge_unit_vector = [ sun_vector_unit[0]*cos_b - sun_vector_unit[1]*sin_b, sun_vector_unit[1]*cos_b + sun_vector_unit[0]*sin_b, sun_vector_unit[2] ]
     
     cos_psi = np.dot(position_vector, diurnal_bulge_unit_vector)/(np.linalg.norm(position_vector)*np.linalg.norm(diurnal_bulge_unit_vector))

     # Print angle out as a heading check
     if print_angle_sv2bulge:
          print(f'Angle from SV to bulge: {math.degrees(math.acos(cos_psi))}')

     if naut_alt < 108:
          raise NotImplementedError('Nautical mile altitude below 180, which is not implemented')
     elif naut_alt < 378:
          # Run drag calculation formula between 108 and 378 nautical miles
          # Slide 36
          a = 0.00368
          b = 15.738
          k = 515.37886

          p_o = np.exp((6.363*np.exp(-0.0048 * naut_alt) - a*naut_alt - b) * math.log(10, math.e))

          p = p_o * (0.85 * f10_actual_scaled) * (1 + 0.02375 * (np.exp(0.0102*naut_alt) - 1.9)*(math.pow(1 + cos_psi, 3))) * k

     elif naut_alt < 1000:
          # Run drag calculation formula between 378 and 1000 nautical miles
          k = 515.37886
          p = ((0.00504 * f10_actual_scaled)/math.pow(naut_alt, 8)) * (0.125*math.pow(1 + cos_psi, 3)*(math.pow(naut_alt, 3) - 6*math.pow(10, 6)) + 6*math.pow(10, 6)) * k

     else:
          # Default case since above 1000 nautical miles, the drag is effectively 0 kg/m^3
          p = 0
     
     return p

def compute_atmospheric_drag(drag_coefficient, drag_area, sv_mass, atmospheric_density, position_vector, velocity_vector, omega_cross_vector):
     '''Computes the atmospheric drag on a satellite. Takes in a drag_coefficient, drag_area in meters squared, the mass of your space vehicle, atmospheric_density (Found from Jacchia), a position_vector, velocity_vector, and a vector for the rotation of the earth omega_cross_vector.
     The reference frame for the position and velocity vector should be ECI.'''

     # Slide 28
     v_r_arrow = velocity_vector - np.cross(omega_cross_vector, position_vector)
     
     velocity_unit_vector = v_r_arrow/np.linalg.norm(v_r_arrow)

     velocity_norm = np.linalg.norm(v_r_arrow)

     pt1 = -((drag_coefficient * drag_area)/(2*sv_mass))

     pt2 = atmospheric_density * math.pow(velocity_norm, 2) * velocity_unit_vector

     return pt1 * pt2

def compute_solar_radiation(exposed_area_to_sun, radiation_pressure_coefficient, sv_mass, sun_shadow_factor, solar_intensity, vector_from_sv_to_sun):
     '''Implements a "Ball" method to predict the solar radiation pressure on the satellite.'''
     # Slide 24
     cram = (radiation_pressure_coefficient * exposed_area_to_sun) / sv_mass

     pt2 = sun_shadow_factor * solar_intensity * np.divide(np.dot(-1, vector_from_sv_to_sun), np.linalg.norm(vector_from_sv_to_sun))

     return cram * pt2


def compute_total_acceleration_sv(accelerations: list):
     '''Adds up all the estimations for different forces together and applies them to the space vehicle. Takes in a list of lists'''
     total_accelerations = [0, 0, 0]
     for a in accelerations:
          total_accelerations += a
     
     return total_accelerations

def convert_eci_ecef(r_vector: list, seconds_since_epoch):
     '''Aproximates the ECI->ECEF transformation. Takes in a positon vector and the number of seconds since the epoch.'''

     rot_angle = (7.292116*math.pow(10, -5))*seconds_since_epoch # Rotation of Earth multiplied by the seconds since the epoch

     rotation_matrix = [[math.cos(rot_angle), math.sin(rot_angle), 0],
                        [-math.sin(rot_angle), math.cos(rot_angle), 0],
                        [0, 0, 1]]

     ecef_coords = np.dot(rotation_matrix, r_vector)

     return ecef_coords

def compute_lat_lon_ecef(r_vector) -> tuple:
     '''Takes in a position vector in ECEF reference frame and computes lat and lon'''
     u = r_vector/np.linalg.norm(r_vector)

     lon = math.atan2(u[1], u[0])

     if lon < 0:
          lon = lon + 2*math.pi
     
     lat = math.asin(u[2])

     return (lat, lon)

def graph_sma_difference(title, x_label, y_label, x_data, y_data, filename):
     '''Built for Exam 2'''
     plt.title(title)
     plt.xlabel(x_label)
     plt.ylabel(y_label)
     plt.plot(x_data, y_data)
     plt.savefig(filename)
     plt.clf()

def compute_sma(r_vector, r_dot_vector):
     mu = 398600441800000
     energy = (math.pow(np.linalg.norm(r_dot_vector), 2)/2) - (mu/np.linalg.norm(r_vector))
     
     sma = -mu/(2*energy)

     if sma < 0:
          print(f'Negative SMA!: {sma}')
          exit(0)

     return sma

def compute_julian_centuries(Tau):
     '''Computes the Julian centuries since J2000. Takes in Tau, the number of UT1 seconds between time of interest and J2000'''
     return (np.floor((Tau/86400)+.5)-.5)/36525

def compute_gha(t, T):
     '''Computes the Greenwich Hour Angle. Takes in the universal time (UT) t, and the number of UT julain centuries since J2000 T'''
     rot_angle = (7.292116*math.pow(10, -5))*t

     pt1 = (6*60*60+41*60+50.54841) + 8640184.812866*T + 0.093104*math.pow(T, 2) - 6.2*math.pow(10, -6)*math.pow(T, 3)
     pt2 = (2*math.pi)/86400

     return rot_angle + pt1 * pt2

def compute_gha_jd(jd):
     '''Computes the Greenwich Hour Angle. Takes in the universal time (UT) t, and the number of UT julain centuries since J2000 T'''
     w_earth = 7.292116*math.pow(10, -5)

     tau = (jd-get_j2000_jd_epoch())*86400

     Tu = (np.floor(tau/86400+0.5)-0.5)/36525.0

     t = tau - Tu*36525*86400

     return t*w_earth + (24110.54841 + 8640184.812866*Tu + 0.093104*math.pow(Tu, 2)-math.pow(6.2, -6)*math.pow(Tu, 3))*(2*math.pi/86400.)  #rot_angle + pt1 * pt2

def convert_eci_ecef_gha(r_vector: list, jd):
     '''Aproximates the ECI->ECEF transformation for the Greenwich Hour Angle. Takes in a positon vector and the Julian Date'''
     tau = (jd-get_j2000_jd_epoch())*86400

     julian_centuries = compute_julian_centuries(tau)
     
     t = tau - julian_centuries*36525*86400

     gha_angle = compute_gha(t, julian_centuries)

     rotation_matrix = [[math.cos(gha_angle), math.sin(gha_angle), 0],
                        [-math.sin(gha_angle), math.cos(gha_angle), 0],
                        [0, 0, 1]]

     ecef_coords = np.dot(rotation_matrix, r_vector)

     return ecef_coords

def convert_ecef_topocentric(lat, lon, ecef_coordinates):
     '''Takes a geodetic latitude, longitude, and altitude and computes '''

     Lamba = lon
     Phi = lat
     
     transformation_matrix = [[-math.sin(Lamba), math.cos(Lamba), 0],
                              [-math.sin(Phi)*math.cos(Lamba), -math.sin(Phi)*math.sin(Lamba), math.cos(Phi)],
                              [math.cos(Phi)*math.cos(Lamba), math.cos(Phi)*math.sin(Lamba), math.sin(Phi)]]
     
     
     return np.dot(transformation_matrix, ecef_coordinates)

def compute_azimuth_elevation(rel_pos) -> tuple:
     '''Returns (Azimuth, Elevation) after being provided, relative_position_vector'''

     az = math.atan2(rel_pos[0], rel_pos[1])

     el = math.atan(rel_pos[2]/math.sqrt(math.pow(rel_pos[0], 2)+math.pow(rel_pos[1], 2)))

     return (az, el)

def convert_ecef_tod(r_vector: list, jd):
     '''Aproximates the ECEF->TOD transformation for the Greenwich Hour Angle. Takes in a positon vector and the Julian Date'''
     tau = (jd-get_j2000_jd_epoch())*86400

     julian_centuries = compute_julian_centuries(tau)
     
     t = tau - julian_centuries*36525*86400

     # gha_angle = compute_gha(t, julian_centuries)
     gha_angle = compute_gha_jd(jd)

     rotation_matrix = [[math.cos(gha_angle), math.sin(gha_angle), 0],
                        [-math.sin(gha_angle), math.cos(gha_angle), 0],
                        [0, 0, 1]]

     ecef_coords = np.dot(np.transpose(rotation_matrix), r_vector)

     return ecef_coords

def instantaneous_range(position_vector, tod_position_vector, velocity_vector, mu):
     '''Low fidelity model to measure the distance from a ground station to a space vehicle'''

     c = 299792458 # Meters /per second

     new_vec = position_vector - tod_position_vector
     rel_mag = np.linalg.norm(new_vec)

     print(f'Vector Difference: {rel_mag}')

     distance = rel_mag/c

     print(f'Instant Light time: {distance}')

     new_pos, new_vel = keplarian_rk4(position_vector, velocity_vector, -distance, mu)

     inst_diff = new_pos - tod_position_vector

     instant_distance = np.linalg.norm(inst_diff)

     return instant_distance

def light_time_range(ground_site_tod, sv_pos_vector, sv_vel_vector, transponder_delay, mu):
     ''''''
     c = 299792458 # Meters /per second
     earth_rotation_speed = 7.292116*math.pow(10, -5)

     tolerance = 0.0000000001
     difference = 100000

     dtprop = 0
     last_dt = 0

     new_vec = sv_pos_vector - sv_vel_vector
     rel_mag = np.linalg.norm(new_vec)
     distance = rel_mag/c
     new_pos, new_vel = keplarian_rk4(sv_pos_vector, sv_vel_vector, -distance, mu)

     print(f'SV Vector at t-dt_inst: {new_pos}')

     rx = sv_pos_vector
     vx = sv_vel_vector

     step = 1

     while np.abs(difference) > tolerance:
          dr = rx - ground_site_tod
          rho = np.linalg.norm(dr)
          dt = rho/c

          difference = dt - last_dt

          last_dt = dt
          rx, vx = keplarian_rk4(sv_pos_vector, sv_vel_vector, -dt, mu)

          step += 1

     dt_receive_leg = dt
     dtprop_rec = dt + transponder_delay

     rr, vr = keplarian_rk4(rx, vx, -transponder_delay, mu)

     rsta_t = ground_site_tod
     difference = 10000
     last_dt = 0

     step = 1

     while np.abs(difference) > tolerance:
          dr = rr - rsta_t
          rho = np.linalg.norm(dr)
          dt = rho/c

          difference = dt - last_dt

          last_dt = dt
          dtprop = dtprop_rec + dt
          rotation = -earth_rotation_speed * dtprop
          R = [[math.cos(rotation), math.sin(rotation), 0],
               [-math.sin(rotation), math.cos(rotation), 0],
               [0, 0, 1]]
          rsta_t = np.dot(R, ground_site_tod)

          step += 1

     dt_transmit_leg = dt

     dtprop = (dt_receive_leg + transponder_delay + dt_transmit_leg)/2

     return dtprop

def compute_hcw_matrix(pos_vec: list, vel_vec: list, mean_anomaly, time):
     '''Given a postion vector, a velocity vector, a mean anomaly, and time, computes the HCW vector'''
     component_matrix = [pos_vec[0], pos_vec[1], pos_vec[2], vel_vec[0], vel_vec[1], vel_vec[2]]

     n = mean_anomaly

     transformation_matrix = [[4-3*math.cos(n * time), 0, 0, (1/n)*math.sin(n*time), (2/n)*(1-math.cos(n*time)), 0],
                              [6*(math.sin(n*time)-n*time), 1, 0, -(2/n)*(1-math.cos(n*time)), (1/n)*(4*math.sin(n*time) - 3*n*time), 0],
                              [0, 0, math.cos(n*time), 0, 0, (1/n)*(math.sin(n*time))],
                              [3*n*math.sin(n*time), 0, 0, math.cos(n*time), 2*math.sin(n*time), 0],
                              [-6*n*(1-math.cos(n*time)), 0, 0, -2*math.sin(n*time), 4*math.cos(n*time)-3, 0],
                              [0, 0, -n*math.sin(n*time), 0, 0, math.cos(n*time)]]
     
     return transformation_matrix

def compute_hcw(pos_vec: list, vel_vec: list, mean_anomaly, time):
     '''Given a postion vector, a velocity vector, a mean anomaly, and time, computes the HCW vector'''
     component_matrix = [pos_vec[0], pos_vec[1], pos_vec[2], vel_vec[0], vel_vec[1], vel_vec[2]]

     n = mean_anomaly

     transformation_matrix = compute_hcw_matrix(pos_vec, vel_vec, mean_anomaly, time)

     final_component_matrix = np.dot(transformation_matrix, component_matrix)

     new_pos = [final_component_matrix[0], final_component_matrix[1], final_component_matrix[2]]
     new_vel = [final_component_matrix[3], final_component_matrix[4], final_component_matrix[5]]

     return (new_pos, new_vel)

def get_uvw_transformation_matrix(pos_vec, vel_vec):
     '''Returns the UVW transformation matrix for a given position and velocity'''
     u = pos_vec/np.linalg.norm(pos_vec)
     w = (np.cross(pos_vec, vel_vec))/np.linalg.norm(np.linalg.cross(pos_vec, vel_vec))
     v = np.cross(w, u)

     transform = [u, v, w]

     return transform

def convert_coordinates_to_uvw(pos_vec, vel_vec) -> tuple:
     '''Convert given ECI coordinates (pos_vec) and its cooresponding velocities (vel_vec) to UVW coordinates.'''
     transform = get_uvw_transformation_matrix(pos_vec, vel_vec)

     uvw_coords = np.dot(transform, pos_vec)

     return uvw_coords

def convert_vel_to_uvw(pos_vec, vel_vec) -> tuple:
     '''Convert given velocity coordinates (vel_vec) and its cooresponding ECI position (pos_vec) to UVW velocity.'''
     transform = get_uvw_transformation_matrix(pos_vec, vel_vec)

     uvw_coords = np.dot(transform, vel_vec)

     return uvw_coords

def compute_rss_diff(vec1: list, vec2: list):
     '''Takes in two vectors, three elements each (X, Y, Z) and computes the RSS differences between the two. These could be position vectors, or velocity vectors, the equation is the same for both.'''
     
     x = math.pow(vec1[0] - vec2[0], 2)
     y = math.pow(vec1[1] - vec2[1], 2)
     z = math.pow(vec1[2] - vec2[2], 2)
     
     delta_diff = math.sqrt(x + y + z)

     return delta_diff

def compute_ke_diff(ke1: KeplerianElements, ke2: KeplerianElements):
     '''Takes in two KeplarianElements classes and computes the Keplarian Element differences between the two'''
     
     return_dict = {}

     return_dict['eci_pos'] = ke1.r_vector - ke2.r_vector
     return_dict['eci_vel'] = ke1.r_dot_vector - ke2.r_dot_vector
     return_dict['sma'] = ke1.semi_major_axis - ke2.semi_major_axis
     return_dict['eccentricity'] = ke1.eccentricity - ke2.eccentricity
     return_dict['inclination'] = math.degrees(ke1.inclination - ke2.inclination) # Degrees
     return_dict['raan'] = math.degrees(ke1.raan - ke2.raan) # Degrees
     return_dict['aop'] = math.degrees(ke1.determine_argument_of_periapsis() - ke2.determine_argument_of_periapsis()) # Degrees
     return_dict['nu'] = math.degrees(ke1.nu - ke2.nu) # Degrees
     return_dict['mean_anomaly'] = math.degrees(ke1.mean_anomaly - ke2.mean_anomaly) # Degrees
     return_dict['arglat'] = math.degrees(ke1.compute_argument_of_latitude() - ke2.compute_argument_of_latitude()) # Degrees
     
     return return_dict

def linear_interpolate(x, x1, x2, y1, y2):
     '''Performs the linear interpolate equation'''
     return y1 + ((x-x1)/(x2-x1))*y2

def time_linear_interpolation(time1: datetime.datetime, time2: datetime.datetime, z1: float, z2: float):
     '''Implements linear interpolation to estimate when a SV crosses the equator given a time an z coordinate before and after'''

     # Assume a common epoch of time1
     # Assume the equator z_ecef coordinate is 0
     x = 0

     # Makes x1 and x2 z_ecef coordinate 1 and z_ecef coordinate 2 respectively
     x1 = z1
     x2 = z2

     # Seconds since epoch which will always be 0 since our epoch is time1
     y1 = 0

     # Total seconds since the epoch
     y2 = (time2 - time1).total_seconds()

     # Linear Interpolation to estimate when the SV crosses the equator
     seconds_at_equator = linear_interpolate(x, x1, x2, y1, y2)

     time_of_pass = time1 + datetime.timedelta(seconds=seconds_at_equator)

     return time_of_pass

def position_linear_interpolation(time1: datetime.datetime, time2: datetime.datetime, time3: datetime.datetime, pos1: float, pos2: float):
     '''Implements linear interpolation to estimate a coordinate position at a given time3 given a before and after coordinate (pos1 and pos2 respectively). time1 is the time at pos1 and time2 is the time at pos2'''

     # Seconds since our epoch of time1
     x = (time1 - time3).total_seconds()

     # x1 and x2 are in seconds to maintain continuity
     x1 = 0 # Always 0 since it would be time1 - time1
     x2 = (time2 - time1).total_seconds()

     # Our value for position 1 (At epoch)
     y1 = pos1

     # Our value for position 2 (After epoch and after time3)
     y2 = pos2 

     # Linear Interpolation to estimate when the SV crosses the equator
     estimated_position = linear_interpolate(x, x1, x2, y1, y2)

     return estimated_position

def calculate_glan(ecef_x, ecef_y, n):
     '''Takes in an ECEF X and Y coordinate, and the norm n of the ECEF vector and computes the GLAN from that'''
     glan = math.acos(ecef_x/n)

     if ecef_y >= 0:
          return glan     
     elif ecef_y < 0:
          glan = (2*math.pi)-glan
     
     return glan