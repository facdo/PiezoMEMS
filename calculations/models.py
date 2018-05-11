""" Classes related to the models used for making any kind of calculation.
    
    The module implements the requred objects for making time and frequency domain calculations,
    for accessing the potential of vibration sensors to monitor vehicle traffic.

"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class Master(object):
    """ Master parent class with generic routines.
        
        Implements a generic set_parameter method on init.
    """
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.set_parameter(k, v)

    def set_parameter(self, parameter, value):
        setattr(self, parameter, value)

class Vehicle(Master):
    """ Informations and methods related to vehicles.
        
        All parameters in SI units.
    """
    tire_len = 0.1
    tire_width = 0.1
    tire_space = 3
    N_wheels = 4
    N_axis = 2
    speed = 10
    weight = 900

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tire_contact_time = self.tire_len/self.speed
        self.max_tire_pressure = (self.weight * 9.81)/(self.tire_len*self.tire_width*self.N_wheels)


    def pressure_vector(self, N_points, time_range = (0, 1), peek_value_relative_time = 0.5, multiple_axis=True):
        """ Generates a temporal pressure normal curve, simulating a vehicle passing at a point.
            
            Args:
                N_points (int): number of samples in the arrays.
                time_range (tuple): the start and stop time for the samples.
                peek_value_relative_time (float): the point where the max_tire_pressure occurrs, relative
                    to the time_range. Expected to be a value between 0 and 1.
                multiple_axis (bool): considers the effect of multiple tires passing through. Generates 
                    multiple pressure pulses delayed by the tire_space and speed.
             
            Returns:
                list: list with two arrays, one corresponding to the time stample, and the other to the 
                    pressure in one point.
        """
        t_array = np.linspace(*time_range, N_points)
        std, m = self.tire_contact_time/6, (time_range[1]-time_range[0])*peek_value_relative_time
        p_array = norm(scale=std, loc=m).pdf(t_array)
            
        if multiple_axis:
            delay = 0
            for axis in np.arange(self.N_axis-1):
                delay += self.tire_space/self.speed
                p_array += norm(scale=std, loc=m+delay).pdf(t_array)
        
        # normalizing and applying the max_pressure to the peak
        p_array /= max(p_array)
        p_array *= self.max_tire_pressure
        return [t_array, p_array]


class Sensor(object):
    """
        Sensor specifications. Can be the master object to describe a 
        cantilever device.
        All parameters in SI units.
    """
    resonant_freq = 150
    # frequency variation corresponding to half the maximum intensity value, Q = f/B
    bandwidth = 5 
    # given in V/ms-2
    sensitivity = 0.5


class ViewUtils():
    """ Class for generating visualizations, in time and frequency domain.

        Uses plotly for generating interactive views.
    """
    def __init__(self, data):
        """ Class instance has the data to be ploted/manipulated

            Args:
                data (dict): data strucutred as key:data_array.
        """
        self.data = data
    
    def convert_to_frequency(self, time_key, data_key):
        """ Converts the data initialized from the time domain to the frequency domain

            Args:
                time_key (str): the key in the self.data dict corresponding to the time array
                data_key (str): the key in the self.data dict corresponding to the data array

            Returns:
                list: list with two arrays, one corresponding to the FFT frequency axis, and the 
                    other corresponding to the Y axis.
        """
        N = len(self.data[time_key])
        T = self.data[time_key][1] - self.data[time_key][0]
        return [np.fft.fftfreq(N, T)[:N//2],
                np.abs(np.sqrt(T)*np.fft.fft(self.data[data_key]))[:N//2]]

    def plot_data(self, **kwargs):
        """ Generates a plot of the data.

            Several options avalible on kwargs parameters.
        """
        pass


a = Vehicle(**{'tire_len':0.4, 'speed':20})
a.set_parameter('model', 'truck')
print(a.speed, a.tire_len, a.model)

t, p = a.pressure_vector(1000)
plot_data = ViewUtils({'t':t,'p':p}).convert_to_frequency('t','p')
# plt.plot(t,p)
plt.plot(plot_data[0], plot_data[1])
plt.show()
        