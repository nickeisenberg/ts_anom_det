import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
from stream import Stream

def make_observed(time, hertz, background_mean, source_intensity, source_std):
    domain = np.arange(0, time, hertz)
    background = np.random.poisson(background_mean, int(time / hertz))
    source_std = hertz * source_std
    source = norm(time // 2, source_std).pdf(domain)
    source /= source.max()
    source *= background_mean * source_intensity
    observed = source + background
    return observed

hertz = 1
time = 1000
background_mean = 10
source_intensity = .5
source_std = 10

observed = make_observed(
    time=time, 
    hertz=hertz, 
    background_mean=background_mean, 
    source_intensity=source_intensity, 
    source_std=source_std
)

plt.plot(observed)
plt.show()

stream = Stream(p_val_thresh=.05, alarm_when=15)
stream.run(observed)

stream.alarms

plt.plot(stream.all_p_values_time, stream.all_p_values)
try:
    plt.scatter(
        stream.alarm_times[:, 0], 
        stream.alarm_times[:, 1], 
        c="red", s=4
    )
except Exception as e:
    print(e)
plt.plot(observed)
plt.show()
