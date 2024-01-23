import matplotlib.pyplot as plt
import numpy as np
from stream import Stream
from scipy.stats import norm
from copy import deepcopy


class ObservedSignal:
    """
    Create an observed signal
    """

    def __init__(self, time, hertz, background_mean=10,
                 background_type="poisson"):
        self.time = time
        self.hertz = hertz
        self.domain = np.arange(0, time, hertz)

        self.background_mean = background_mean
        self.background_type = background_type
        self.background = self._make_background()

        self.observed = deepcopy(self.background).astype(float)

    def _make_background(self):
        if self.background_type=="poisson":
            background = np.random.poisson(
                self.background_mean, int(time / hertz)
            )
            return background

    def add_source(self, location, intensity, std):
        source = norm(int(time * location), std).pdf(self.domain)
        source /= source.max()
        source *= self.background_mean * intensity
        self.observed += source
    
    def view(self):
        plt.plot(self.domain, self.observed)
        plt.show()


hertz = 1
time = 1000

observedSignal = ObservedSignal(time, hertz)

observedSignal.add_source(.2, 1, 10)
observedSignal.add_source(.5, 1, 10)
observedSignal.add_source(.9, 1, 10)

observedSignal.view()

stream = Stream(p_val_thresh=.01, alarm_when=15)
stream.run(observedSignal.observed)

plt.plot(stream.all_p_values_time, stream.all_p_values)
try:
    plt.scatter(
        stream.alarm_times[:, 0], 
        stream.alarm_times[:, 1], 
        c="red", s=4
    )
except Exception as e:
    print(e)
plt.plot(observedSignal.observed)
plt.show()
