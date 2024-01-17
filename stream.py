import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


class Stream:
    """
    Parameters
    ----------
    p_val_thresh: float
        p_val for hypothesis test
    reference_interval: int 
        The interval to compare the present with
    alarm_interval: int
        Time interval to check alarms
    alarm_when: int
        The number of hypthesis rejections in an interval before triggering
        an alarm

    Notes
    -----
    The reference interval is right before the alarm interval. Ie, if the 
    reference interval is 120 and the alarm interval is 20 then what happens
    is that time 0 to 120 is compared to time 120 to 140. In this case 140 is
    the present and 0 is when the experiment started.
    """

    def __init__(self, 
                 p_val_thresh=.05,
                 reference_interval=120,
                 alarm_interval=20, 
                 alarm_when=5):

        self.p_val_thresh = p_val_thresh
        self.reference_interval = reference_interval 
        self.alarm_interval = alarm_interval
        self.alarm_when = alarm_when

        self.stream: np.ndarray = np.array([])
        self.p_values: np.ndarray = np.array([])
        self.t_values: np.ndarray = np.array([])
        self.alarms = 0
        self.alarm_times = []

        # some debugging 
        self.all_p_values: np.ndarray = np.array([])
        self.all_p_values_time: np.ndarray = np.array([])
        self.time = 0

    def run(self, time_series):
        for x in time_series:
            self._update(x)

        self.alarm_times = np.array(self.alarm_times)

    def _update(self, x: float):
        if self.stream.size < self.reference_interval + self.alarm_interval:
            self.stream = np.hstack((self.stream, x))
            self.time += 1
            return None

        self.stream = np.hstack(
            (self.stream[-self.reference_interval - self.alarm_interval + 1:], x)
        )

        t_test = sm.stats.ttest_ind(
            self.stream[-self.reference_interval - self.alarm_interval: -self.alarm_interval], 
            self.stream[-self.alarm_interval:]
        )

        self.p_values = np.hstack(
            (self.p_values[-self.alarm_interval+1:], t_test[1])
        )
        self.t_values = np.hstack(
            (self.t_values[-self.alarm_interval+1:], t_test[0])
        )

        self._trigger_alarm()
        
        # debugging stuff
        self.all_p_values = np.hstack(
            (self.all_p_values, t_test[1])
        )
        self.all_p_values_time = np.hstack(
            (self.all_p_values_time, self.time)
        )

        self.time += 1

        return None

    def _trigger_alarm(self):
        if (self.p_values < self.p_val_thresh).sum() > self.alarm_when:
            self.alarms += 1
            for ind, t in enumerate(np.arange(self.time - self.alarm_interval, self.time)):
                self.alarm_times.append([t, self.p_values[ind]])


if __name__ == "__main__":
    data = np.random.normal(0, 1, 400)
    stream = Stream(p_val_thresh=.05, alarm_interval=10)
    stream.run(data)
    stream.alarms
    stream.alarm_times
