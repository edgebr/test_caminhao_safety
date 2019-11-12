import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import time
from numpy_ringbuffer import RingBuffer
import math


class LiveGraph:

    def _frame_gen(self):
        dt = 0.1
        i = 0
        while True:
            i += dt
            yield i

    def __init__(self, title: str, color: str, windows_length: float,
                 ymin: float, ymax: float, int_ms: int, data_gen, yticks=[],
                 walk=True):
        self.walk = walk
        self.title = title
        self.fig, self.ax = plt.subplots(1, 1)
        self.ln, = plt.plot([], [], color + '-')
        self.last_frame = 0
        self.xdata = []
        self.ydata = []
        self.xmin = 0
        self.xmax = windows_length
        self.ymin = ymin
        self.ymax = ymax
        self.yticks = yticks
        self.data_gen = data_gen
        self.ani = FuncAnimation(self.fig, self.update,
                                 frames=self._frame_gen(),
                                 init_func=self.init, blit=True,
                                 interval=int_ms)

    def init(self):
        self.ax.set_title(self.title)
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        if self.yticks:
            self.ax.set_yticks(self.yticks)
        self.ax.grid(True)
        return self.ln,

    def update(self, frame):
        dt = frame - self.last_frame
        self.last_frame = frame
        if frame > self.xmax and self.walk:
            self.xmin += dt
            self.xmax += dt
        self.ax.set_xlim(self.xmin, self.xmax)
        self.xdata.append(frame)
        self.ydata.append(self.data_gen(frame))
        self.ln.set_data(self.xdata, self.ydata)
        return self.ln,


class VectorGraph:
    def _frame_gen(self):
        dt = 0.1
        i = 0
        while True:
            i += dt
            yield i

    def __init__(self, title: str, windows_length: float, vmin: float,
                 vmax: float, int_ms: int, data_gen):
        self.title = title
        self.fig, self.ax = plt.subplots(1, 1)
        self.lnx, = self.ax.plot([], [], 'r-')
        self.lny, = self.ax.plot([], [], 'g-')
        self.lnz, = self.ax.plot([], [], 'b-')
        self.last_frame = 0
        self.tdata = []
        self.xdata = []
        self.ydata = []
        self.zdata = []
        self.tmin = 0
        self.tmax = windows_length
        self.vmin = vmin
        self.vmax = vmax
        self.data_gen = data_gen
        self.ani = FuncAnimation(self.fig, self.update,
                                 frames=self._frame_gen(),
                                 init_func=self.init, blit=True,
                                 interval=int_ms)
        self.last_time = 0

    def init(self):
        self.ax.set_title(self.title)
        self.ax.grid(True)
        self.ax.set_xlim(self.tmin, self.tmax)
        self.ax.set_ylim(self.vmin, self.vmax)
        return self.lnx, self.lny, self.lnz

    def update(self, frame):
        self.last_time = time()
        dt = frame - self.last_frame
        self.last_frame = frame
        if frame > self.tmax:
            self.tmin += dt
            self.tmax += dt
        self.tdata.append(frame)
        x, y, z = self.data_gen(frame)
        self.ax.set_xlim(self.tmin, self.tmax)
        self.xdata.append(x)
        self.ydata.append(y)
        self.zdata.append(z)
        self.lnx.set_data(self.tdata, self.xdata)
        self.lny.set_data(self.tdata, self.ydata)
        self.lnz.set_data(self.tdata, self.zdata)

        # print(f'{self.title}: {1/(time() - self.last_time)/1000} kHz')
        return self.lnx, self.lny, self.lnz


class VectorFilteredGraph:
    def _frame_gen(self):
        dt = 0.1
        i = 0
        while True:
            i += dt
            yield i

    def __init__(self, title: str, windows_length: float, vmin: float,
                 vmax: float, int_ms: int, data_gen, filter_points: list, gain=1):
        self.title = title
        self.k = gain
        self.ax = [None] * 3
        gridsize = (2, 1)
        self.fig = plt.figure(figsize=(5, 6))
        self.ax[0] = plt.subplot2grid(gridsize, (0, 0))
        self.ax[1] = plt.subplot2grid(gridsize, (1, 0))

        self.gain = gain
        # self.fig.set_size_inches(8, 8)
        self.lnx = []
        self.lny = []
        self.lnz = []
        for i in range(2):
            self.lnx.append(self.ax[i].plot([], [], 'r-')[0])
            self.lny.append(self.ax[i].plot([], [], 'g-')[0])
            self.lnz.append(self.ax[i].plot([], [], 'b-')[0])
        self.last_frame = 0
        self.tdata = [[], []]
        self.xdata = [[], [], []]
        self.ydata = [[], [], []]
        self.zdata = [[], [], []]
        self.ring = {
            'x': RingBuffer(capacity=filter_points[0], dtype=float),
            'y': RingBuffer(capacity=filter_points[0], dtype=float),
            'z': RingBuffer(capacity=filter_points[0], dtype=float)
        }
        self.filter_alt_length = filter_points[1]
        self.tmin = 0
        self.tmax = windows_length
        self.vmin = vmin
        self.vmax = vmax
        self.data_gen = data_gen
        self.ani = FuncAnimation(self.fig, self.update,
                                 frames=self._frame_gen(),
                                 init_func=self.init, blit=True,
                                 interval=int_ms)
        self.last_time = 0

    def init(self):
        titles = [
            self.title,
            self.title + ' - low pass filter',
            self.title + ' - alternative filter'
        ]

        out = []

        for i in range(2):
            self.ax[i].set_title(titles[i])
            self.ax[i].grid(True)
            self.ax[i].set_xlim(self.tmin, self.tmax)
            self.ax[i].set_ylim(self.vmin, self.vmax)

            out.append(self.lnx[i])
            out.append(self.lny[i])
            out.append(self.lnz[i])

        return tuple(out)

    def _filter_single_lp(self, data, ring_name):
        mean = 0
        self.ring[ring_name].append(data)
        for d in self.ring[ring_name]:
            mean += d
        mean /= len(self.ring[ring_name])

        return mean * self.k

    def filter_lp(self, data_x, data_y, data_z):
        xmean = self._filter_single_lp(data_x, 'x')
        ymean = self._filter_single_lp(data_y, 'y')
        zmean = self._filter_single_lp(data_z, 'z')

        return xmean, ymean, zmean

    def update(self, frame):
        self.last_time = time()
        dt = frame - self.last_frame
        self.last_frame = frame
        if frame > self.tmax:
            self.tmin += dt
            self.tmax += dt
        self.tdata[0].append(frame)

        el = {
            'x': [0.0] * 3,
            'y': [0.0] * 3,
            'z': [0.0] * 3
        }

        el['x'][0], el['y'][0], el['z'][0] = self.data_gen(frame)
        el['x'][0], el['y'][0], el['z'][0] = el['x'][0] * self.gain, el['y'][0] * self.gain, el['z'][0] * self.gain
        el['x'][1], el['y'][1], el['z'][1] = self.filter_lp(el['x'][0], el['y'][0], el['z'][0])

        out = []

        for i in range(2):
            x, y, z = el['x'][i], el['y'][i], el['z'][i]
            self.ax[i].set_xlim(self.tmin, self.tmax)
            self.xdata[i].append(x)
            self.ydata[i].append(y)
            self.zdata[i].append(z)
            self.lnx[i].set_data(self.tdata[0], self.xdata[i])
            self.lny[i].set_data(self.tdata[0], self.ydata[i])
            self.lnz[i].set_data(self.tdata[0], self.zdata[i])

            out.append(self.lnx[i])
            out.append(self.lny[i])
            out.append(self.lnz[i])

        return tuple(out)


class HardBrakeGraph:
    def _frame_gen(self):
        dt = 0.1
        i = 0
        while True:
            i += dt
            yield i

    def __init__(self, title: str, windows_length: float, int_ms: int,
                 data_gen, ts, te):
        self.title = title
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.set_size_inches(8, 3.8)
        self.lnx, = self.ax.plot([], [], 'r-')
        self.lny, = self.ax.plot([], [], 'g-')
        self.last_frame = 0
        self.tdata = []
        self.xdata = []
        self.ydata = []
        self.data = []
        self.filter_ring = RingBuffer(capacity=10, dtype=float)
        self.ring = RingBuffer(capacity=int(windows_length * 5), dtype=float)
        self.ring2 = RingBuffer(capacity=int(windows_length * 5), dtype=float)
        self.tmin = 0
        self.count = 0
        self.evt_status = 0
        self.tmax = windows_length
        self.data_gen = data_gen
        self.ts = ts
        self.te = te
        self.ani = FuncAnimation(self.fig, self.update,
                                 frames=self._frame_gen(),
                                 init_func=self.init, blit=True,
                                 interval=int_ms)
        self.last_time = 0

    def init(self):
        self.ax.set_title(self.title)
        self.ax.grid(True)
        self.ax.set_xlim(self.tmin, self.tmax)
        self.ax.set_ylim(-0.1, 1.1)

        return self.lnx, self.lny

    def _filter(self, data: float) -> float:
        summ = 0
        self.filter_ring.append(data)
        for d in self.filter_ring:
            summ += d
        summ = summ / len(self.filter_ring)

        return summ

    def _build_data(self):
        data = []
        data += self.ring
        data += self.ring2

        maxx = max(data)
        minn = min(data)

        for d in data:
            if maxx - minn == 0:
                self.xdata.append(math.inf)
            else:
                self.xdata.append((d - minn)/(maxx - minn))

    def _build_template(self):
        for t in range(int(self.tmax * 10)):
            dt = t * 0.1
            if dt <= self.ts or dt >= self.te:
                self.ydata.append(1)
            else:
                y = 0.5*(1 + math.cos(2*math.pi*(dt - self.ts)/(self.te - self.ts)))
                self.ydata.append(y)

    def update(self, frame):
        d = self.data_gen(frame)
        d = self._filter(d)
        d = -abs(d)

        threshold = -0.3

        if d <= threshold:
            self.evt_status = True

        if self.evt_status:
            self.count += 1
            self.ring2.append(d)
        else:
            self.ring.append(d)

        if self.count >= int(self.tmax * 5):
            self.tdata.clear()
            self.xdata.clear()
            self.ydata.clear()

            # build t
            for dt in range(int(self.tmax * 10)):
                self.tdata.append(dt * 0.1)

            # build x
            self._build_data()

            # build y
            self._build_template()

            self.lnx.set_data(self.tdata, self.xdata)
            self.lny.set_data(self.tdata, self.ydata)

            self.count = 0
            self.evt_status = False

        return self.lnx, self.lny
