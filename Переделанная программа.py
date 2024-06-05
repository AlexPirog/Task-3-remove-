import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

class FieldDisplay:
    def __init__(self, size_m, dx, ymin, ymax, probePos, sourcePos):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot(np.arange(0, size_m, dx), [0]*int(size_m/dx))[0]
        self.ax.plot(probePos*dx, 0, 'xr')  # Позиция датчика
        self.ax.plot(sourcePos*dx, 0, 'ok')  # Позиция источника
        self.ax.set_xlim(0, size_m)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel('Ez, В/м')
        self.probePos = probePos
        self.sourcePos = sourcePos

    def update(self, data):
        self.line.set_ydata(data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Probe:
    def __init__(self, probePos, maxT, dt):
        self.maxT = maxT
        self.dt = dt
        self.probePos = probePos
        self.t = 0
        self.E = np.zeros(self.maxT)

    def add(self, data):
        self.E[self.t] = data[self.probePos]
        self.t += 1

    def showSpectrum(self):
        sp = fftshift(np.abs(fft(self.E)))
        df = 1/(self.maxT*self.dt)
        freq = np.arange(-self.maxT*df /2, self.maxT*df/2, df)
        plt.figure()
        plt.plot(freq, sp/max(sp))
        plt.xlabel('f, Гц')
        plt.ylabel('|S|/|S|max')
        plt.xlim(0, 0.5e9)  # Ограничение частотного диапазона до 0.5 ГГц
        plt.show()

    def showSignal(self):
        plt.figure()
        plt.plot(np.arange(0, self.maxT*self.dt, self.dt), self.E)
        plt.xlabel('t, c')
        plt.ylabel('Ez, В/м')
        plt.xlim(0, self.maxT*self.dt)
        plt.show()

# Параметры моделирования
eps = 3.5  # Диэлектрическая проницаемость
W0 = 120*np.pi  # Коэффициент пространственной дискретизации
Sc = 1  # Коэффициент устойчивости
maxT = 2200  # Количество временных шагов
size_m = 5.5  # Размер области моделирования
dx = size_m / 1875  # Пространственный шаг
maxSize = int(size_m / dx)  # Максимальный размер массива
probePos = int(size_m / 4 / dx)  # Позиция датчика
sourcePos = int(size_m / 2 / dx)  # Позиция источника
dt = dx * np.sqrt(eps) * Sc / 3e8  # Временной шаг

probe = Probe(probePos, maxT, dt)
display = FieldDisplay(size_m, dx, -1, 1, probePos, sourcePos)
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)
Ezq = np.zeros(3)
Ezq1 = np.zeros(3)
A_max = 10
F_max = 0.5e9  # Максимальная частота сигнала
w_g = np.sqrt(np.log(A_max)) / (np.pi * F_max)/dt
d_g = w_g * np.sqrt(np.log(A_max))

# Создание Гауссова импульса
class GaussianPulse:
    def __init__(self, magnitude, dg, wg):
        self.magnitude = magnitude
        self.dg = dg
        self.wg = wg

    def generate(self, time):
        return self.magnitude * np.exp(-((time - self.dg) / self.wg) ** 2)

gaussian_pulse = GaussianPulse(1, d_g, w_g)

# Потери в среде. PML
loss = np.zeros(maxSize)
layer_loss_x = maxSize-100 # Позиция начала слоя PML
loss[layer_loss_x:] = 0.01

# Коэффициенты для расчета поля E
ceze = (1 - loss) / (1 + loss)
cezh = W0 / (eps * (1 + loss))

# Коэффициенты для расчета поля H
chyh = (1 - loss) / (1 + loss)
chye = 1 / (W0 * (1 + loss))

# Моделирование распространения волны
for q in range(1, maxT):
    # Расчет компоненты поля H
    Hy[:-1] = chyh[:-1] * Hy[:-1] + chye[:-1] * (Ez[1:] - Ez[:-1])

    # Применение источника возбуждения (Гауссов импульс)
    Ez[sourcePos] += gaussian_pulse.generate(q)

    # Расчет компоненты поля E
    Ez[1:] = ceze[1:] * Ez[1:] + cezh[1:] * (Hy[1:] - Hy[:-1])

    # Запись данных в датчик
    probe.add(Ez)

    # Обновление отображения поля
    if q % 20 == 0:
        display.update(Ez)


plt.ioff()
probe.showSignal()
probe.showSpectrum()