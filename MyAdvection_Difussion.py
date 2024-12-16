import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 10.0      # Длина области [m] 
T = 15.0       # Время моделирования [sec]
Nx = 1600      # Число узлов по x [ ]
Nt = 3200      # Число временных шагов [ ]
D = 0.1      # Коэффициент диффузии [ ]
v = 0.001       # Скорость адвекции [m/sec ]