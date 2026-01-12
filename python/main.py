from env import Simulator
from matplotlib.animation import FuncAnimation
from param import *

s=Simulator()
ani=FuncAnimation(MONITOR_MAP_FIGURE,s.update,blit=True,cache_frame_data=False)
plt.show()

    