from env import Simulator
from matplotlib.animation import FuncAnimation
from param import *
from gen_head_file import generate_param_header
generate_param_header()
s=Simulator()
ani=FuncAnimation(MONITOR_MAP_FIGURE,s.update,blit=True,interval=0,cache_frame_data=False)
plt.show()

    