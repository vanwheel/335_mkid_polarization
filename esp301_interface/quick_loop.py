# This is just an example of how to use esp301_interface.py basic commands.
#  So far I have usually run from command line very interactively, but this could obviously
#  be put in a script to loop pre defined angles.

# VW Mar 9 2023 - This is a quick dirty script used to run a loop of angles

import esp301_interface
import numpy as np
import time

## Below are inputs to specify

# port details
port = '/dev/ttyUSB0' # will need to be specified
baud = 19200 # the default for rs232 config on the controller

# List of angles to visit (Here I avoid going past +-170 degree due to mech stops)
angles = np.arange(-150,150,30,dtype=float)
angles = np.insert(angles,0,-170.0)
angles = np.append(angles,170.0)

# time to wait at that angle
t_wait = 5 # seconds


## Below the script is executed
esp = esp301_interface.esp301_interface(port=port,baudrate=baud) # port name of controller
                # There are other args but defaults should be fine
                # Expects motor connected to axis 1, but can be specified
# begin logging
esp.startlog('quick_loop_out.txt')

for angle in angles:
    esp.moveto(angle)
    time.sleep(t_wait) # wait roughly this time

esp.endlog()
esp.close()
