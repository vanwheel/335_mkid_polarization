# VW Mar 9 2023 - This is a quick dirty script used to run a loop of angles

import esp301_interface
import numpy as np
import time

## Below are inputs to specify

# port details (rs232 to usb adaptor is typical use case)
#port = '/dev/ttyUSB0' # will need to be specified
#baud = 19200 # the default for rs232 config on the controller

# port details for my local testing on windows with newport usb connection
port = 'COM6'
baud = 921600

# List of angles to visit (Here I avoid going past +-170 degree due to mech stops)
angles = np.arange(-150,150,30,dtype=float)
angles = np.insert(angles,0,-170.0)
angles = np.append(angles,170.0)

# time to wait at that angle
t_wait = 5 # seconds


## Below the script is executed
esp = esp301_interface.esp301_interface(port=port,baudrate=baud,initialize=False) # port name of controller
                # There are other args but defaults should be fine
                # Expects motor connected to axis 1, but can be specified
# begin logging
#esp.startlog('outputs/quick_loop/quick_loop_out.txt')

for angle in angles:
    esp.moveto(angle)
    time.sleep(t_wait) # wait roughly this time

#esp.endlog()
esp.close()
