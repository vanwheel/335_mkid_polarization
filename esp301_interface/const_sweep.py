# VW Mar 13 2023 - This is a quick dirty script used to run a long sweep at constant speed
#  If set very slowly, the script will likely timeout, causing the later calls not to be sent.
#  good enough for use right now at least


import esp301_interface
import time

## Below are inputs to specify

# port details (rs232 to usb adaptor is typical use case)
#port = '/dev/ttyUSB0' # will need to be specified
#baud = 19200 # the default for rs232 config on the controller

# details for my local windows machine using newport usb connection and drivers
port = 'COM6'
baud = 921600

# List of angles (Here I avoid going past +-170 degree due to mech stops)
#  here just start at, then stop at
angles = [-170,170]

spd = 3.000 # desired speed of rotation deg/s

## Below the script is executed
esp = esp301_interface.esp301_interface(port=port,baudrate=baud,initialize=False)
                # Port name etc of controller
                # For mar 13 use case, set initialize to 0,
		#  assume motor on and home found already
                # Expects motor connected to axis 1, but can be specified
esp.setvel() # make sure set to default speed first

esp.moveto(angles[0]) # go to start angle
print('start',time.time())
esp.setvel(spd) # change to desired speed
esp.moveto(angles[1]) # move to stop angle
print('stop',time.time())

esp.setvel() # return to default speed

# for repeated measurements, wait a bit then return to first angle bound
time.sleep(10)
esp.moveto(angles[0])
# now ready for the next


esp.close()
