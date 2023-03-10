# This is just an example of how to use esp301_interface.py basic commands.
#  So far I have usually run from command line very interactively, but this could obviously
#  be put in a script to loop pre defined angles.

import esp301_interface

esp = esp301_interface.esp301_interface('COM6') # port name of controller
                # There are other args but defaults should be fine
                # Expects motor connected to axis 1, but can be specified

esp.move(94.533) # moves this value in degrees relative to current position
                 #  input up to 3 decimals of precision

esp.moveto(0.000) # moves to absolute position in degrees

# Note that there are mechanical stops at ~ +/- 172 degrees so it will not rotate full 360 degrees
#  this can also currently cause some issues in my script logging with duplicated postions but
#  different time windows.

esp.startlog('filename.txt') # begin logging positions and time windows to specified file
                             #  will start on next movement command

esp.endlog() # stops logging info to file

esp.close() # Terminates serial connection, will end logging if not already done.
