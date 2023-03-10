''' Vance Wheeler 15 Dec 2022 - Using the python serial package to pass commands and read outputs from the ESP301
     controller. This is defining a module which simplifies the future use. Most of the functions defined here are
     built with manual control in mind, but to log this automatically. Can be added to or extended for more automated
     applications too.

'''
import time
import os.path
import serial
import decimal

class esp301_interface:

    def  __init__(self,port,baudrate=19200,axis=1,initialize=True,startlogging=None):
                                       # Specify the name of the port and baud at which the
                                       # controller is located, and the axis which should
				       # be referenced, for now assume this is axis 1
                                       
                                       # note default for rs232 is 19200, over usb is 921600 only

				       # Initialize tells the module it needs to  enable motors,
				       #  and search for home - default True, specify False if already called
				       #  previously and deem it unneccesary
				       # Begin_log tells it to set self.logging during initialization to begin with
				       #  by specifying the file name
        
        self.a = axis # internal axis variable
        self.dev = serial.Serial(port,baudrate=baudrate,timeout=30) # assign device object using serial, set baudrate and
                                               #  and read timeout, otherwise defaults are fine
					       # note, if the rotation speed is set particularly
					       #  slow, timeout may need to be increased
        err_msg = self._TE() # initial error check
        if float(err_msg) != 0:
            print('Error encountered during initial connection to controller:',err_msg)
	    print('See ESP 301 manual for lookup table (often can be ignored)'


        if(initialize):
            bytesent = self.dev.write(b'%dMO\r'%self.a) # first make sure the stage motor is enabled
            bytesent = self.dev.write(b'%dOR\r'%self.a) # find home, in other words find the actual defined 0
	                                # rather than using the current position on power up as 0. Also move there.
            self._WS()

        self.logging = False
        if startlogging is not None: # assumes string filename. Should at some point add checks for type, etc.
            self.startlog(startlogging)

# Create functions for the useful commands, generally for internal use
#  these use the same serial commands as the manual for simplicity
    def _TE(self): # report error buffer
        bytesent = self.dev.write(b'TE\r') # bytesent is a temp variable to store the returned value of bytes sent output
                                           #  by serial's write command
        return self.dev.readline()

    def _WS(self): # wait for axis to stop before executing next command - should follow any motion command here
        bytesent = self.dev.write(b'%dWS\r'%self.a)

    def _PA(self,pos): # rotate axes to the absolute position pos
        bytesent = self.dev.write(b'%dPA%.3f\r'%(self.a,pos))

    def _PR(self,pos): # rotate the axis by the relative position pos
        bytesent = self.dev.write(b'%dPR%.3f\r'%(self.a,pos))

    def _TP(self): # tell position of axis
        bytesent = self.dev.write(b'%dTP\r'%self.a) # this asks for output, so tell to read for output.
        return decimal.Decimal(self.dev.readline().decode('utf8')) # rather than float() to keep precision,
                                                                   # so first need to decode to str, then decimal
        ## I suspect this is on it's way to becoming a de facto error check too, so keep that in mind

    def _logtime(self,pos=None): # This is used to record the time window and call/contain the funcs to write it
        if self.logging: # to make more efficient, only do something if we are meant to log
            
            tnew = time.time() # get timestamp of the current call
	    
            # determine if this new time is a start or and of the window, this can be done just by the fact that I only
	    #  pass a postion at the end of the move command (when it has arrived at a new pos) so if pos is not none,
	    #  it should be the start of a window - no need of increment and modulus, but a good check.
            if pos is not None: # should be start of window
                self.tstart = tnew
                self.poswindow = pos
            elif self.counter > 0: # should be end of the window, so we should also write it - disregard the
	                           #  first call since it would be incorrectly considered as a window end
                self.tend = tnew
                self.logfile.write('%.3f    %.3f    %.3f\n'%(self.poswindow,self.tstart,self.tend)) # output position
		                 # and time window to the logfile
            else:
                pass
	    
            self.counter += 1 # increment that logtime has been called

        else:
            pass

    # Create comprehensive functions to be called externally

    def move(self,pos):  # move by pos relative to current position
        self._logtime()  # log the time at which a movement command was given (end of window)
        self._PR(pos)    # write motion command
        self._WS()       # tell to wait until stopped moving to execute next command
        out_pos = self._TP() # verify motion complete (this will read buffer until the result of TP is written to it,
	               #  since WS sent, it won't do this until the motion has stopped, but this really means it's hit
		       #  the target, and might need to do a few small corrections for overshoot, etc
        time.sleep(1)  # so sleep for a beat
        out_pos = self._TP()      # then ask for postion again now that it is settled.
        self._logtime(out_pos) # and record the time now that it is settled taking an argument of TP to indicate the
	                      #  start of the time window at this position

	## put a check about if the value is within some tolerance of the angle specified, if not print message
	##  and/or also check for errors after each move command?


    def moveto(self,pos): # absolute move by pos, principally same as move(), but calling PA rather than PR
        self._logtime()
        self._PA(pos)
        self._WS()    
        out_pos = self._TP()
        time.sleep(1)
        out_pos = self._TP() 
        self._logtime(out_pos)

    def pass_cmd(self,cmdstr,read=False): # Allow to pass any command to the esp301, formatting it appropriately
                                    #  really a sort of bypass or debug thing, note that it may mess up the log
				    #  file if told to move by this command.
        bytesent = self.dev.write(b'%b\r'%cmdstr.encode()) # simply fromat the input string
        if read: # if told to read an output, do so
            return self.dev.readline()
    
    def startlog(self,filename,overwrite=False): # start logging movements by printing to specified file
        if self.logging:
            print('Already logging at %s'%self.logfile_name)
        else:

	    # open logfile for now just to write to, in future consider option for append.
            if os.path.exists(filename): # check if file exists already
                print('File %s already exists, cannot start log'%filename)
            else:
                self.logfile_name=filename
                self.logging=True
                self.counter = 0 # initialize the counter used to determine if window timestamps are start or end
                print('Beginning log at %s'%self.logfile_name)

                self.logfile = open(self.logfile_name,'w')
                self.logfile.write('position    t_arrive    t_leave\n') # write simple header for columns

    def endlog(self):
        if not self.logging:
            print('No log to end')
        else:
            print('Ending log at %s'%self.logfile_name)
            if self.counter > 1:
                self._logtime() # call logtime with no position to end the current time window and print
		               #  buffered final position information
            self.logging = False
            self.logfile.close()

    def close(self): # close the connection to the device, stop logging and save files
        
        # currently has no setting to stop motion or clear command buffer on controller, may want to include later

        if self.logging: # if currently logging, end
            self.endlog()
	
        self.dev.close() # close serial connection
        print('Connection to controller closed')

