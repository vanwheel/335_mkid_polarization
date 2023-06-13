''' IQ to df/f Vance Wheeler Apr 18 2023
This module contains methods written to get df/f from sweep reduc and timestream outputs of the roach,
but is likely a bit more generally applicable.

Given the sweep file, will transform the data and interpolate the function needed to find df/f from
a timestream data file. The parameters of this transform and the interpolator are saved so they
can be applied to the timestream data.

Given a timestream file, will apply the transform to the relevant data, interpolate f, and calculate df/f
'''

import numpy as np
import scipy.optimize
import scipy.interpolate
import fnmatch


def read_sweep(swfile): # simply read the sweep data
    # call find_sweep? Then read the file it produces - or seperate to just do with the output of find_sweep,
    #  for now just assume it takes specified infile
    
    # this is fine for now, but see above markdown cell for reasons why this is overkill and I could just be using
    #  the .file attribute to find the "keys" for a numpy file.
    
    # read data
    data_in = np.load(swfile,allow_pickle=True,encoding='latin1') # extra args needed in case of Tyler's conversion
                                                                  # in case of Elyssa's, not needed
    try: data_in.keys() # check to make sure the input is dict like, if not, could be 0d array,
    except AttributeError:
            data_in = data_in.item() # use .item() to recover dict within
            pass
    
    # check if keys are standard string or bytes, assign string start appropriately
    if list(data_in.keys())[0][:2] == "b'": strt = "b'"
    else: strt = ""
    
    # find kid channel freqs
    #  use fnmatch to search keys with wildcard since end may/may not have ' in the string
    lo = data_in[fnmatch.filter(data_in.keys(),strt+"lo_freqs*")[0]] # local oscillator freqs
    bb = data_in[fnmatch.filter(data_in.keys(),strt+"bb_freqs*")[0]] # baseband freq, each index corresponds to one channel
    refreq = data_in[fnmatch.filter(data_in.keys(),strt+"calparam.f0s*")[0]] # so called res freq chosen
                 # during sweep calibration to monitor

    sw_freq = [b + lo for b in bb] # create single array for freqs of size n channels, freq.
                                  #  less mem efficient, but simpler
    
    # find kid channel IQ data
    chan_keys = fnmatch.filter(data_in.keys(),strt+"K*") # for now I assume this is sorted automatically
    
    kidno = []
    sw_iq=[]
    for k in chan_keys:    
        kk = k.split("'")
        if len(kk) > 1: kidno.append(kk[1]) # if there is ' in the str, take the part between them
        else: kidno.append(kk[0]) # otherwise just take the whole value

        sw_iq.append(data_in[k])

    
    return np.array(kidno),np.array(sw_freq),np.array(sw_iq),refreq

def line(x,m,b): # Simple linear fit func
    return m*x + b

def cable_fit(sw_freq,sw_iq): # this fits the cable delay to produce the parameter needed to circularize

    pointno = 30 # number of points on each end of the array to fit
    
    # find and unwrap phase:
    phi = np.arctan2(np.imag(sw_iq),np.real(sw_iq))
    phi = unwrap(phi)

    slope =[]
    for i in range(len(phi)):
    
        # fit linear model to phase(f) to first and last pointno points
        ## should get a way to estimate p0
     
        popt_lo,pcov_lo = scipy.optimize.curve_fit(line,sw_freq[i,:pointno],phi[i,:pointno],p0=[1e-7,1e3]) # lo freq end
        popt_hi,pcov_hi = scipy.optimize.curve_fit(line,sw_freq[i,-pointno:],phi[i,-pointno:],p0=[1e-7,1e3]) # high freq end
    
        # average the 2 models to find slope, intercept doesn't matter, const phase offset
        #  will be accounted in later processing
    
        slope.append((popt_lo[0]+popt_hi[0])/2)
    
    return np.array(slope)

def circularize(freq_in,iq_in,slope):
    
    # Correct phase by the slope caused primarily by cable delay
    circ_iq =[]
    for i in range(len(iq_in)):
        phi = np.arctan2(np.imag(iq_in[i]),np.real(iq_in[i]))
        mag = np.absolute(iq_in[i])
        
        # apply correction
        phi = phi - slope[i]*freq_in[i]
    
        # convert back to complex
        circ_iq.append(mag*np.cos(phi) + 1j*mag*np.sin(phi))
    
    return np.array(circ_iq)

def get_center_simple(circ_iq): # this finds the center coords to be used for translation and rotation, needs circularized first
    # this is just super simple currently, as above taking the average of the max extents - can be changed later
    center = []
    for iq in circ_iq:
        xcen = (np.min(np.real(iq)) + np.max(np.real(iq)))/2
        ycen = (np.min(np.imag(iq)) + np.max(np.imag(iq)))/2
        
        # combine to single complex number again
        center.append(xcen + 1j*ycen)
    
    return np.array(center)

def fit_ellipse(x, y):
    """
    VW: From https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    VW: From https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    VW: From https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def get_center(circ_iq,full_output=False): # using ellipse fitting method from an article on scipython.org
    # full out prints all the params, used in case I want to plot the fitted ellipse.
    end_cut = 100 # cut this many points from either end of the IQ loop to remove the dense junk tails
                  #  and keep mostly just the actual loop. In future this can be improved by making cuts
                  #  for each data set based on point density or similar.
    ellicent = []
    elliparams = []
    for iq in circ_iq:
        ellicoeffs = fit_ellipse(np.real(iq[end_cut:-end_cut]),np.imag(iq[end_cut:-end_cut]))
        
        elliparams.append(cart_to_pol(ellicoeffs)) # turn cartesian expanded coeffs into meaningul parameters
         
        ellicent.append(elliparams[-1][0] + 1j*elliparams[-1][1]) # combine the x and y center params into single complex
        
    if full_output:
        return np.array(ellicent),np.array(elliparams)
    else:
        return np.array(ellicent)

def trans_rot(circ_iq,center):
    # translate the circularized data by xcen, and rotate by phase defined by xcen
    iq_out = []
    for i in range(len(circ_iq)):
        
        # translate first
        iq_trans = center[i] - circ_iq[i] # gao defines translation in this way
        
        # now rotate
        phi_trans = np.arctan2(np.imag(iq_trans),np.real(iq_trans))
        mag_trans = np.absolute(iq_trans)
        
        phi_cen = np.arctan2(np.imag(center[i]),np.real(center[i]))
    
        phi_out = phi_trans - phi_cen
        
        #phi_out = unwrap(phi_out) # unwrap about -pi to pi so the fit later sees continuous line
        ## Maybe I will simply unwrap before the fit, which I think is where it's most important
        ##  if I output IQ from this, it will just be wrapped again by arctan anyway
            
        # convert back to complex
        iq_out.append(mag_trans*np.cos(phi_out) + 1j*mag_trans*np.sin(phi_out))
        
    return np.array(iq_out)

def transform(freq_in,iq_in,slope,center):
    # This will apply the cicularization and translation+rotation
    # Want this to apply generally, so given params do the transformation. For the sweep data this might be a bit redundant
    #  since to get each of these params we do this step by step - so this might be more for just applying to final data.
    
## Actually, there is a bit more to conisder here about cable delay removal - the timestream data has only a single
##  freq which it is watched it, so will need to pass or handle freqs differently!
##  currently I think the whole phase - slope*f will work fine for array slope or single scalar slope, so just pass it
##  differently!

    iq_out = circularize(freq_in,iq_in,slope)
    iq_out = trans_rot(iq_out,center)
    
    return iq_out

def unwrap(phi_in): # maybe this can go in gao_fit fit?
    # I actually think to be safe the data being read in from timestreams should be unwrapped too since they are to be
    #  interpolated onto the sweep fit.
    
    wrap_crit = 3/2*np.pi # the criteria for how close to pi,-pi to consider that it has been wrapped
    
    phi_out = []
    for i in range(len(phi_in)):
        phi = phi_in[i]
    # for the fitting in next step, un wrap phi about -pi to pi since wrapping breaks the fit, check for discontinuity
        discont,k = False,0
        while not discont and k < len(phi)-2: # len -1 for 0 index and -1 more for k+1 used in loop
        # check for discontinuity
            if np.abs(phi[k+1] - phi[k]) > wrap_crit:
                discont = True
            k += 1 # I am relying on adding 1 even after discont found in next step
        if discont:
            means = np.array([np.mean(phi[:k]),np.mean(phi[k:])])
            #print(i,discont,means) ## for diagnostic
            if np.abs(means[0]) > np.abs(means[1]): # the left side needs to be wrapped
                if means[0] < 0: # left side on bottom, needs to be wrapped to top
                    phi[:k] = phi[:k] + 2*np.pi
                else: # on top, needs to be wrapped to bottom
                    phi[:k] = phi[:k] - 2*np.pi
            else: # assume only other case is opposite
                if means[1] < 0: # right side on bottom, needs to be wrapped to top
                    phi[k:] = phi[k:] + 2*np.pi
                else: # on top, needs to be wrapped to bottom
                    phi[k:] = phi[k:] - 2*np.pi
                    
            # need to put in a little extra check to see if some parts ended up being kicked too far the wrong way
            #  since the phase(f) can sometimes wiggle back to the other side of the line, and the above simply moves
            #  everything on one side of the discontinuity
            for m in range(len(phi)):
                if phi[m] > 2*np.pi: phi[m] = phi[m] - 2*np.pi
                if phi[m] < -2*np.pi: phi[m] = phi[m] + 2*np.pi
                    
                # this check is pretty ham fisted, but certainly seems to work at least.
    
        phi_out.append(phi)
    
    return np.array(phi_out)

def gao_model(f,fr,qr,theta_0): # This is the model distribution for phase as func of freq used by Gao
    return -theta_0 + 2*np.arctan(2*qr*(1-f/fr))

def gao_fit(sw_freq,phi_uw): # this is really only to find f and phi at resonace, so depending on if we use
    #                           the freq from this fit as the reference, may not need it
    # Input unwrapped phi and freqs
    
    refreq = [] # resonant point, f0,phi0
    for i in range(len(phi_uw)):
        popt,pcov = scipy.optimize.curve_fit(gao_model,sw_freq[i],phi_uw[i],p0=[sw_freq[i,150],1e3,0.0]) # again would like an estimator
                                                                                           #  to guess p0 better
        refreq.append(np.array(popt[0],gao_model(popt[0],*popt))) # using the y pos that fr gives us,
                # rather than whatever business is up with Gao's use of theta_0 and some geometric relation.
    
    return np.array(refreq)

def get_interp(freq_in,iq_in): # get interpolated function objects for each phi(f) distribution
    
    phi_uw = unwrap(np.arctan2(np.imag(iq_in),np.real(iq_in)))

    interp = []
    for i in range(len(freq_in)):
        # append interpolator func to the output
        interp.append(scipy.interpolate.interp1d(phi_uw[i],freq_in[i],kind='quadratic'))
    
    return np.array(interp) # array of interpolator function objects

def save_params(outfile,kidno,slope,center,refreq,interp): # save parameters, basically think I want to similarly pickle a dict kind of object like Elyssa did
    #outdict = {'kids':kidno,'delay_slope':slope,'center':center,'interpolator':interp} # now just save this!
    # actually it seems that npz already virtually saves as a dict like object with keywords as the array names
    
    # note that .npz will be appended automatically
    
    np.savez_compressed(outfile,kids=kidno,delay_slope=slope,center=center,ref_freq=refreq,interpolator=interp)

def read_params(infile): # read the correct params, which can then be passed into transform
    params_in = np.load(infile,allow_pickle=True)
    
    # break out and return the constituent arrays to be used
    return params_in['kids'],params_in['delay_slope'],params_in['center'],params_in['ref_freq'],params_in['interpolator']

def read_timestream(infile):
    
    # read in the whole data set I think, and pick out the IQ and freqs, as above for the sweep
    # Hmm, I'll want to have some way to index by kid channel - or keep that stored alonge somewhere
    #  for reference later and for lableling, etc
    
    # Tyler's conversion is in a different structure to his conversion of the sweep reduc files, so of course
    #  .files doesn't work. What a pain. For now I will design to just use Elyssa's format. All this will change
    #  when the Roach goes away anyway. Her array names for the ts data are not byte strings, for some reason.
    
    # read data
    data_in = np.load(infile,allow_pickle=True,encoding='latin1') # extra args needed in case of Tyler's conversion
                                                                   # in case of Elyssa's, not needed
#### The .item thing is here to read tyler's conversion, need to add if statements like read_sweep!
    #data_in = data_in.item() # just ended hard code swapping it for like 1 data set
    # there's also big issues in making sure all the fields are properly sorted to correspond
    #  to the assumed order from other modules. That is unfixed currently!
    
    pytime = data_in['python_timestamp'] # just taking this timestamp of the various options
    freq = data_in['f0s'] # freq at which resonator observed
    
    Ikey = fnmatch.filter(data_in,'K*_I')
    Qkey = fnmatch.filter(data_in,'K*_Q')
    
    kidno = np.array([k.split('_')[0] for k in Ikey]) # get the channel labels
    
    ts_iq = []
    for i in range(len(Ikey)):
        # combine IQ into single complex number to fit form I have elsewhere
        ts_iq.append(data_in[Ikey[i]] + (1j*data_in[Qkey[i]]))
    
    return kidno,freq,pytime,np.array(ts_iq)

def get_dfof(iq,refreq,interp): # Called after transforming data, calculate dfof by interpolating

    phi = np.arctan2(np.imag(iq),np.real(iq))
    phi = unwrap(phi) # since func trained on unwrapped data, don't want any problems just in case.
    
    dfof = []
    for i in range(len(phi)):
        dfof.append((interp[i](phi[i]) - refreq[i])/refreq[i]) # change in freq = interp'd freq from current phase - ref freq
                                                         #  scaled by freq monitored at
    return np.array(dfof)

def save_dfof(outfile,kidno,time,dfof): # save the fully processed results
    
    np.savez_compressed(outfile,kids=kidno,time=time,dfof=dfof)

### Below are the sort of main methods which combine the above, and are the ones used in the end

def create_paramfile(path_srd,output_dir,return_path=False): # process everything on the sweep
    # reduc side of things, and create the parameter file needed to calculate dfof
    # Given path of sweep reduc file, and desired output dir for param file
    # can specify to return path of the created file
    
    # read sweep reduc file
    kidno,freq,sw_iq_in,refreq = read_sweep(path_srd)

    slope  = cable_fit(freq,sw_iq_in)
    sw_iq  = circularize(freq,sw_iq_in,slope)
    center = get_center_simple(sw_iq)
    sw_iq  = trans_rot(sw_iq,center)

    interp = get_interp(freq,sw_iq)

    # append dfof params file name to output directory, labeling using the sweep file name.
    param_filname = output_dir+(path_srd.split('/')[-1]).split('_reduc')[0]+'_dfof_params.npz'

    save_params(param_filname,kidno,slope,center,refreq,interp)
    
    if return_path:
        return param_filname

def create_dfoffile(path_ts,path_param,output_dir,return_path=False):
    # Use the parameters generated from sweep to calculate dfof for a timestream file and save it
    # Given the path of the timestream data, the associated srd params, and the output dir
    # Can specify to return path of created file 
     
    kidno_sw,slope,center,refreq_sw,interp = read_params(path_param)

    kidno,refreq,time,ts_iq = read_timestream(path_ts) # both freq from ts and sweep should
    # match, thats the only reason to get it from two sources, but not yet implemented

    iq = transform(refreq,ts_iq,slope,center)

    dfof = get_dfof(iq,refreq,interp)

    # similar to above method for determining the filename
    dfof_filname = output_dir+(path_ts.split('/')[-1]).split('.')[0]+'_dfof.npz'

    save_dfof(dfof_filname,kidno,time,dfof)

    if return_path:
        return dfof_filname
