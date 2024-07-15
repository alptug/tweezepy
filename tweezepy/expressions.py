import autograd.numpy as np
"""
Closed-form expressions for the expected AV and PSD of a bead trajectory.
"""

def SMMAV(t,ts,g,k,e,kT = 4.1):
    """
    Analytical function for the AV of a trapped bead.
    Eq. 17 from Lansdorp et al. (2012) for the single-molecule allan variance.

    Parameters
    ----------
    t : array
        taus.
    ts : float
        Sampling time.
    g : float
        drag coefficient.
    k : float
        spring constant.
    e : float
        tracking error.
    kT : float
        Thermal energy in pN nm. Default value is 4.1

    Returns
    -------
    oav : array
        theoretical allan variance.

    """
    tc = np.true_divide(g,k)
    oav = 2.*kT*tc/(k*t) * (1. + 2. * (tc/t)*np.exp(-t/tc) - (tc/(2.*t))*np.exp(-2.*t/tc) - 3.*tc/(2.*t)) + pow(e,2)*ts/t
    return oav

def SMMHV(t,ts,g,k,e,kT = 4.1):
    """
    Analytical function for the HV of a trapped bead.
    An equation derived by AU

    Parameters
    ----------
    t   : array
        taus.
    ts  : float
        Sampling time.
    g   : float
        drag coefficient.
    k   : float
        spring constant.
    e   : float
        white tracking error.
    kT  : float
        Thermal energy in pN nm. Default value is 4.1

    Returns
    -------
    oav : array
        theoretical allan variance.

    """
    tc = np.true_divide(g,k)
    ohv = 2.*kT*tc/(k*t) * (1. + 2.5 * (tc/t)*np.exp(-t/tc) - (tc/t)*np.exp(-2.*t/tc) + (tc/(6.*t))*np.exp(-3.*t/tc) - 5.*tc/(3.*t))
    if isinstance(e, (int, float,np.float64,np.float32,np.float16,np.int64,np.int32,np.int16,np.int8)):
        ohv += e/t
        if e != 0:
            raise ValueError('e should be an array')
    else:
        ohv += HV_noise(t,ts,*e)
    return ohv

def SMMHV_noise(t,ts,g,k,e, e_1, e_2,kT = 4.1):
    """
    Analytical function for the HV of a trapped bead.
    An equation derived by AU

    Parameters
    ----------
    t   : array
        taus.
    ts  : float
        Sampling time.
    g   : float
        drag coefficient.
    k   : float
        spring constant.
    e   : float
        white tracking error.
    e_1 : float
        pink tracking error.
    e_2 : float
        brown tracking error.
    kT  : float
        Thermal energy in pN nm. Default value is 4.1

    Returns
    -------
    ohv : array
        theoretical hadamard variance.

    """
    tc = np.true_divide(g,k)
    ohv = 2.*kT*tc/(k*t) * (1. + 2.5 * (tc/t)*np.exp(-t/tc) - (tc/t)*np.exp(-2.*t/tc) + (tc/(6.*t))*np.exp(-3.*t/tc) - 5.*tc/(3.*t)) + 0.5*e*e*ts/t + 0.5 * np.log(256/27) * e_1*e_1 + (np.pi*np.pi / 3.) * e_2* e_2 * t / ts
    return ohv

def HV_noise(t,ts,h_m2,h_m1,h0):
    """
    Analytical function for the HV of power-law noise.
    TODO: Add reference

    Parameters
    ----------
    t   : array
        taus.
    ts  : float
        Sampling time.
    h_m2 : float
        violet tracking error.
    h_m1 : float
        blue tracking error.
    h0   : float
        white tracking error.
    h1   : float
        pink tracking error.
    h2   : float
        brown tracking error.

    Returns
    -------
    ohv : array
        theoretical hadamard variance.

    """
    fs = 1/ts
    ohv = 0.5*h0/t + 0.5 * np.log(256/27) * h_m1 + (np.pi*np.pi / 3.) * h_m2 * t #+ 5. * h2 * fs / (6. * (np.pi * t)**2) + (5. * h1  / (6. * (np.pi**3 * t)**2)) * (np.euler_gamma + 0.1 * np.log(48) + np.log(fs * np.pi * t) )
    return 2*ohv
def lansdorpPSD(f,fs,g,k,e,kT = 4.1):
    """
    Analytical function for the PSD of a trapped bead with aliasing and lowpass filtering.
    Eq. 7 in Lansdorp et al. (2012).

    Parameters
    ----------
    f : array-like
        frequency.
    fs : float
        Acquisition frequency.
    g : float
        drag coefficient.
    k : float
        spring constant.
    kT : float
        Thermal energy in pN nm, defaults to 4.1

    Returns
    -------
    PSD : array
        theoretical power spectral density.
    """
    tc = g/k
    fc = 1./tc
    PSD = 2.*kT*tc/k * (1. + 2.*tc*fs*np.sin(np.pi*f/fs)**2 * np.sinh(fc/fs)/(np.cos(2.*np.pi*f/fs) - np.cosh(fc/fs))) + pow(e,2)/fs
    return PSD
    

def aliasPSD(f,fs,a,k,e, kT = 4.1):
    """
    Analytical function for the PSD of a trapped bead with aliasing.
    Eq. 8 in Lansdorp et al. (2012).

    Parameters
    ----------
    f : array-like
        frequency.
    fs : float
        Acquisition frequency.
    a : float
        alpha.
    k : float
        kappa.
    kT : float
        Thermal energy in pN nm. Default value is 4.1

    Returns
    -------
    PSD : array
        theoretical power spectral density.
    """
    kT = 4.1 # thermal energy in pNnm
    return kT/(k*fs) * (np.sinh(k/(a*fs))/(np.cosh(k/(a*fs))-np.cos(2*np.pi*f/fs))) + pow(e,2)/fs
