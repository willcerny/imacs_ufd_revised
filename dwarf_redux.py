import sys
from mpfit import mpfit
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy import stats
import os
from scipy import ndimage
from numpy.polynomial.legendre import legfit, legval
import time
from scipy.interpolate import CubicSpline

import warnings
warnings.filterwarnings("ignore")

#################################################
######### SET PARAMETERS TILL LINE 100 ##########
#################################################

# number of mcmc steps. 1000 for official runs, 100 for test runs to save time.
nsam=1000

# snr threshold, any spectra with snr below snr_min will be skipped for RV or EW fit.
snr_min = 2

# normalize the spectra before combining the spectra (default: 1),  0 was used in the past (i.e. normalization done after combine)
normalizeb4combine = 1

# resample spec or not, default: uniform_resample = 0
# in the past, uniform_resample = 1 was used
uniform_resample = 0

# resample step, if uniform_resample = 0, then this number is not used in the program since no resampling will be done.
# used 0.1 A in the past, should use 0.19 A in the future since that is the actual grid if uniform_resample = 1.
# if using step other than 0.19, then the error spectra will be wrong. TODO, rescale the error spectra for a different step size.
resample_step = 0.19

# if doing cubicspline interpolation, then cubic = 1. If linear then cubic = 0
cubic = 1

#save output catalog or not
savedata = 0

#show the combine spectra that is used for the fit
showcombinedspectra = 0

#show plots for how normalization is done
show_normalization = 0

#show the spectra with the best fit RV templates around CaT lines
showrvplot = 0
#save the plot above
savervplot = 0

#show the spectra with the best fit CaT EW
showewplot = 0
#save the plot above
saveewplot = 0

#display the parameters from the EW fit in the terminal (but not saved)
dispara = 0

#do you want to assess the fitting quality manually? if yes, then zquality = 1 and it will ask you to enter 1 or 0 during the run, usually 1 = good fit, 0 = bad fit
zquality = 0

#write the combined and normalized spectra to a txt file (including all wavelength, not just the spectral fitting window)
#saved txt file will be in the same folder as the input fits file, i.e. objdir (see below)
writespec = 0

#directory for output catalog and figure
outputdir = './'

#directory for saved figures
figdir = outputdir+'fig_cen1_r2_jul11_jul13/'

#file name for the output catalog
outputfile = outputdir+'cen1_r2_jul11_jul13.txt'

#input directory for the 1D spectra. it should be a directory and the code will run on all spectra (filename ended with .fits) in this directory
objdir = '../spec_1d/cen1_r2_jul11_jul13_spec1d/'

#path for the rv template and telluric template
rv_fname = '/Users/tingli/Dropbox/dwarfgalaxy/Magellan/stds/imacs-030817.fits'
telluric_fname = '/Users/tingli/Dropbox/dwarfgalaxy/Magellan/stds/imacs-aband-063016.fits'

# number of pixels to be removed at the edge since the spectra near the edge are sometimes bad (default = 5)
# you may want to decrease this number if a line of interest (e.g. CaT) is near chip gap
# nbuff cannot be zero due to how the code is written, 1 is the minimal.
nbuff = 5

###########################
######  SINGLE MODE #######
#running with one specific spectrum or not, if single = 1, then the file with object_fname_single will run, not the objdir files
#if single = 1 and bhb = 1, then only BHB template will be fit, not the other two templates
#bhb=1 only work when single=1 (i.e. bhb=1 won't work during batch mode above)
single = 1
bhb = 0
object_fname_single = \
'/Users/tingli/Dropbox/dwarfgalaxy/Magellan/Car23_IMACS/1Dspec_new/car2r1_spec1d_n4only/spec1d.car2r1xx.2017239293.fits'
##########################
##########################

##############################################
######## END OF PARAMETER SETTING HERE #######
##############################################

# spectra display window (just for plotting)
CaT1min=8480
CaT1max=8520

CaT2min=8520
CaT2max=8565

CaT3min=8640
CaT3max=8680

# spectra fitting window
wlmaskmin = 8450
wlmaskmax = 8685 # was using 8700 in the past but now switch to match with Josh's

# BHB spectra fitting window
wlmaskmin_bhb = 8450
wlmaskmax_bhb = 8900

#speed of light
c = 2.99792458e5

#create the path for storing the figures if not exist
if not os.path.exists(figdir):
    os.makedirs(figdir)

    
def normalize_spec(wl, spec, dspec):
    """
    normalize the spectra with a legendre polynomial
    """
    idx = (np.isnan(spec))
    spec[idx] = 0
    dspec[idx] = 1e15
    idx = np.isnan(dspec)
    dspec[idx] = 1e15
    idx = spec < 0
    dspec[idx] = 1e15
    idx = dspec < 0
    dspec[idx] = 1e15

    snr = np.median(spec/dspec)
    thlow = 0.8
    thhigh = 1.2
    maxiter = 10

    cont = np.median(spec)
    idx1 = (spec/cont > thlow) & (spec/cont < thhigh)


    #if wl.min() < 7530:
    #    lowbound = 7530
    #else:
    #    lowbound = 7580
    lowbound = 7580

    if snr > 7:
        for i in range(maxiter):
            idx1 = idx1 & ((wl < lowbound) | (wl > 7700))
            #plt.plot(wl[idx1], spec[idx1], lw = 2)
            z = legfit(wl[idx1], spec[idx1], 2, w = 1./dspec[idx1])
            cont = legval(wl,z)
            idx2 = (spec/cont > thlow) & (spec/cont < thhigh) & ((wl < lowbound) | (wl > 7700))
            if all(idx1 == idx2):
                break
            else:
                idx1 = idx2
    else:
        z = legfit(wl[idx1], spec[idx1], 0, w = 1./dspec[idx1])
        cont = legval(wl,z)

    if show_normalization:
        plt.show()
        plt.figure()
        plt.plot(wl, spec)
        plt.plot(wl[idx1], spec[idx1], lw = 2)
        plt.plot(wl, cont)
        plt.show()
        
    spec = spec/cont
    dspec = dspec/cont

    return spec,dspec
'''

def normalize_spec(wl, spec, dspec):
    idx = (np.isnan(spec))
    spec[idx] = 0
    dspec[idx] = 1e15
    idx = np.isnan(dspec)
    dspec[idx] = 1e15
    idx = spec < 0
    dspec[idx] = 1e15
    idx = dspec < 0
    dspec[idx] = 1e15

    scale = np.median(spec)
    spec = spec / scale
    dspec = dspec / scale

    idx = spec > 0.9

    if np.median(spec / dspec) > 7:
        z = np.polyfit(wl[idx], spec[idx], 2, w=1. / dspec[idx])
    else:
        z = np.polyfit(wl[idx], spec[idx], 0)
    p = np.poly1d(z)
    spec = spec / p(wl)
    dspec = dspec / p(wl)

    # plt.plot(pixels, p(pixels)/max(p(pixels)))
    # plt.show()

    return spec, dspec
'''

def lp_post(rv, rvmin, rvmax, mask, wl, model, obj, objerr):
    lp = -np.inf
    
    if rv < rvmax and rv > rvmin:
        z = rv/c
        lp_prior=0.0

        new_wl = wl*(1+z)
        if cubic:
            p = CubicSpline(new_wl,model)
            model = p(wl)
        else:
            model = np.interp(wl,new_wl,model)
        model = model[mask]
        obj = obj[mask]
        objerr = objerr[mask]

        lp_post= - np.sum((obj-model)**2/(2.0*(objerr**2)))

        if np.isfinite(lp_post):
            lp=lp_post+lp_prior

    return lp

def chi2cal(theta, mask, wl, model, obj, objerr):
    rv = theta
    z = rv/c

    new_wl = wl*(1+z)
    if cubic:
        p = CubicSpline(new_wl,model)
        model = p(wl)
    else:
        model = np.interp(wl,new_wl,model)
    model = model[mask]
    obj = obj[mask]
    objerr = objerr[mask]
    chi2 = np.sum((obj-model)**2/(objerr**2))
    return chi2

def get_snr(filename):
    data = pyfits.open(filename)
    data[7].verify('fix')
    temp = data[7].data
    snr = np.median(temp['SPEC'].flatten()/np.sqrt(1/(abs(temp['IVAR']).flatten())))
    return snr


def combine_imacs_spec_resample_uniform(filename, nbuff=3):
    """
    resample to a common grid with 0.19 step size
    """
    data = pyfits.open(filename)
    wl = np.arange(7400, 9000, resample_step)
    spec = np.zeros([len(wl), 4])
    dspec = np.zeros([len(wl), 4])
    spec_wgt = np.zeros([len(wl), 4])
    k = 0
    for i in range(5, 9):
        data[i].verify('fix')
        temp = data[i].data
        wltemp = temp['LAMBDA'].flatten()[::-1][nbuff:-nbuff]
        spectemp = temp['SPEC'].flatten()[::-1][nbuff:-nbuff]
        dspectemp = np.sqrt(1. / (abs(temp['IVAR']).flatten()))[::-1][nbuff:-nbuff]
        if normalizeb4combine:
            spectemp, dspectemp = normalize_spec(wltemp, spectemp, dspectemp)
        # plt.plot(wltemp, spectemp,'k')
        spec[:, k] = np.interp(wl, wltemp, spectemp, left=0., right=0.)
        dspec[:, k] = np.interp(wl, wltemp, dspectemp, left=1.e99, right=1.e99)
        spec_wgt[:, k] = 1. / dspec[:, k] ** 2
        k = k + 1

    speccombine = np.sum(spec * spec_wgt, axis=1) / np.sum(spec_wgt, axis=1)
    dspeccombine = np.sqrt(1. / np.sum(spec_wgt, axis=1))

    if writespec:
        np.savetxt(filename+'.txt', np.column_stack((wl,speccombine)))
        print('save to', filename+'.txt')

    if showcombinedspectra:
        plt.plot(wl, speccombine/np.median(speccombine),'k')
        plt.plot(wl, dspeccombine/np.median(speccombine),'b')
        plt.ylim(-1,2)
        #plt.xlim(wlmaskmin, wlmaskmax)
        plt.show()

    return wl, speccombine, dspeccombine

def combine_imacs_spec_resample(filename, nbuff=3):
    """
    resample to a common grid defined by the raw 1D spectra.
    for the overlap region, the redder spectra's grid is used
    """
    data = pyfits.open(filename)

    temp = data[5].verify('fix')
    temp = data[5].data
    wl = temp['LAMBDA'].flatten()[::-1][nbuff:-nbuff]
    for i in range(6, 9):
        temp = data[i].verify('fix')
        temp = data[i].data
        wltemp = temp['LAMBDA'].flatten()[::-1][nbuff:-nbuff]
        wl = np.concatenate((wl[wl < wltemp[0]],wltemp))
    spec = np.zeros([len(wl), 4])
    dspec = np.zeros([len(wl), 4])
    spec_wgt = np.zeros([len(wl), 4])
    k = 0
    for i in range(5, 9):
        data[i].verify('fix')
        temp = data[i].data
        wltemp = temp['LAMBDA'].flatten()[::-1][nbuff:-nbuff]
        spectemp = temp['SPEC'].flatten()[::-1][nbuff:-nbuff]
        dspectemp = np.sqrt(1. / (abs(temp['IVAR']).flatten()))[::-1][nbuff:-nbuff]
        if normalizeb4combine:
            spectemp, dspectemp = normalize_spec(wltemp, spectemp, dspectemp)
        spec[:, k] = np.interp(wl, wltemp, spectemp, left=0., right=0.)
        dspec[:, k] = np.interp(wl, wltemp, dspectemp, left=1.e99, right=1.e99)
        spec_wgt[:, k] = 1. / dspec[:, k] ** 2
        k = k + 1

    speccombine = np.sum(spec * spec_wgt, axis=1) / np.sum(spec_wgt, axis=1)
    dspeccombine = np.sqrt(1. / np.sum(spec_wgt, axis=1))
    return wl, speccombine, dspeccombine


def combine_imacs_spec(filename, nbuff=3):
    '''
    combine the four chips. Use this as default
    '''
    data = pyfits.open(filename)
    data[5].verify('fix')
    temp = data[5].data
    wl = temp['LAMBDA'].flatten()[::-1][nbuff:-nbuff]
    spec = temp['SPEC'].flatten()[::-1][nbuff:-nbuff]
    dspec = np.sqrt(1. / (abs(temp['IVAR']).flatten()))[::-1][nbuff:-nbuff]
    if normalizeb4combine:
        spec, dspec = normalize_spec(wl, spec, dspec)
    if showcombinedspectra:
        plt.plot(wl, spec)
    
    overlap = 0

    for i in range(6, 9):
        data[i].verify('fix')
        temp = data[i].data
        wltemp = temp['LAMBDA'].flatten()[::-1][nbuff:-nbuff]
        spectemp = temp['SPEC'].flatten()[::-1][nbuff:-nbuff]
        dspectemp = np.sqrt(1. / (abs(temp['IVAR']).flatten()))[::-1][nbuff:-nbuff]
        if normalizeb4combine:
            spectemp, dspectemp = normalize_spec(wltemp, spectemp, dspectemp)
        if showcombinedspectra:
            plt.plot(wltemp, spectemp)
        if wltemp[0]-wl[-1] > 0.19:
            wl_gap = np.arange(wl[-1]+0.19, wltemp[0], 0.19)
            spec_gap = np.zeros_like(wl_gap)
            dspec_gap = np.zeros_like(wl_gap)+1e99
            wl = np.concatenate((wl, wl_gap, wltemp))
            spec = np.concatenate((spec, spec_gap, spectemp))
            dspec = np.concatenate((dspec, dspec_gap, dspectemp))
        elif (wltemp[0] - wl[-1] <= 0.19) & (wltemp[0] - wl[-1] > 0):
            wl = np.concatenate((wl, wltemp))
            spec = np.concatenate((spec, spectemp))
            dspec = np.concatenate((dspec, dspectemp))
        else:
            overlap = 1
            print('THERE ARE OVERLAPS BETWEEN CHIPS!')
    if overlap == 1:
        wl,speccombine,dspeccombine = combine_imacs_spec_resample(filename, nbuff = nbuff)
    else:
        speccombine = spec
        dspeccombine = dspec
    if writespec:
        np.savetxt(filename+'.txt', np.column_stack((wl,speccombine)))
        print('save to', filename+'.txt')

    if showcombinedspectra:
        plt.plot(wl, speccombine/np.median(speccombine),'k')
        plt.plot(wl, dspeccombine/np.median(speccombine),'b')
        plt.ylim(-1,2)
        #plt.xlim(wlmaskmin, wlmaskmax)
        plt.show()

    return wl, speccombine, dspeccombine


def read_rv_stds(filename, stdnum):
    temp = pyfits.open(filename)[0].data[stdnum]
    hdr = pyfits.open(filename)[0].header
    coeff0 = hdr['COEFF0']
    coeff1 = hdr['COEFF1']
    rvwl = 10**(coeff0 + coeff1 * np.arange(len(temp)))
    rvspec = temp.astype('float')
    return rvwl, rvspec
    #new_rvwl = np.arange(7500,9000.1,0.1)
    #new_rvspec = np.interp(new_rvwl, rvwl, rvspec)
    #return new_rvwl, new_rvspec

def read_tell_stds(filename):
    temp = pyfits.open(filename)[0].data
    hdr = pyfits.open(filename)[0].header
    coeff0 = hdr['COEFF0']
    coeff1 = hdr['COEFF1']
    rvwl = 10**(coeff0 + coeff1 * np.arange(len(temp)))
    rvspec = temp.astype('float')
    return rvwl, rvspec
    #new_rvwl = np.arange(7500,9000.1,0.1)
    #new_rvspec = np.interp(new_rvwl, rvwl, rvspec)
    #return new_rvwl, new_rvspec

def get_rv(wl, spec, dspec, rvwl, rvspec, object, rvstar):
    
    if single and bhb:
        fitstart = (np.abs(wl-8400)).argmin()
        fitend = (np.abs(wl-9000)).argmin()
    else:
        fitstart = (np.abs(wl-8400)).argmin()
        fitend = (np.abs(wl-8700)).argmin()

    spec = spec[fitstart:fitend]
    dspec = dspec[fitstart:fitend]
    wl = wl[fitstart:fitend]
    
    spec,dspec = normalize_spec(wl, spec, dspec)

    ndim=1
    nwalkers=20
    rvmin = -800
    rvmax = 800
    
    nstars = len(rvstar)
    rvdist = np.zeros([nstars, nwalkers * nsam])
    chi2rv = np.zeros(nstars)

    rvspec_temp = np.zeros([nstars, len(wl)])

    # MCMC needs some time to produce reasonable "d" from the likelihood, which is called the "burn-in" period.
    # Adjusting the "burn-in" period is quite empirical.
    nburn=50
    
    for kk in range(0, nstars, 1):
        if cubic:
            p = CubicSpline(rvwl[kk], rvspec[kk])
            tempspec = p(wl)
        else:
            tempspec = np.interp(wl, rvwl[kk], rvspec[kk])
        rvspec_temp[kk] = tempspec

    rvspec = rvspec_temp

    if single and bhb:
        wlmask = (wl > wlmaskmin_bhb)  & (wl < wlmaskmax_bhb)
    else:
        wlmask = (wl > wlmaskmin)  & (wl < wlmaskmax)
    
    snr = np.nanmedian(spec[wlmask]/dspec[wlmask])
    print('SNR = '+str(snr))


    for kk in range(0, nstars, 1):
        # here we use the chi-square minimization to find the starting p0
        rvarr =  np.arange(rvmin,rvmax)
        likearr = np.array([lp_post(i,rvmin, rvmax, wlmask, wl, rvspec[kk], spec, dspec) for i in rvarr])
        p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
        #p0= p0 * rvmax * 2 - rvmax
        p0 = p0 + rvarr[max(likearr) == likearr][0]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_post, args=(rvmin, rvmax, wlmask, wl, rvspec[kk], spec, dspec))
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, nsam)

        rvdist[kk, :] = sampler.flatchain[:, 0]
        rv_mean = np.nanmedian(rvdist[kk, :])
        rv_std = np.std(rvdist[kk, :])
        masklen = len(wl[wlmask])
        chi2rv[kk] = chi2cal(rv_mean, wlmask, wl, rvspec[kk], spec, dspec) / masklen
        print(rv_mean, rv_std, chi2rv[kk], rvstar[kk])


    chi2rv[chi2rv == 0] = 1e10
    rvidx = (chi2rv == np.min(chi2rv))
    jj = np.arange(0, nstars)[rvidx][0]
    temp = rvdist[jj]
    #temp_mean = np.nanmedian(temp)
    #if np.std(temp) > 10: temp = temp[(temp < temp_mean + 150) & (temp > temp_mean - 150)]
    temp = stats.sigmaclip(temp, low=5, high=5)[0]
    rv_mean = np.nanmedian(temp)
    #rv_std = np.std(temp)
    rv_std = 0.5 * (np.percentile(temp, 84) - np.percentile(temp, 16))

    print('best fit', rv_mean, rv_std, chi2rv[jj], rvstar[jj])
    
    # Plot the result.
    if single and bhb:
        fig, axarr = plt.subplots(1, 2, figsize=(15,6))
        axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean - 5*rv_std, rv_mean + 5*rv_std])
        axarr[0].set_title('RV Histogram', fontsize=16)
        axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3))
        axarr[0].set_xlabel('RV')
        axarr[0].set_xlim(rv_mean - 5*rv_std, rv_mean + 5*rv_std)
        #axarr[0].axvline(np.percentile(temp, 16), ls='--', color='r')
        #axarr[0].axvline(np.percentile(temp, 84), ls='--', color='r')
        axarr[0].axvline(np.percentile(temp, 50), ls='--', color='r')

        axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
        axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b')
        axarr[1].set_xlim(wlmaskmin_bhb,wlmaskmax_bhb)
        axarr[1].set_ylim(-0.5,1.5)
        axarr[1].set_title(object+'+'+rvstar[jj])

    else:
        fig, axarr = plt.subplots(1, 4, figsize=(15,6))
        axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean - 5*rv_std, rv_mean + 5*rv_std])
        axarr[0].set_title('RV Histogram', fontsize=16)
        axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3))
        axarr[0].set_xlabel('RV')
        axarr[0].set_xlim(rv_mean - 5*rv_std, rv_mean + 5*rv_std)
        #axarr[0].axvline(np.percentile(temp, 16), ls='--', color='r')
        #axarr[0].axvline(np.percentile(temp, 84), ls='--', color='r')
        axarr[0].axvline(np.percentile(temp, 50), ls='--', color='r')

        axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
        axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b')
        axarr[1].set_xlim(CaT1min*(1+rv_mean/c),CaT1max*(1+rv_mean/c))
        axarr[1].set_ylim(-0.5,1.5)
        axarr[1].xaxis.set_major_locator(plt.MultipleLocator(10))
        axarr[1].axvline(8498.03*(1+rv_mean/c), ls='--', color='r')

        axarr[2].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
        axarr[2].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b')
        axarr[2].set_xlim(CaT2min*(1+rv_mean/c),CaT2max*(1+rv_mean/c))
        axarr[2].set_ylim(-0.5,1.5)
        axarr[2].xaxis.set_major_locator(plt.MultipleLocator(10))
        axarr[2].axvline(8542.09*(1+rv_mean/c), ls='--', color='r')
        axarr[2].set_title(object+'+'+rvstar[jj])
        axarr[2].set_xlabel('Wavelength')

        axarr[3].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
        axarr[3].plot(wl[wlmask]*(1+rv_mean/c), rvspec[jj][wlmask], 'b')
        axarr[3].set_xlim(CaT3min*(1+rv_mean/c),CaT3max*(1+rv_mean/c))
        axarr[3].set_ylim(-0.5,1.5)
        axarr[3].axvline(8662.14*(1+rv_mean/c), ls='--', color='r')
        axarr[3].xaxis.set_major_locator(plt.MultipleLocator(10))

    if savervplot:
        plt.savefig(figdir+str(object)+'_rv.png')
    if showrvplot:
        plt.show()

    return rv_mean, rv_std, chi2rv[jj], snr, rvstar[jj]


def get_telluric_corr(wl, spec, dspec, rvwl, rvspec, object, rv):
    ndim=1
    nwalkers=20
    rvmin = -10
    rvmax = 10
    p0=np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    p0= p0 * rvmax * 2 - rvmax
    
    wl0 = wl
    spec0 = spec
    dspec0 = dspec
    rvspec0 = np.interp(wl0, rvwl, rvspec)
    
    spec = spec[(wl > 7530)  & (wl < 7720)]
    dspec = dspec[(wl > 7530)  & (wl < 7720)]
    wl = wl[(wl > 7530)  & (wl < 7720)]

    # MCMC needs some time to produce reasonable "d" from the likelihood, which is called the "burn-in" period.
    # Adjusting the "burn-in" period is quite empirical.
    nburn=50

    if cubic:
        p = CubicSpline(rvwl, rvspec)
        rvspec = p(wl)
    else:
        rvspec = np.interp(wl, rvwl, rvspec)
    #spec = np.interp(rvwl, wl, spec)
    #dspec = np.interp(rvwl, wl, dspec)

    #wlmask = (rvwl > 7550)  & (rvwl < 7700)
    wlmask = (wl > 7550)  & (wl < 7700)
    
    snr = np.median(spec[wlmask]/dspec[wlmask])
    print('SNR = '+str(snr))
    
    #MCMC
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_post, args=(rvmin, rvmax, wlmask, rvwl, rvspec, spec, dspec), a=0.01)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lp_post, args=(rvmin, rvmax, wlmask, wl, rvspec, spec, dspec), a=0.01)

    pos, prob, state = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    sampler.run_mcmc(pos, nsam)

    rvdist = sampler.flatchain[:,0]
    temp = rvdist
    temp_mean = np.nanmedian(temp)
    if np.std(temp) > 10 : temp = temp[(temp < temp_mean + 70) & (temp > temp_mean -70)]
    temp = stats.sigmaclip(temp,low=5, high=5)[0]
    rv_mean = np.nanmedian(temp)
    #rv_std = np.std(temp)
    rv_std = 0.5 * (np.percentile(temp, 84) - np.percentile(temp, 16))

    #masklen=len(rvwl[wlmask])
    masklen = len(wl[wlmask])
    chi2rv = chi2cal(rv_mean, wlmask, wl, rvspec, spec, dspec) / masklen


    if True:
        # Plot the result.
        fig, axarr = plt.subplots(1, 4, figsize=(15,6))
        axarr[0].hist(temp, 100, color="k", histtype="step", range = [rv_mean - 5*rv_std, rv_mean + 5*rv_std])
        axarr[0].set_title('aband RV Histogram', fontsize=16)
        axarr[0].xaxis.set_major_locator(plt.MultipleLocator(rv_std*3))
        axarr[0].set_xlabel('RV')
        axarr[0].set_xlim(rv_mean - 5*rv_std, rv_mean + 5*rv_std)
        #axarr[0].axvline(np.percentile(temp, 16), ls='--', color='r')
        #axarr[0].axvline(np.percentile(temp, 84), ls='--', color='r')
        axarr[0].axvline(np.percentile(temp, 50), ls='--', color='b')

        axarr[1].plot(wl[wlmask], spec[wlmask], 'lime',lw=0.5)
        axarr[1].plot(wl[wlmask]*(1+rv_mean/c), rvspec[wlmask], 'b')
        axarr[1].set_xlim(7550,7700)
        axarr[1].set_ylim(-0.5,1.5)
        axarr[1].set_title(str(object)+' telluric', fontsize=16)
        axarr[1].set_xlabel('Wavelength')
        axarr[1].xaxis.set_major_locator(plt.MultipleLocator(40))

        axarr[2].plot(wl0, spec0, 'lime',lw=0.5)
        axarr[2].plot(wl0*(1+rv_mean/c), rvspec0, 'b')
        axarr[2].axvline(8183.25*(1+rv/c), ls='--', color='r')
        axarr[2].axvline(8194.79*(1+rv/c), ls='--', color='r')
        axarr[2].set_xlim(8160,8220)
        axarr[2].set_ylim(-0.5,1.5)
        axarr[2].xaxis.set_major_locator(plt.MultipleLocator(30))
        axarr[2].set_title('Na I at 8200A')
        
        axarr[3].plot(wl0, spec0, 'lime',lw=0.5)
        axarr[3].plot(wl0*(1+rv_mean/c), rvspec0, 'b')
        axarr[3].axvline(8806.8*(1+rv/c), ls='--', color='r')
        axarr[3].set_xlim(8790*(1+rv/c),8820*(1+rv/c))
        axarr[3].set_ylim(-0.5,1.5)
        axarr[3].xaxis.set_major_locator(plt.MultipleLocator(30))
        axarr[3].set_title('Mg I at 8807A')



        if savervplot:
            plt.savefig(figdir+str(object)+'_tell.png')
        if showrvplot:
            plt.show()

    return rv_mean, rv_std, chi2rv, snr


def Flin(x,p):

    #P[0] = CONTINUUM LEVEL
    #P[1] = GAUSSIAN HEIGHT/DEPTH FOR MIDDLE CAT LINE
    #P[2] = LINE POSITION
    #P[3] = GAUSSIAN WIDTH
    #P[4] = LORENTZIAN HEIGHT/DEPTH FOR MIDDLE CAT LINE
    #P[5] = LORENTZIAN WIDTH
    #P[6] = GAUSSIAN HEIGHT/DEPTH FOR 8498 CAT LINE
    #P[7] = GAUSSIAN HEIGHT/DEPTH FOR 8662 CAT LINE
    #P[8] = LORENTZIAN HEIGHT/DEPTH FOR 8498 CAT LINE
    #P[9] = LORENTZIAN HEIGHT/DEPTH FOR 8662 CAT LINE

    gauss = p[1]*np.exp(-0.5*((x-p[2])/p[3])**2)+ \
            p[6]*np.exp(-0.5*( (x-p[2]*0.994841)/p[3] )**2) + \
            p[7]*np.exp(-0.5*( (x-p[2]*1.01405)/p[3] )**2)
    lorentz = p[4]*p[5]/( (x-p[2])**2 + (p[5]/2.)**2 ) + \
              p[8]*p[5]/( (x-p[2]*0.994841)**2 + (p[5]/2.)**2 ) + \
              p[9]*p[5]/( (x-p[2]*1.01405)**2 + (p[5]/2.)**2 )

    return p[0] * (1 + gauss + lorentz)
    
'''
# if the relative ratio between three lines are fixed, then use this Flin(x,p)
def Flin(x,p):

    gauss = p[1]*np.exp(-0.5*((x-p[2])/p[3])**2)+ \
            0.6 * p[1]*np.exp(-0.5*( (x-p[2]*0.994841)/p[3] )**2) + \
            0.9 * p[1]*np.exp(-0.5*( (x-p[2]*1.01405)/p[3] )**2)
    lorentz = p[4]*p[5]/( (x-p[2])**2 + (p[5]/2.)**2 ) + \
              0.6 * p[4]*p[5]/( (x-p[2]*0.994841)**2 + (p[5]/2.)**2 ) + \
              0.9 * p[4]*p[5]/( (x-p[2]*1.01405)**2 + (p[5]/2.)**2 )

    return p[0] + gauss + lorentz
'''

def myfunctlin(p, fjac=None, x=None, y=None, err=None):
    model = Flin(x, p)
    status = 0
    #return [status, (y-model)**2/(2*err**2)]
    return [status, ((y-model)/err)]

def get_ew(object, wl, spec, dspec, rv, gaussianonly = 0):
    
    fitstart = (np.abs(wl-8400)).argmin()
    fitend = (np.abs(wl-8700)).argmin()
    spec = spec[fitstart:fitend]
    dspec = dspec[fitstart:fitend]
    wl = wl[fitstart:fitend]
    spec,dspec = normalize_spec(wl, spec, dspec)
    

    fitstart = (np.abs(wl-8483*(1+rv/c))).argmin()
    fitend = (np.abs(wl-8677*(1+rv/c))).argmin()
    
    #orignal
    #fitstart = (np.abs(w-8484)).argmin()
    #fitend = (np.abs(w-8682)).argmin()
    
    contstart = (np.abs(wl-8563*(1+rv/c))).argmin()
    contend = (np.abs(wl-8577*(1+rv/c))).argmin()

    peakfindstart = (np.abs(wl-8538.09*(1+rv/c))).argmin()
    peakfindend = (np.abs(wl-8546.09*(1+rv/c))).argmin()

    sn = np.median(spec[fitstart:fitend]/dspec[fitstart:fitend])
    

    if sn < -7:
        guassianonly = 1
    else:
        gaussianonly = 0

    contlevel = np.nanmedian(spec[contstart:contend][spec[contstart:contend]>0])
    if np.isnan(contlevel) :
        contstart = (np.abs(wl-8590*(1+rv/c))).argmin()
        contend = (np.abs(wl-8610*(1+rv/c))).argmin()
        contlevel = np.nanmedian(spec[contstart:contend][spec[contstart:contend]>0])

    spec = spec / contlevel
    dspec = dspec / contlevel

    smoothspec = ndimage.filters.uniform_filter(spec,size=5)

    linepos = smoothspec[peakfindstart:peakfindend].argmin()
    depth = min(smoothspec[peakfindstart:peakfindend]) - np.median(spec[fitstart:fitend])

    initial_guesses = np.zeros(10)
    param_control = [{'fixed':0, 'limited':[0,0], 'limits':[0.,0.]} for i in range(10)]

    initial_guesses[0] = np.median(spec[contstart:contend])
    initial_guesses[1] = 0.5*depth
    #initial_guesses[2] = (wl[fitstart:fitend])[linepos+peakfindstart-fitstart]
    initial_guesses[2] = 8542.09*(1+rv/c)  # changed this to fix the chip gap issue
    initial_guesses[3] = 1.0
    initial_guesses[4] = 0.3*depth
    initial_guesses[5] = 1.0
    initial_guesses[6] = 0.25*depth
    initial_guesses[7] = 0.4*depth
    initial_guesses[8] = 0.15*depth
    initial_guesses[9] = 0.24*depth


    param_control[1]['limited'][1] = 1
    param_control[1]['limits'][1] = 0.
    param_control[1]['limited'][0] = 1
    param_control[1]['limits'][0] = -1.
    param_control[4]['limited'][1] = 1
    param_control[4]['limits'][1]  = 0.
    param_control[4]['limited'][0] = 1
    param_control[4]['limits'][0] = -1.
    param_control[6]['limited'][1] = 1
    param_control[6]['limits'][1]  = 0.
    param_control[6]['limited'][0] = 1
    param_control[6]['limits'][0] = -1.
    param_control[7]['limited'][1] = 1
    param_control[7]['limits'][1]  = 0.
    param_control[7]['limited'][0] = 1
    param_control[7]['limits'][0] = -1.
    param_control[8]['limited'][1] = 1
    param_control[8]['limits'][1]  = 0.
    param_control[8]['limited'][0] = 1
    param_control[8]['limits'][0] = -1.
    param_control[9]['limited'][1] = 1
    param_control[9]['limits'][1]  = 0.
    param_control[9]['limited'][0] = 1
    param_control[9]['limits'][0] = -1.


    #FORCE LINE WIDTHS TO BE AT LEAST 1 RESOLUTION ELEMENT (0.8AA??) AND LESS THAN 300 KM/S
    param_control[3]['limited'][0] = 1
    param_control[3]['limits'][0] = 0.2 #0.8/2.35
    param_control[3]['limited'][1] = 1
    param_control[3]['limits'][1] = 3.63
    param_control[5]['limited'][0] = 1
    param_control[5]['limits'][0] = 0.2 #0.8
    param_control[5]['limited'][1] = 1
    param_control[5]['limits'][1] = 3.63

    if gaussianonly:
        initial_guesses[4] = 0.
        initial_guesses[5] = 1.0
        param_control[4]['fixed'] = 1
        param_control[5]['fixed'] = 1
    fa = {'x':wl[fitstart:fitend], 'y': spec[fitstart:fitend], 'err':dspec[fitstart:fitend]}

    try:
        m = mpfit(myfunctlin, initial_guesses, functkw=fa, quiet=1, parinfo=param_control, xtol = 1.0e-15)
    except ValueError:
        print("Oops!  Something wrong.")

    modelgl = Flin(wl[fitstart:fitend], m.params)
    covargl = m.covar
    errmsg = m.errmsg
    status = m.status
    print('iter', m.niter)
    if m.niter <= 2:
        print('MPFIT STUCK! RESULTS MAY BE WRONG')
    niter = m.niter
    perrorgl = m.perror
    chisqgl = sum((spec[fitstart:fitend]-modelgl)**2/dspec[fitstart:fitend]**2)
    outparams = m.params
    lineparams=outparams

    if perrorgl is None:
            perrorgl = np.zeros(10) + 1
            covargl = np.zeros([10, 10]) + 1
            niter = 2

    if dispara:
        print('  CHI-SQUARE = %10.1f' %chisqgl)
        print('  DOF = %10.1f' %(fitend-fitstart+1-len(outparams)))
        print('  P(0) = %7.3f +/- %7.3f' %(outparams[0],perrorgl[0]))
        print('  P(1) = %7.3f +/- %7.3f' %(outparams[1],perrorgl[1]))
        print('  P(2) = %10.4f +/- %10.4f' %(outparams[2],perrorgl[2]))
        print('  P(3) = %7.3f +/- %7.3f' %(outparams[3],perrorgl[3]))
        print('  P(4) = %7.3f +/- %7.3f' %(outparams[4],perrorgl[4]))
        print('  P(5) = %7.3f +/- %7.3f' %(outparams[5],perrorgl[5]))
        print('  P(6) = %7.3f +/- %7.3f' %(outparams[6],perrorgl[6]))
        print('  P(7) = %7.3f +/- %7.3f' %(outparams[7],perrorgl[7]))
        print('  P(8) = %7.3f +/- %7.3f' %(outparams[8],perrorgl[8]))
        print('  P(9) = %7.3f +/- %7.3f' %(outparams[9],perrorgl[9]))


    if True:
        # Plot the result.
        fig, axarr = plt.subplots(1, 3, figsize=(17,6))

        axarr[0].plot(wl[fitstart:fitend],spec[fitstart:fitend],c='lime')
        axarr[0].plot(wl[fitstart:fitend],spec[fitstart:fitend]-modelgl,c='y')
        axarr[0].plot(wl[fitstart:fitend],modelgl,lw=2,c='k')
        axarr[0].set_xlim(CaT1min*(1+rv/c),CaT1max*(1+rv/c))
        axarr[0].set_ylim(-0.5,1.5)
        axarr[0].xaxis.set_major_locator(plt.MultipleLocator(10))

        axarr[1].plot(wl[fitstart:fitend],spec[fitstart:fitend],c='lime')
        axarr[1].plot(wl[fitstart:fitend],spec[fitstart:fitend]-modelgl,c='y')

        axarr[1].plot(wl[fitstart:fitend],modelgl,lw=2,c='k')
        axarr[1].set_xlim(CaT2min*(1+rv/c),CaT2max*(1+rv/c))
        axarr[1].set_ylim(-0.5,1.5)
        axarr[1].xaxis.set_major_locator(plt.MultipleLocator(10))
        axarr[1].set_title(str(object), fontsize=16)
        axarr[1].set_xlabel('Wavelength')

        axarr[2].plot(wl[fitstart:fitend],spec[fitstart:fitend],c='lime')
        axarr[2].plot(wl[fitstart:fitend],spec[fitstart:fitend]-modelgl,c='y')
        axarr[2].plot(wl[fitstart:fitend],modelgl,lw=2,c='k')
        axarr[2].set_xlim(CaT3min*(1+rv/c),CaT3max*(1+rv/c))
        axarr[2].set_ylim(-0.5,1.5)
        axarr[2].xaxis.set_major_locator(plt.MultipleLocator(10))

        if saveewplot:
            plt.savefig(figdir+str(object)+'_ew.png')
        if showewplot:
            plt.show()
        #plt.close()

    gaussian_integral = outparams[1] * outparams[3] * np.sqrt(2*np.pi)
    dgaussian_integral = np.sqrt((outparams[1] * perrorgl[3] * np.sqrt(2*np.pi))**2 + \
                               (perrorgl[1] * outparams[3] * np.sqrt(2*np.pi))**2)

    lorentzian_integral = 2*np.pi*outparams[4]
    dlorentzian_integral = 2*np.pi*perrorgl[4]
    ew2_fit = gaussian_integral + lorentzian_integral
    dew2_fit = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2)
    dew2_fit_covar = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2 + \
                          2*np.pi*outparams[1]*outparams[3]*covargl[1,3]*perrorgl[1]*perrorgl[3] + \
                          (2*np.pi)**1.5*outparams[3]*covargl[1,4]*perrorgl[1]*perrorgl[4] + \
                          (2*np.pi)**1.5*outparams[1]*covargl[3,4]*perrorgl[3]*perrorgl[4])

    v2 = (outparams[2] - 8542.09)/8542.09*c
    dv2 = perrorgl[2]/8542.09*c
    if dispara:
        print('V_CaT2: %10.3f +/- %10.3f' %(v2,dv2))
        print('CaT2 (fit): %10.3f +/- %10.3f' %(ew2_fit,dew2_fit))
        print('CaT2 (fit, covar) %10.3f +/- %10.3f' %(ew2_fit,dew2_fit_covar))


    gaussian_integral = outparams[6] * outparams[3] * np.sqrt(2*np.pi)
    dgaussian_integral = np.sqrt( (outparams[6] * perrorgl[3] * np.sqrt(2*np.pi))**2 + \
                               (perrorgl[6] * outparams[3] * np.sqrt(2*np.pi))**2 )

    lorentzian_integral = 2*np.pi*outparams[8]
    dlorentzian_integral = 2*np.pi*perrorgl[8]
    ew1_fit = gaussian_integral + lorentzian_integral
    dew1_fit = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2)
    dew1_fit_covar = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2 + \
                          2*np.pi*outparams[6]*outparams[3]*covargl[6,3]*perrorgl[6]*perrorgl[3] + \
                          (2*np.pi)**1.5*outparams[3]*covargl[6,8]*perrorgl[6]*perrorgl[8] + \
                          (2*np.pi)**1.5*outparams[6]*covargl[3,8]*perrorgl[3]*perrorgl[8])

    if dispara:
        print('CaT1 (fit): %10.3f +/- %10.3f' %(ew1_fit,dew1_fit))
        print('CaT1 (fit, covar) %10.3f +/- %10.3f' %(ew1_fit,dew1_fit_covar))
        print('CaT1 (fit): %10.3f +/- %10.3f' %(0.6 * ew2_fit, 0.6 * dew2_fit))

    gaussian_integral = outparams[7] * outparams[3] * np.sqrt(2*np.pi)
    dgaussian_integral = np.sqrt( (outparams[7] * perrorgl[3] * np.sqrt(2*np.pi))**2 + \
                               (perrorgl[7] * outparams[3] * np.sqrt(2*np.pi))**2 )

    lorentzian_integral = 2*np.pi*outparams[9]
    dlorentzian_integral = 2*np.pi*perrorgl[9]
    ew3_fit = gaussian_integral + lorentzian_integral
    dew3_fit = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2)
    dew3_fit_covar = np.sqrt(dgaussian_integral**2 + dlorentzian_integral**2 + \
                          2*np.pi*outparams[7]*outparams[3]*covargl[7,3]*perrorgl[7]*perrorgl[3] + \
                          (2*np.pi)**1.5*outparams[3]*covargl[7,9]*perrorgl[7]*perrorgl[9] + \
                          (2*np.pi)**1.5*outparams[7]*covargl[3,9]*perrorgl[3]*perrorgl[9])

    if dispara:
        print('CaT3 (fit): %10.3f +/- %10.3f' %(ew3_fit,dew3_fit))
        print('CaT3 (fit, covar) %10.3f +/- %10.3f' %(ew3_fit,dew3_fit_covar))
        print('CaT3 (fit): %10.3f +/- %10.3f' %(0.9 * ew2_fit, 0.9 * dew2_fit))

    ews = ew1_fit+ew2_fit+ew3_fit
    dews = np.sqrt(dew1_fit**2+dew2_fit**2+dew3_fit**2)
    vcat = v2

    #ews = ew2_fit * (0.6+1.+0.9)
    #dews = dew2_fit * (0.6+1.+0.9)
    return -ew1_fit, dew1_fit, -ew2_fit, dew2_fit, -ew3_fit, dew3_fit, -ews, dews, vcat, niter

def helio2gsr(vhelio, l, b):
    usol=11.1
    vsol=12.24
    wsol=7.25
    theta=220.0
    vcirc=vsol+theta
    vgsr = vhelio+usol*np.cos(b*np.pi/180.)*np.cos(l*np.pi/180.)+vcirc*np.cos(b*np.pi/180.)*np.sin(l*np.pi/180.)+wsol*np.sin(b*np.pi/180)
    return vgsr


if __name__ == "__main__":

    rvwl1, rvspec1 = read_rv_stds(rv_fname, 0)
    rvwl2, rvspec2 = read_rv_stds(rv_fname, 1)
    rvwl3, rvspec3 = read_rv_stds(rv_fname, 8)

    if not (all(rvwl1 == rvwl2) and all(rvwl2 == rvwl3)):
        print("SOMETHING WRONG WITH STELLAR TEMPLATES")

    rvspec1[rvspec1 == 0] = 1
    rvspec2[rvspec2 == 0] = 1
    rvspec3[rvspec3 == 0] = 1

    if bhb and single:
        rvstar = np.array(['HD161817'])
    else:
        rvstar = np.array(['HD122563', 'HD26297', 'HD161817'])
        
    num = len(rvwl1)
    nstars = len(rvstar)

    rvwl = np.zeros([nstars, num])
    rvspec = np.zeros([nstars, num])

    if bhb and single:
        rvwl[0] = rvwl3
        rvspec[0] = rvspec3
    
    else:
        rvwl[0] = rvwl1
        rvspec[0] = rvspec1
    
        rvwl[1] = rvwl2
        rvspec[1] = rvspec2

        rvwl[2] = rvwl3
        rvspec[2] = rvspec3
    
    telluwl, telluspec = read_tell_stds(telluric_fname)

    objlist = os.listdir(objdir)
    objlist = np.sort(objlist)
    k = 0
    
    if savedata:
        f = open(outputfile,'a')
        f.write('#INDEX OBJECT    SNR    V    dV   template  chi2rv  zq1  abandSNR   aband  daband   chi2aband  zq2   EW1   dEW1   EW2  dEW2  EW3  dEW3    EW     dEW  VCaT  zq3  niter\n')
        f.close()
    if single:
        object_fname = object_fname_single
        
        if uniform_resample:
            wl, spec, dspec = combine_imacs_spec_resample_uniform(object_fname, nbuff=nbuff)
        else:
            wl, spec, dspec = combine_imacs_spec(object_fname, nbuff = nbuff)
        
        if not normalizeb4combine:
            spec, dspec = normalize_spec(wl, spec, dspec)
        
        hdr = pyfits.open(object_fname)[5].header
        object = hdr['OBJNO']
        print('OBJECT ID = %s'%object)

        rv, rverr, chi2, snr, template = get_rv(wl,spec,dspec,rvwl,rvspec,object, rvstar)
        print('RV = %8.3f +/- %5.3f' %(rv, rverr))
        print('chi-square = %5.2f' %chi2)

        #abandmask = (wl < 8300) & (wl > 7000)
        #abandv, abandverr, abandchi2, abandsnr = get_telluric_corr(wl[abandmask], spec[abandmask], dspec[abandmask],telluwl, telluspec, object, rv)
        abandv, abandverr, abandchi2, abandsnr = get_telluric_corr(wl, spec, dspec,telluwl, telluspec, object, rv)
                
        print('ABAND_V = %8.3f +/- %5.3f' %(abandv, abandverr))
        print('ABAND_chi-square = %5.2f' %abandchi2)
        ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, niter = get_ew(object, wl, spec, dspec, rv, 0)
        print('EW = %8.2f +/- %5.2f' %(ews, dews))
        print('V_CaT = %8.2f'%vcat)


    else:
        for ii in range(0, len(objlist)):
            if objlist[ii][-5:] == '.fits':
                fname = objlist[ii]
                object_fname = objdir + fname
                snr = get_snr(object_fname)

                if snr < snr_min:
                    continue

                if uniform_resample:
                    wl, spec, dspec = combine_imacs_spec_resample_uniform(object_fname, nbuff=nbuff)
                else:
                    wl, spec, dspec = combine_imacs_spec(object_fname, nbuff = nbuff)

                if not normalizeb4combine:
                    spec, dspec = normalize_spec(wl, spec, dspec)

                hdr = pyfits.open(object_fname)[5].header
                object = hdr['OBJNO']
                print('OBJECT ID = %s'%object)
                rv, rverr, chi2, snr, template = get_rv(wl,spec,dspec,rvwl,rvspec,object, rvstar)

                print('RV = %8.3f +/- %5.3f' %(rv, rverr))
                print('chi-square = %5.2f' %chi2)

                if zquality:
                    while True:
                        temp = input('quality (0 or 1) --> ')
                        if temp == '0' or temp == '1':
                            zq1 = int(temp)
                            break
                else:
                    zq1 = -1

                #abandmask = (wl < 8300) & (wl > 7000)
                #abandv, abandverr, abandchi2, abandsnr = get_telluric_corr(wl[abandmask], spec[abandmask], dspec[abandmask], telluwl, telluspec, object, rv)
                abandv, abandverr, abandchi2, abandsnr = get_telluric_corr(wl, spec, dspec,telluwl, telluspec, object, rv)
                print('ABAND_V = %8.3f +/- %5.3f' %(abandv, abandverr))
                print('ABAND_chi-square = %5.2f' %abandchi2)

                if zquality:
                    while True:
                        temp = input('quality (0 or 1) --> ')
                        if temp == '0' or temp == '1':
                            zq2 = int(temp)
                            break
                else:
                    zq2 = -1

                ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, niter = get_ew(object, wl, spec, dspec, rv, 0)
                print('EW = %8.2f +/- %5.2f' %(ews, dews))
                print('V_CaT = %8.2f'%vcat)

                if zquality:
                    while True:
                        temp = input('quality (0 or 1) --> ')
                        if temp == '0' or temp == '1':
                            zq3 = int(temp)
                            break
                else:
                    zq3 = -1


                if savedata:
                    f = open(outputfile, 'a')
                    f.write('%2d %22s %7.2f %7.2f %5.2f %10s %5.2f %3i %7.2f %7.2f %5.2f %5.2f %3i %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %7.2f %3i %3i \n'\
                            %(ii, object, snr, rv, rverr, template, chi2, zq1, abandsnr, abandv, abandverr, abandchi2, zq2,  ew1, dew1, ew2, dew2, ew3, dew3, ews, dews, vcat, zq3, niter))
                    f.close()



