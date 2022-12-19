import numpy as np
import alphashape
from descartes import PolygonPatch
from astropy.io import fits
import matplotlib.pyplot as plt
# from localreg import * # If this is uncommented logging will not work
from astropy.io import fits
import os 


# AFS implementation from Xu et al. 2019

def AFS_continuum_norm_1order(wv, spec, sigma, wv_to, spec_to, sigma_to, q=0.95, plot=False):

    # Step zero: set some initial values
    alpha = (1./6.) * (np.max(wv) - np.min(wv))

    # Step 1: rescale scaled flux vector by u ----------------------
    u = (np.max(wv) - np.min(wv))/(10.*np.max(spec))
    spec *= u
    spec_to *= u
    sigma_to *= u
    #alpha = 2. * u

    # Step 2: calculate alpha shape ------------------------------
    spec_stack = np.transpose(np.vstack((wv, spec)))
    spec_stack_tuple = tuple(map(tuple, spec_stack))
    alpha_shape = alphashape.alphashape(spec_stack_tuple, alpha)

    # Step 3: Extract vertices that correspond to ----------------
    #         y-maxima in the alpha shape.
    #         Carry out local polynomial regression.

    try:
        x, y = alpha_shape.exterior.coords.xy
    except:
        # If we have a multipolygon
        polygon_points = [point for polygon in alpha_shape for point in polygon.exterior.coords[:-1]]
        x, y = list(zip(*polygon_points))[0], list(zip(*polygon_points))[1]


    AS_tilde = get_AS_tilde(wv, x, y)
    x_AS_tilde, y_AS_tilde = np.transpose(AS_tilde)[0], np.transpose(AS_tilde)[1]

    # Normalize x-values for weighting
    x_AS_tilde_norm = (x_AS_tilde[1] - np.min(x_AS_tilde))/(np.max(x_AS_tilde) - np.min(x_AS_tilde))
    #B1 = localreg(x_AS_tilde, y_AS_tilde, degree=2, kernel=rbf.tricube, width=20)
    B1 = local_polynomial_regr(x_AS_tilde, y_AS_tilde, x_AS_tilde)

    # Make sure there are no NaNs; set equal to continuum if they are found
    #B1 = np.nan_to_num(B1, nan=1.0)

    # Divide out continuum to get initial guess
    y1 = spec/B1

    # Step 4: Identify non-absorption points
    overlap_spec_indices = np.in1d(np.transpose(AS_tilde)[1], spec).nonzero()
    W_alpha = np.transpose(np.array([wv[overlap_spec_indices], spec[overlap_spec_indices]]))

    if plot == True:
        plot_alpha_shape_fit(wv, spec, alpha_shape, AS_tilde, W_alpha)

    # Step 5: Carry out second local polynomial regression -------
    S_alpha = get_S_alpha(y1, W_alpha, wv, spec, q) # q is the quantile
    x_S_alpha, y_S_alpha = np.transpose(S_alpha)[0], np.transpose(S_alpha)[1]

    # Normalize x-values for weighting
    x_S_alpha_norm = (x_S_alpha - np.min(x_S_alpha))/(np.max(x_S_alpha) - np.min(x_S_alpha))
    wv_all_norm = (wv - np.min(x_S_alpha))/(np.max(x_S_alpha) - np.min(x_S_alpha))
    #B2 = localreg(x_S_alpha, y_S_alpha, x0=wv, degree=2, kernel=rbf.tricube, width=20)
    B2 = local_polynomial_regr(x_S_alpha, y_S_alpha, wv_to)

    # Make sure there are no NaNs; set equal to continuum if they are found
    #B2 = np.nan_to_num(B2, nan=1.0)

    # Step 6: divide by B2 to get final blaze-removed spectrum ---
    y2 = spec_to/B2

    # divide sigma too
    y2_sig = sigma_to/B2

    # rename so everything is clearer
    spec_norm, sigma_norm, cfit = y2, y2_sig, B2

    return spec_norm, sigma_norm, cfit


def AFS_continuum_norm_1star(wv_1star, spec_1star, sigma_1star):

    num_pix_1order = len(wv_1star[0])

    spec_norm_1star = np.zeros(num_pix_1order)
    sigma_norm_1star = np.zeros(num_pix_1order)
    cfit_1star = np.zeros(num_pix_1order)

    for order in range(0, len(wv_1star)):
        wv_to, spec_to, sigma_to = wv_1star[order], spec_1star[order], sigma_1star[order]

        # Copy to avoid overwriting original values
        wv1, spec1, sigma1 = wv_1star[order], spec_1star[order], sigma_1star[order]

        wv_vetted, spec_vetted, sigma_vetted = remove_cosmic_rays_1order(wv1, spec1, sigma1)
        spec_norm, sigma_norm, cfit = AFS_continuum_norm_1order(wv_vetted, spec_vetted, \
                                                                sigma_vetted, wv_to, spec_to, sigma_to)

        spec_norm_1star = np.vstack((spec_norm_1star, spec_norm))
        sigma_norm_1star = np.vstack((sigma_norm_1star, sigma_norm))
        cfit_1star = np.vstack((cfit_1star, cfit))

    spec_norm_1star = spec_norm_1star[1:]
    sigma_norm_1star = sigma_norm_1star[1:]
    cfit_1star = cfit_1star[1:]

    return spec_norm_1star, sigma_norm_1star, cfit_1star


def remove_cosmic_rays_1order(wv1, spec1, sigma1, qs=0.98, qlim=0.99):

    Delta_L = np.diff(spec1)
    Q_qs = np.quantile(Delta_L, qs)
    Q_j_minus_1 = np.quantile(Delta_L, qlim)
    while Q_qs < Q_j_minus_1:

        keep_pix = np.where(abs(Delta_L) <= Q_j_minus_1)[0] + 1
        wv1 = wv1[keep_pix]
        spec1 = spec1[keep_pix]
        sigma1 = sigma1[keep_pix]

        Delta_L = np.diff(spec1)
        Q_j = np.quantile(Delta_L, qlim)
        Q_j_minus_1 = Q_j

    #print('%i pixels kept after CR masking' %(len(wv1)))
    return wv1, spec1, sigma1


def get_AS_tilde(wv, x, y):

    # extract only y vertices that correspond to the maximum y in the alpha shape

    AS_tilde = np.zeros(2)
    for x_at_point in wv:
        ymax = 0

        # check to find highest point in the alpha shape at that wavelength
        for j in range(0, len(x)-1):
            if x[j] == x_at_point:
                if y[j] > ymax:
                    ymax = y[j]
            elif x[j] < x_at_point:
                if x[j+1] > x_at_point:
                    # get new y value by fitting that line
                    m, b = np.polyfit([x[j], x[j+1]], [y[j], y[j+1]], 1)
                    y_fit = (m*x_at_point) + b
                    if y_fit > ymax:
                        ymax = y_fit
                    else:
                        pass
        AS_tilde = np.vstack((AS_tilde, np.array([x_at_point, ymax])))

    # Get rid of initialization zeroes
    AS_tilde = AS_tilde[1:]

    return AS_tilde


def local_polynomial_regr(wv_inp, flux_inp, wv_to, poly_deg=2, m0=0.25):

    # Number of nearby pixels to include
    num_pix = int(len(wv_inp) * m0)

    flux_regr = np.array([])
    for wv_val in wv_to:

        # Subtract off current wavelength before completing fit
        wv_norm = wv_inp - wv_val

        # Pull out the num_pix nearby pixels for use in the fit
        nearby_wv_inds = np.argsort(abs(wv_norm))[:num_pix]
        wv_nearby, flux_nearby = wv_norm[nearby_wv_inds], flux_inp[nearby_wv_inds]

        # Get weights for polynomial fit
        weight_inp = abs(wv_nearby)/np.max(abs(wv_nearby))
        w = K_weight(weight_inp)

        # Complete the fit
        p = np.poly1d(np.polyfit(wv_nearby, flux_nearby, deg=poly_deg, w=w))

        # The intercept is an approximation to the estimate at the given lambda
        flux_regr = np.append(flux_regr, p(0))

    return flux_regr



def K_weight(x):

    return (1 - (x*x*x))**3.


def get_S_alpha(y1, W_alpha, wv, spec, q):

    S_alpha = np.zeros(2)
    for j in range(0, len(W_alpha)-1):

        # find y1 values that fall in the window
        min_wv_window, max_wv_window = W_alpha[j][0], W_alpha[j+1][0]
        wv_inwindow = wv[(wv >= min_wv_window) & (wv <= max_wv_window)]
        spec_inwindow = spec[(wv >= min_wv_window) & (wv <= max_wv_window)]
        y1_inwindow = y1[(wv >= min_wv_window) & (wv <= max_wv_window)]

        # find values above the q quantile in that window
        quant = np.quantile(y1_inwindow, q)
        spec_cont = spec_inwindow[(y1_inwindow > quant)]
        wv_cont = wv_inwindow[(y1_inwindow > quant)]

        points_cont = np.transpose(np.vstack((wv_cont, spec_cont)))
        S_alpha = np.vstack((S_alpha, points_cont))

    S_alpha = np.unique(S_alpha[1:], axis=0) # remove initialization zeroes and duplicates

    return S_alpha


def plot_alpha_shape_fit(wv, spec, alpha_shape, AS_tilde, W_alpha):

    # Handle different potential types of polygons
    lines = []
    boundary = alpha_shape.boundary
    if boundary.type == 'MultiLineString':
        for line in boundary:
            lines.append(line)
    else:
        lines.append(boundary)

    # Plot alpha shape and vertices
    fig, ax = plt.subplots()
    ax.plot(wv, spec)
    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
    for point in lines[0].coords:
      x, y = point
      ax.scatter(x, y, color='k')

    # Plot AS_tilde
    ax.plot(np.transpose(AS_tilde)[0], np.transpose(AS_tilde)[1], color='gray', label=r'$\tilde{AS}$')
    ax.scatter(np.transpose(W_alpha)[0], np.transpose(W_alpha)[1], marker='o', color='red', label=r'$W_{\alpha}$')
    ax.set_xlabel('wavelength (\AA)')
    ax.set_ylabel('normalized flux')
    ax.legend()
    plt.show()


def contfit_alpha_hull(starname, spec_raw, sigma_raw, wv_raw, save_dir, plot=False):

    spec_norm, sigma_norm, cfit = AFS_continuum_norm_1star(wv_raw, spec_raw, sigma_raw)


    if plot == True:
        plt.plot(np.ndarray.flatten(spec_norm), color='purple')
        plt.xlabel('pixel number')
        plt.ylabel('normalized flux')
        plt.tight_layout()

        plt.savefig( os.path.join(save_dir, '%s_specnorm.png' %(starname)), dpi=300)
        plt.clf()
        plt.close()

    # # Save continuum fit
    # np.save(save_dir+'%s_cfit.npy' %(starname), cfit)

    # Save spectrum
    np.save( os.path.join(save_dir, '%s_specnorm.npy' %(starname)), spec_norm)


    # Preprocessing step: Get rid of all large peaks that will create problems for interpolation
    spec_norm_nopeaks = np.zeros(len(spec_norm[0]))
    if plot == True: # If you want to use normalized spectra with no peaks 
        for spec_norm_1order in spec_norm:
            spec_norm_nopeaks = np.vstack((spec_norm_nopeaks, remove_spurious_peaks(spec_norm_1order, threshold=1.2)))
        spec_norm_nopeaks = spec_norm_nopeaks[1:]

        # Save spectrum, no peaks
        np.save( os.path.join(save_dir, '%s_specnorm_nopeaks.npy' %(starname)), spec_norm_nopeaks)

    # Save uncertainty
    np.save( os.path.join(save_dir, '%s_sigmanorm.npy' %(starname)), sigma_norm)
    

    if plot == True:
        plt.plot(np.ndarray.flatten(spec_norm_nopeaks), color='purple')
        plt.xlabel('pixel number')
        plt.ylabel('normalized flux')
        plt.tight_layout()
        plt.savefig(save_dir+'%s_specnorm_nopeaks.png' %(starname), dpi=300)
        plt.clf()
        plt.close()

    return spec_norm, sigma_norm

def remove_spurious_peaks(spec_1order, threshold=1.2):

    # Find pixel peaks and replace with 1 (default continuum value).
    # Leave sigma the same (will be a large value)
    peaks, = np.where(spec_1order > threshold)
    spec_1order[peaks] = 1.

    return spec_1order