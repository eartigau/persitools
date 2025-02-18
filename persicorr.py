import numpy as np  # Importing NumPy for numerical operations
from astropy.io import fits  # Importing astropy.io.fits for handling FITS files
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot for plotting
from scipy.constants import c  # Importing the speed of light constant from scipy.constants
from scipy.interpolate import InterpolatedUnivariateSpline as ius  # Importing InterpolatedUnivariateSpline for interpolation
import bottleneck as bn  # Importing bottleneck for fast array operations
import os  # Importing os for operating system interface
from astropy.table import Table  # Importing Table from astropy.table for handling table data
import argparse  # Importing argparse for command-line argument parsing

def write_t(data, file):
    """
    Write a dictionary of data to a Multi-Extension FITS (MEF) file.

    This function opens a FITS file for reading and writing ('rw' mode), and allows overwriting if the file already exists.
    It iterates over each key in the data dictionary, retrieves the data and its associated header, and writes them
    to the FITS file as new extensions.

    Parameters:
    data (dict): A dictionary containing the data and headers to be written to the FITS file.
    file (str): The path to the FITS file to be written.

    Returns:
    None
    """

    # set primary hdu
    if '_header' not in data:
        data['_header'] = fits.Header()
        hdul0 = fits.PrimaryHDU(header=data['_header'])
    else:
        hdul0 = fits.PrimaryHDU(header=data['_header'])

    hdus = [hdul0]
    for key in data.keys():
        if '_header' in key:
            continue

        data_fits = data[key]
        if key+'_header' not in data:
            header = fits.Header()
        else:
            header = data[key + '_header']
        header['EXTNAME'] = (key,'Name of the extension')
        # find if it is a table
        if isinstance(data_fits, Table):
            hdu = fits.BinTableHDU(data_fits, header=header)#, name=key)
        else:
            hdu = fits.ImageHDU(data_fits, header=header)#, name=key)
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(file, overwrite=True)        

def read_t(file):
    """
    Read a Multi-Extension FITS (MEF) file and create a dictionary containing all the extensions and their headers.

    This function opens a FITS file for reading and iterates over each Header Data Unit (HDU) in the file.
    It reads the data and headers from each HDU and stores them in a dictionary, with the extension names as keys.

    Parameters:
    file (str): The path to the FITS file to be read.

    Returns:
    dict: A dictionary containing the data and headers from all the extensions in the FITS file.
    """
    
    data = dict()

    with fits.open(file) as hdul:
        for hdu in hdul:
            if 'EXTNAME' in hdu.header:
                key = hdu.header['EXTNAME']
                data[key] = hdu.data
                data[key + '_header'] = hdu.header
            else:
                data['_header'] = hdu.header
    
    return data


def mk_wave_grid(wave1, wave2, step=1000):
    """
    Create a 1D wavelength grid with a step of 1 km/s.
    The wavelength spans from wave1 to wave2 nm.

    Parameters:
    wave1 (float): The starting wavelength in nm.
    wave2 (float): The ending wavelength in nm.
    step (float): The step size in km/s.

    Returns:
    np.ndarray: A 1D array of wavelengths.
    """
    
    # Initialize the wavelength grid with the starting wavelength
    wave = [wave1]
    
    # Append wavelengths to the grid until the ending wavelength is reached
    while wave[-1] < wave2:
        wave.append(wave[-1] * (1 + step / c))
    
    # Convert the list to a numpy array
    wave = np.array(wave)

    return wave

def mk1d(wave2d, sp2d, wave1d):
    """
    Create a 1D version of the 2D spectrum data by interpolating onto a 1D wavelength grid.

    Parameters:
    wave2d (np.ndarray): 2D array of wavelengths.
    sp2d (np.ndarray): 2D array of spectrum data.
    wave1d (np.ndarray): 1D array of target wavelengths.

    Returns:
    np.ndarray: A 2D array of interpolated spectrum data.
    """
    
    # Initialize the output array with zeros
    outsp = np.zeros([len(wave1d), sp2d.shape[0]])
    
    # Iterate over each order in the 2D spectrum data
    for iord in range(sp2d.shape[0]):
        sp = sp2d[iord]
        wave = wave2d[iord]
        
        # Identify valid (finite) data points
        valid = np.isfinite(sp)
        
        # Interpolate the spectrum data onto the 1D wavelength grid
        outsp[:, iord] = ius(wave[valid], sp[valid], k=1, ext=1)(wave1d)

    return outsp

def bin100(map2d):
    """
    Bin all pixels in the 2D map along the first axis by 100 pixels.

    Parameters:
    map2d (np.ndarray): 2D array of data to be binned.

    Returns:
    np.ndarray: A 2D array of binned data.
    """
    
    # Initialize the output array with zeros
    map1d = np.zeros([int(map2d.shape[0] / 100), map2d.shape[1]])
    
    # Iterate over each bin and calculate the mean of the pixels in the bin
    for i in range(map1d.shape[0]):
        map1d[i] = bn.nanmean(map2d[i * 100:(i + 1) * 100], axis=0)
    
    return map1d

def correct_persistence(filename, path_to_persifile='persi.fits'):
    """
    Corrects the persistence effect in astronomical data by using a persistence map.

    This function reads the input FITS file and a persistence map, processes the data to correct for persistence,
    and writes the corrected data back to a new FITS file. It also generates a plot comparing the data before and
    after the correction.

    Parameters:
    filename (str): The path to the input FITS file containing the data to be corrected.
    path_to_persifile (str): The path to the persistence map FITS file.

    Returns:
    None
    """
    
    print("\tStarting persistence correction...")

    if not os.path.exists(path_to_persifile):
        print(f"\tError: Persistence file {path_to_persifile} not found.")
        print(f'\tPlease provide the correct path to the persistence file.')
        print(f'\tIf you do not have the persistence file, please download it from')
        print('\twww.astro.umontreal.ca/~artigau/persistence/persi.fits')
        return
    
    print("\tReading persistence map...")
    # Read the persistence map from the FITS file
    persimap = fits.getdata(path_to_persifile)
    
    print("\tNormalizing persistence map...")
    # Normalize each row of the persistence map by its median value
    for i in range(persimap.shape[0]):
        persimap[i] = persimap[i] / np.nanmedian(persimap[i])

    print("\tCreating wavelength grid...")
    # Create a wavelength grid from 950 to 2500 nm with steps of 1000 km/s
    wavegrid = mk_wave_grid(950, 2500, 1000)
    
    # Truncate the wavelength grid to the nearest 100 in length
    wavegrid = wavegrid[:int(len(wavegrid) / 100) * 100]

    # Define the keys for the wave, flux, and blaze data in the FITS file
    wave_key, flux_key, blaze_key = 'WaveAB', 'FluxAB', 'BlazeAB'

    print("\tReading wave, flux, and blaze data from input FITS file...")
    # Read the wave, flux, and blaze data from the input FITS file
    wave = fits.getdata(filename, wave_key)
    flux = fits.getdata(filename, flux_key)
    blaze = fits.getdata(filename, blaze_key)

    print("\tNormalizing blaze data...")
    # Normalize each order of the blaze data by its median value
    for iord in range(blaze.shape[0]):
        blaze[iord] /= np.nanmedian(blaze[iord])

    print("\tCreating 1D versions of wave, flux, persistence map, and blaze data...")
    # Create 1D versions of the wave, flux, persistence map, and blaze data
    flux1d = mk1d(wave, flux, wavegrid)
    persi1d = mk1d(wave, persimap, wavegrid)
    blaze1d = mk1d(wave, blaze, wavegrid)

    print("\tBinning the 1D data by 100 pixels...")
    # Bin the 1D data by 100 pixels along the first axis
    flux1db = bin100(flux1d)
    persi1db = bin100(persi1d)
    blaze1db = bin100(blaze1d)

    print("\tCalculating weights for the binned blaze data...")
    # Calculate the weights for the binned blaze data
    weights1db = bn.nansum(blaze1db, axis=1)

    print("\tInitializing persistence amplitude array...")
    # Initialize the persistence amplitude array
    amps_persi = np.zeros(persimap.shape[0], dtype=float)

    print("\tSolving for persistence amplitudes...")
    # Iteratively solve for the persistence amplitudes
    for ite in range(20):
        # Calculate the spectrum by summing the flux minus the persistence contribution
        with np.errstate(invalid='ignore', divide='ignore'):
            sp1db = bn.nansum(flux1db - persi1db * amps_persi, axis=1)
            # Normalize the spectrum by the weights
            sp1db /= weights1db
        
        # Backproject the spectrum onto a 2D grid
        with np.errstate(invalid='ignore'):
            diff = (flux1db - blaze1db * sp1db.reshape(len(sp1db), 1))
        
        # Update the persistence amplitudes
        amps_persi = np.nansum(diff * persi1db, axis=0) / np.nansum(persi1db ** 2, axis=0)
        
        # Ensure that the persistence amplitudes are non-negative
        amps_persi[amps_persi < 0] = 0

    print("\tCreating final persistence model...")
    # Create the final persistence model
    persi2d = persimap * amps_persi.reshape([49, 1])
    
    print("\tSubtracting persistence model from flux data...")
    # Subtract the persistence model from the flux data
    flux2 = flux - persi2d

    print("\tCreating 1D versions of the original and corrected flux data...")
    # Create 1D versions of the original and corrected flux data
    with np.errstate(invalid='ignore', divide='ignore'):
        flux1_1d = mk1d(wave, flux, wavegrid)
        sp1d = bn.nansum(flux1_1d, axis=1) / bn.nansum(blaze1d, axis=1)

    # Interpolate the 1D spectrum to create a 2D model
    valid = np.isfinite(sp1d)
    spl = ius(wavegrid[valid], sp1d[valid], k=1, ext=1)
    model1_2d = spl(wave) * blaze

    # Create 1D versions of the corrected flux data
    with np.errstate(invalid='ignore', divide='ignore'):
        flux2_1d = mk1d(wave, flux2, wavegrid)
        sp2d = bn.nansum(flux2_1d, axis=1) / bn.nansum(blaze1d, axis=1)

    # Interpolate the 1D corrected spectrum to create a 2D model
    valid = np.isfinite(sp2d)
    spl = ius(wavegrid[valid], sp2d[valid], k=1, ext=1)
    model2_2d = spl(wave) * blaze

    print("\tGenerating plot to compare data before and after persistence correction...")
    # Create a plot to compare the data before and after persistence correction
    fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True, sharey=True)
    for iord in range(blaze.shape[0]):
        if iord == 0:
            label1 = 'Flux'
            label2 = 'Model'
            label3 = 'Persistence'
            label4 = 'Residual'
        else:
            label1 = None
            label2 = None
            label3 = None
            label4 = None

        # Plot the original flux, model, persistence, and residuals
        ax[0].plot(wave[iord], flux[iord], label=label1, alpha=0.5, color='black')
        ax[0].plot(wave[iord], model1_2d[iord], label=label2, alpha=0.5, color='red')
        ax[0].plot(wave[iord], flux[iord] - model1_2d[iord], label=label4, alpha=0.5, color='blue')
        ax[0].plot(wave[iord], persi2d[iord], label=label3, alpha=0.5, color='orange')

        # Plot the corrected flux, model, persistence, and residuals
        ax[1].plot(wave[iord], flux2[iord], label=label1, alpha=0.5, color='black')
        ax[1].plot(wave[iord], model2_2d[iord], label=label2, alpha=0.5, color='red')
        ax[1].plot(wave[iord], persi2d[iord], label=label3, alpha=0.5, color='orange')
        ax[1].plot(wave[iord], flux2[iord] - model2_2d[iord], label=label4, alpha=0.5, color='blue')

    # Set plot titles and labels
    ax[0].set_title('Before Persicorr')
    ax[1].set_title('After Persicorr')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel('Flux')
    ax[1].set_ylabel('Flux')
    ax[1].set_xlabel('Wavelength (nm)')

    # Set the x-axis limits
    xlim = [1600, 1750]
    ax[1].set(xlim=xlim)

    # Set the y-axis limits based on the data range
    ylim_wave_range = (wave > xlim[0]) & (wave < xlim[1])
    ylimmax = np.nanpercentile(flux2[ylim_wave_range], 99) * 1.1
    ylimmin = np.nanpercentile((flux - model1_2d)[ylim_wave_range], 1)
    ax[1].set(ylim=[ylimmin, ylimmax])

    print("\tSaving plot to PNG file...")
    # Save the plot to a PNG file
    plt.savefig(filename.replace('.fits', '_persicorr.png'))
    plt.close()

    print("\tCreating new FITS file with corrected flux data...")
    # Create a new FITS file with the corrected flux data
    outname = filename.replace('.fits', '_persicorr.fits')
    os.system('cp ' + filename + ' ' + outname)
    dd = read_t(outname)
    dd[flux_key] = flux2
    write_t(dd, outname)

    print("\tPersistence correction completed.")

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Correct persistence in astronomical data using a persistence map.")
    parser.add_argument("filenames", type=str, nargs='+', help="The path(s) to the input FITS file(s) containing the data to be corrected.")
    parser.add_argument("--persifile", type=str, default='persi.fits', help="The path to the persistence map FITS file (default: 'persi.fits').")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Process each file
    for filename in args.filenames:
        print(f"Processing file: {filename}")
        correct_persistence(filename, args.persifile)
