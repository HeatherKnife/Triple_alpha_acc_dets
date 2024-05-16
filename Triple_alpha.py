import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_chn_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        counts = struct.unpack(f'{len(data) // 2}H', data)
        counts = counts[::2]
    return counts


def read_spe_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Skip metadata and find the start of the spectrum data
        spectrum_start = lines.index("$DATA:\n") + 2
        counts = []
        for line in lines[spectrum_start:]:
            if line.strip().isdigit():
                counts.append(int(line.strip()))
    return counts

def plot_histogram(counts, title, hist_range=None):
    plt.figure()
    plt.hist(range(len(counts)), bins=len(counts), weights=counts, edgecolor='black', range=hist_range)
    plt.title(title)
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.show()

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def perform_gaussian_fit(counts, subpeak_ranges):
    fit_params = []
    fit_errors = []
    x_data = np.arange(len(counts))

    plt.figure(figsize=(15, 7))
    plt.hist(x_data, bins=len(counts), weights=counts, edgecolor='black', alpha=0.5, label='Data')

    for idx, (start, end) in enumerate(subpeak_ranges):

        print("start ", start)
        print("end ", end)

        x_range = x_data[start:end]
        y_range = counts[start:end]

        # Initial guess for the parameters: amplitude, mean, and standard deviation
        initial_guess = [max(y_range), np.mean(x_range), np.std(x_range)]

        # Perform the Gaussian fit
        params, covariance = curve_fit(gaussian, x_range, y_range, p0=initial_guess)
        fit_params.append(params)

        # Calculate the standard errors (uncertainties) of the fit parameters
        errors = np.sqrt(np.diag(covariance))
        fit_errors.append(errors)

        # Plot the fitted Gaussian
        x_fit = np.linspace(start, end, 500)
        y_fit = gaussian(x_fit, *params)
        plt.plot(x_fit, y_fit, label=f'Fit Range {start}-{end}')

    plt.title('Gaussian Fit')
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()

    return fit_params, fit_errors

def perform_gaussian_fit_calibrated(counts, slope, intercept, subpeak_ranges):
    fit_params = []
    fit_errors = []
    x_data = np.arange(len(counts))

    # Apply the calibration
    calibrated_x_data = slope * x_data + intercept

    plt.figure(figsize=(15, 7))
    plt.hist(calibrated_x_data, bins=len(counts), weights=counts, edgecolor='black', alpha=0.5, label='Data')

    for idx, (start_val, end_val) in enumerate(subpeak_ranges):

        # Find the indices corresponding to the calibrated values
        start_idx = np.searchsorted(calibrated_x_data, start_val, side='left')
        end_idx = np.searchsorted(calibrated_x_data, end_val, side='right')
        
        x_range = calibrated_x_data[start_idx:end_idx]
        y_range = counts[start_idx:end_idx]

        # Skip empty ranges
        if len(y_range) == 0 or max(y_range) == 0:
            print(f"Empty or invalid y_range for range: {start_val}-{end_val}")
            continue

        # Initial guess for the parameters: amplitude, mean, and standard deviation
        initial_guess = [max(y_range), np.mean(x_range), np.std(x_range)]

        # Perform the Gaussian fit
        params, covariance = curve_fit(gaussian, x_range, y_range, p0=initial_guess)
        fit_params.append(params)

        # Calculate the standard errors (uncertainties) of the fit parameters
        errors = np.sqrt(np.diag(covariance))
        fit_errors.append(errors)

        # Plot the fitted Gaussian
        x_fit = np.linspace(min(x_range), max(x_range), 500)
        y_fit = gaussian(x_fit, *params)
        plt.plot(x_fit, y_fit, label=f'Fit Range {start_val}-{end_val}')

    plt.title('Gaussian Fit with Calibration')
    plt.xlabel('Calibrated Channel')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()

    return fit_params, fit_errors

def linear(x, m, c):
    return m * x + c

def perform_energy_channel_calibration(fit_params, fit_errors):
    centroids = [params[1] for params in fit_params]
    uncertainties = [errors[1] for errors in fit_errors]

    plt.figure(figsize=(10, 5))

    energies = np.array([5.155, 5.486, 5.805])

    # Perform linear fit
    initial_guess = [1, 0]  # Initial guess for slope (m) and intercept (c)
    params, covariance = curve_fit(linear, centroids, energies, sigma=uncertainties, absolute_sigma=True, p0=initial_guess)
    fit_errors = np.sqrt(np.diag(covariance))
    
    # Linear fit parameters and their uncertainties
    slope, intercept = params
    slope_error, intercept_error = fit_errors

    # Plot centroids with error bars
    plt.errorbar(centroids, energies, xerr=uncertainties, fmt='o', label='Fit Centroids')

    # Plot linear fit
    x_fit = np.linspace(min(centroids), max(centroids), 100)
    y_fit = linear(x_fit, *params)
    plt.plot(x_fit, y_fit, label=f'Linear Fit: y = {slope:.3f}x + {intercept:.3f}')

    plt.title('Fit Centroids with Uncertainties and Specified Points')
    plt.xlabel('Energy')
    plt.ylabel('Centroid')
    plt.legend()
    plt.show()

    # If needed for debugging
    # print(f"Linear Fit Parameters: Slope = {slope:.3f} ± {slope_error:.3f}, Intercept = {intercept:.3f} ± {intercept_error:.3f}")

    return slope, intercept, slope_error, intercept_error

def plot_sigmas_with_errors(fit_params1, fit_errors1, fit_params2, fit_errors2, fit_params3, fit_errors3, fit_params4, fit_errors4, fit_params5, fit_errors5, fit_params6, fit_errors6, labels1, labels2, labels3, labels4, labels5, labels6):

    sigmas1 = [2.355 * params[2] * 1e3 for params in fit_params1]
    sigma_errors1 = [2.355 * errors[2] * 1e3 for errors in fit_errors1]

    sigmas2 = [2.355 * params[2] * 1e3 for params in fit_params2]
    sigma_errors2 = [2.355 * errors[2] * 1e3 for errors in fit_errors2]

    sigmas3 = [2.355 * params[2] * 1e3 for params in fit_params3]
    sigma_errors3 = [2.355 * errors[2] * 1e3 for errors in fit_errors3]

    sigmas4 = [2.355 * params[2] * 1e3 for params in fit_params4]
    sigma_errors4 = [2.355 * errors[2] * 1e3 for errors in fit_errors4]

    sigmas5 = [2.355 * params[2] * 1e3 for params in fit_params5]
    sigma_errors5 = [2.355 * errors[2] * 1e3 for errors in fit_errors5]

    sigmas6 = [2.355 * params[2] * 1e3 for params in fit_params6]
    sigma_errors6 = [2.355 * errors[2] * 1e3 for errors in fit_errors6]

    positions = np.arange(len(sigmas1))

    plt.figure(figsize=(10, 5))
    
    plt.errorbar(positions, sigmas1, yerr=sigma_errors1, fmt='o', label=labels1)
    plt.errorbar(positions, sigmas2, yerr=sigma_errors2, fmt='o', label=labels2)
    plt.errorbar(positions, sigmas3, yerr=sigma_errors3, fmt='o', label=labels3)
    plt.errorbar(positions, sigmas4, yerr=sigma_errors4, fmt='o', label=labels4)
    plt.errorbar(positions, sigmas5, yerr=sigma_errors5, fmt='o', label=labels5)
    plt.errorbar(positions, sigmas6, yerr=sigma_errors6, fmt='o', label=labels6)

    peaks = ['Pu', 'Am', 'Cm']

    plt.xticks(positions, peaks)
    plt.xlabel('Peaks')
    plt.ylabel('Sigma (keV)')
    plt.title('Sigmas with Errors from Two Data Sets')
    plt.legend()
    plt.show()    

# For the chn file reading .....
header_size_chn = 256
dtype_chn = np.float32
data_chn = read_chn_file('Triple_alpha_HARSHAW_6185.Chn')

# If needed for debugging
# plot_histogram(data_chn, 'First data')

# For the spe file reading .....
data_spe = read_spe_file('Triple_alpha_HARSHAW_6185_closer2det.Spe')
data_spe_1 = read_spe_file('Triple_alpha_HARSHAW_6185_different_shaping_times.Spe')
data_spe_3 = read_spe_file('../FW _Accelerators_and_detectors_-_Lab_data/bu014300500.Spe')
data_spe_4 = read_spe_file('../FW _Accelerators_and_detectors_-_Lab_data/SN4985.Spe')
data_spe_5 = read_spe_file('../FW _Accelerators_and_detectors_-_Lab_data/SN7268.Spe')

# Define the ranges of Pu Am Cm peaks for Gaussian fit
ranges_spe = [(1410, 1430), (1497, 1526), (1594, 1610)]
ranges_spe_1 = [(1400, 1420), (1493, 1510), (1584, 1596)]
ranges_spe_2 = [(1366, 1383), (1457, 1472), (1545, 1557)]
ranges_spe_3 = [(1765, 1788), (1881, 1902), (1994, 2007)]
ranges_spe_4 = [(1710, 1750), (1822, 1856), (1934, 1959)]
ranges_spe_5 = [(1755, 1775), (1873, 1889), (1985, 2000)]

# Perform Gaussian fit and plot data
fit_params_spe, fit_errors_spe = perform_gaussian_fit(data_spe, ranges_spe)
fit_params_spe_1, fit_errors_spe_1 = perform_gaussian_fit(data_spe_1, ranges_spe_1)
fit_params_spe_2, fit_errors_spe_2 = perform_gaussian_fit(data_chn, ranges_spe_2)
fit_params_spe_3, fit_errors_spe_3 = perform_gaussian_fit(data_spe_3, ranges_spe_3)
fit_params_spe_4, fit_errors_spe_4 = perform_gaussian_fit(data_spe_4, ranges_spe_4)
fit_params_spe_5, fit_errors_spe_5 = perform_gaussian_fit(data_spe_5, ranges_spe_5)

# Energy-channel calibration
slope, intercept, slope_error, intercept_error = perform_energy_channel_calibration(fit_params_spe, fit_errors_spe)
slope_1, intercept_1, slope_error_1, intercept_error_1 = perform_energy_channel_calibration(fit_params_spe_1, fit_errors_spe_1)
slope_2, intercept_2, slope_error_2, intercept_error_2 = perform_energy_channel_calibration(fit_params_spe_2, fit_errors_spe_2)
slope_3, intercept_3, slope_error_3, intercept_error_3 = perform_energy_channel_calibration(fit_params_spe_3, fit_errors_spe_3)
slope_4, intercept_4, slope_error_4, intercept_error_4 = perform_energy_channel_calibration(fit_params_spe_4, fit_errors_spe_4)
slope_5, intercept_5, slope_error_5, intercept_error_5 = perform_energy_channel_calibration(fit_params_spe_5, fit_errors_spe_5)

ranges_spe_calibrated = [(5.12, 5.19), (5.46, 5.52), (5.78, 5.83)]

fit_params_spe_calibration, fit_errors_spe_calibration = perform_gaussian_fit_calibrated(data_spe, slope, intercept, ranges_spe_calibrated)
fit_params_spe_calibration_1, fit_errors_spe_calibration_1 = perform_gaussian_fit_calibrated(data_spe_1, slope_1, intercept_1, ranges_spe_calibrated)
fit_params_spe_calibration_2, fit_errors_spe_calibration_2 = perform_gaussian_fit_calibrated(data_chn, slope_2, intercept_2, ranges_spe_calibrated)
fit_params_spe_calibration_3, fit_errors_spe_calibration_3 = perform_gaussian_fit_calibrated(data_spe_3, slope_3, intercept_3, ranges_spe_calibrated)
fit_params_spe_calibration_4, fit_errors_spe_calibration_4 = perform_gaussian_fit_calibrated(data_spe_4, slope_4, intercept_4, ranges_spe_calibrated)
fit_params_spe_calibration_5, fit_errors_spe_calibration_5 = perform_gaussian_fit_calibrated(data_spe_5, slope_5, intercept_5, ranges_spe_calibrated)

print("SPE Fit Parameters and Uncertainties:")
for i, (params, errors) in enumerate(zip(fit_params_spe_calibration, fit_errors_spe_calibration)):
    height, centroid, sigma = params
    height_err, centroid_err, sigma_err = errors
    print(f"Range {ranges_spe_calibrated[i]}:")
    print(f"  Centroid: {centroid:.3f} MeV ± {centroid_err:.3f} MeV")
    print(f"  Sigma: {(2.355*sigma)*1e3:.5f} keV ± {(2.355*sigma_err)*1e3:.5f} keV")
    print(f"  Height: {height:.3f} ± {height_err:.3f}")

print("SPE Fit Parameters and Uncertainties farer from detector:")
for i, (params, errors) in enumerate(zip(fit_params_spe_calibration_1, fit_errors_spe_calibration_1)):
    height, centroid, sigma = params
    height_err, centroid_err, sigma_err = errors
    print(f"Range {ranges_spe_calibrated[i]}:")
    print(f"  Centroid: {centroid:.3f} MeV ± {centroid_err:.3f} MeV")
    print(f"  Sigma: {(2.355*sigma)*1e3:.5f} keV ± {(2.355*sigma_err)*1e3:.5f} keV")
    print(f"  Height: {height:.3f} ± {height_err:.3f}")

print("SPE Fit Parameters and Uncertainties first SP:")
for i, (params, errors) in enumerate(zip(fit_params_spe_calibration_2, fit_errors_spe_calibration_2)):
    height, centroid, sigma = params
    height_err, centroid_err, sigma_err = errors
    print(f"Range {ranges_spe_calibrated[i]}:")
    print(f"  Centroid: {centroid:.3f} MeV ± {centroid_err:.3f} MeV")
    print(f"  Sigma: {(2.355*sigma)*1e3:.5f} keV ± {(2.355*sigma_err)*1e3:.5f} keV")
    print(f"  Height: {height:.3f} ± {height_err:.3f}") 

# Plot results
plot_sigmas_with_errors(fit_params_spe_calibration, fit_errors_spe_calibration, fit_params_spe_calibration_1, fit_errors_spe_calibration_1, fit_params_spe_calibration_2, fit_errors_spe_calibration_2, fit_params_spe_calibration_3, fit_errors_spe_calibration_3, fit_params_spe_calibration_4, fit_errors_spe_calibration_4, fit_params_spe_calibration_5, fit_errors_spe_calibration_5, 'Close to source', 'Far from source', 'first SP', 'bu014300500', 'SN4985', 'SN4985')