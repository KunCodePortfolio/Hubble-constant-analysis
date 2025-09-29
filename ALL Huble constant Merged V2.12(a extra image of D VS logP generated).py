# import needed function
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf
from scipy.stats import norm

"""
For data part 1

imported file 

then write each needed variable 

then run gauss for each of the data point(M) to find error for each of data point(M)

then fit the function to linear y = (x-x(mean)) + c2 with curve_fit, 
find α and β, α_error and β_error use curve fit

then plot

then print overall data include both M error use gauss method and error propagation formula

then calculated chi-square base on real value of M and the ideal value of fit line
"""

# import data from local file, name is same to the orign file name
Pmas, Pmas_er, P, m, A, A_er = pd.read_csv(
    'MW_Cepheids.dat',
    sep=r"\s+",
    comment='#',
    skiprows=2,
    names=["parallax", "err_parallax", "Period", "m", "A", "err_A"]
).T.values

# create a copy of dataset 1, can easily add data to this copy
dataset1 = pd.read_csv(
    'MW_Cepheids.dat',
    sep=r"\s+",
    comment='#',
    skiprows=2,
    names=["parallax", "err_parallax", "Period", "m", "A", "err_A"]
)


# the detailed variable calculaiton part.

# distance in pc
d_pc = 1000 / Pmas
# distance error
d_er = (1000 / Pmas**2) * Pmas_er
# the function of M = m - 5log(d) + 5 - A
M = m - 5 * np.log10(d_pc) + 5 - A


# guass fit to find error
# number of samples
n = 90000

# create a empty list of M(error) to allow the for loop to put in value
M_er = []
# gauss fit for each individual datapoint
for i in range((len(Pmas))):
    # define which order data is using for each time
    nA = A[i]
    nA_er = A_er[i]
    nPmas = Pmas[i]
    nPmas_er = Pmas_er[i]
    nm = m[i]

    # guass fit
    # generate samples, only alpha, beta change, logP error too low
    A_sample = np.random.normal(nA, nA_er, n)
    Pmas_sample = np.random.normal(nPmas, nPmas_er, n)
    
    
    # give function of M, M = m - 5log(1000/Pmas) + 5 - A
    M1_samples = nm - 5*np.log10(1000/Pmas_sample) + 5 - A_sample
    # median of this M1_samples
    M1_median = st.median(M1_samples)
    # normal distribution fit, which is guass fit
    # also defined the mu(average), sigma(uncertainity)
    mu1, sigma1 = norm.fit(M1_samples)
    # mu is the average of the M(asbolute magnitude), just define for easier use in coding
    M1_avg = mu1
    # add a new uncertainity following order in the M(error) list
    M_er.append(sigma1)


# orign logP, treat as x1
lgP_all = np.log10(P)
# new logP, treat as x2, x2 = (x1-x(mean))
lgP = lgP_all - st.mean(lgP_all)


# define function to use later in curve_fit, it is a linear fit
def func(x, a, b):
    return a*x +b

# get variable of actual valye of α, β, error = error of both α, β, but actually include
# more a covariance matrix(include both error and correlation between α, β)
actual_value, error = cf(func, lgP, M)

# error of α, β
error_1 = np.sqrt(np.diag(error))

# value of α, β from new function y = (x-x(mean)) + c2, M = α lgP2 + β2
alpha = actual_value[0]
beta = actual_value[1]

# error define
alpha_err = error_1[0]
beta_err = error_1[1]

# print result of α, β with their uncertainity
print("α:",alpha,"±", alpha_err)
print("β:",beta,"±", beta_err)
print("Relationship between α and β:", error[1][1]*100,"%")
print()


# plot the graph

# plot graph with adjusted size, origin was look too squeezed
plt.figure(figsize=(10, 6))
# plot error bar and data point also the description of "Data with Error Bars", and pick point, not line use fmt
plt.errorbar(lgP, M, M_er, fmt="o", capsize=5, label='Data with Error Bars', color='blue', alpha=0.6)
# fit line definition, to make the line straight
fit_line1 = alpha * lgP + beta
# plot fit line(straight line)
plt.plot(lgP, fit_line1, label=f"best Fit of P-luminosity relation: M = {alpha:.2f} "+r"$\log_{10}(P)$ "+f"{beta:.2f}", color= "red")

# Adjusted title size, too small originally
plt.title("M  VS  lgP", fontsize=14)
# x axis title
plt.xlabel(r"$\log_{10}(P)$ (P in Days)", fontsize=12)
# y axis title
plt.ylabel(r" M (Absolute Magnitude)", fontsize=12)
# plot background grid, otherwise it is blank
plt.grid()
# plot explainition bar, where explain what is red line and blue dots with bars
plt.legend()
# showed the plot
plt.show()

# =============================================================================
# start of chi-square part
# =============================================================================
# define observed value of M(absolute magnitude), actual
observed1 = M
# define excepted value of M(absolute magnitude), fit line value
excepted1 = fit_line1
# observed value - excepted value
gap1 = observed1 - excepted1
# chi-square for a single data point
individual_chi_square1 = (gap1 / M_er) ** 2
# sum of all data point's chi-square value
chi_square1 = np.sum(individual_chi_square1) 

# find degree of freedom, -2 due to only 2 parameter
degree_of_freedom1 = len(observed1) - 2
# reduced chi-square, base on gauss method
reduced_chi_square1 = chi_square1 / degree_of_freedom1

# print degree of freedom
print("degree of freedom =", degree_of_freedom1)
# print chi-square value, based on gauss method
print("X² = ", chi_square1)
# print reduced chi-square value, based on gauss method
print("reduced Xᵥ²:", reduced_chi_square1)
print()

# =============================================================================
# end of chi-square part
# =============================================================================

M_error_formula = np.sqrt((5*d_er / (np.log(10)*d_pc))**2 + (A_er)**2)
dataset1["M error(Gauss)"] = M_er
dataset1["M error(formula)"] = M_error_formula

# print overall data include both M error use gauss method and error propagation formula
print(dataset1)
print("__________________________________________________________________________")
print()



"""
For data part 2

imported file 

then write each needed variable 

then run gauss for each of the data point(d_Mpc) to find error for each of data point(d_Mpc)

then plot the [d  VS  logP] with weighted mean as the ideal best fit line

then calculate chi-square base on real value of d and the value of weighted mean
and also improved chi-square base on result of mean from gauss method
"""

# import data from local file, name is same to the orign file name
logP2, m2 = pd.read_csv(
    'ngc4527_cepheids.dat',
    sep=r"\s+",
    comment='#',
    skiprows=1,
    names=["logP", "m"]
).T.values

# A given by professor
A = 0.0682

# median of all values
m2_median = st.median(m2)
logP2_median = st.median(logP2)

# generate samples, only alpha, beta change, logP error too low
alpha_sample = np.random.normal(alpha, alpha_err, n)
beta_sample = np.random.normal(beta, beta_err, n)


d_Mpc_er = []
d_Mpc_orign = []
for i in range((len(logP2))):
    nlogP2 = logP2[i]
    nm2 = m2[i]

    # guass fit
    # generate samples, only alpha, beta change, logP error too low
    M2 = alpha_sample * (nlogP2 - st.mean(lgP_all)) + beta_sample
    d_Mpc2_origin = 10**((M2 - nm2 -5 + A)/(-5)) / 10**6
    
    # normal distribution fit, which is guass fit
    # also defined the mu(average), sigma(uncertainity)
    mu2, sigma2 = norm.fit(d_Mpc2_origin)
    
    d_Mpc_er.append(sigma2)
    d_Mpc_orign.append(mu2)

d_Mpc_orign_weighted_mean = np.average(d_Mpc_orign, weights = d_Mpc_er)

# size of the plot
plt.figure(figsize=(10, 5))
# plot title
plt.title("Distance  VS  logP")
# plot x-axis title
plt.xlabel("logP")
# plot y-axis title
plt.ylabel("d(Mpc)")

# plot error bar and data point also the description of "Data with Error Bars", and pick point, not line use fmt
plt.errorbar(logP2, d_Mpc_orign, yerr = d_Mpc_er, fmt="o", capsize=5, label='Data with Error Bars', color='blue', alpha=0.6)
# fit line definition, to make the line straight
fit_line2 = d_Mpc_orign_weighted_mean*logP2/logP2
# plot fit line(straight line)
plt.plot(logP2, fit_line2, label="best Fit of d(Mpc) VS logP ", color= "red")
# plot description bar on the top right, says what is 
plt.legend()
# background lines(grids)
plt.grid()
# plot the graph
plt.show()


M2_median = alpha_sample * (logP2_median - st.mean(lgP_all)) + beta_sample

# give function of M, what is d(Mpc) = 10^[(M-m-5+1)/(-5)] / 10^6
d_Mpc = 10**((M2_median - m2_median -5 + A)/(-5)) / 10**6
d_Mpc_median = st.median(d_Mpc)
# normal distribution fit, which is guass fit
# also defined the mu(average), sigma(uncertainity)
mu2, sigma2 = norm.fit(d_Mpc)
# mu is the average of the d(Mpc), just define for easier use in coding
d_Mpc_avg = mu2


# size of the plot
plt.figure(figsize=(10, 5))
# plot title
plt.title("Distance  VS  logP (with better fit line)")
# plot x-axis title
plt.xlabel("logP")
# plot y-axis title
plt.ylabel("d(Mpc)")

# plot error bar and data point also the description of "Data with Error Bars", and pick point, not line use fmt
plt.errorbar(logP2, d_Mpc_orign, yerr = d_Mpc_er, fmt="o", capsize=5, label='Data with Error Bars', color='blue', alpha=0.6)
# fit line definition, to make the line straight
fit_line2_new = d_Mpc_median*logP2/logP2
# plot fit line(straight line)
plt.plot(logP2, fit_line2_new, label="best Fit of d(Mpc) VS logP ", color= "red")
# plot description bar on the top right, says what is 
plt.legend()
# background lines(grids)
plt.grid()
# plot the graph
plt.show()

# =============================================================================
# chi-square part
# =============================================================================
# observed value
observed2 = d_Mpc_orign
# excepted value
excepted2 = fit_line2
# degree of freedom, -1 due to only only 1 parameter
degree_of_freedom2 = len(observed2) - 1

# value of obsvered - estimated
gap2 = observed2 - excepted2
# individual chi_square and chi-square
individual_chi_square2 = (gap2 / d_Mpc_er) ** 2
chi_square2 = np.sum(individual_chi_square2)
reduced_chi_square2 = chi_square2 / degree_of_freedom2

# outlier value of obsvered - estimated 
gap2_outlier = d_Mpc_orign[6] - d_Mpc_orign_weighted_mean
# individual chi_square and new chi-square when remove outlier
individual_chi_square2_outlier = (gap2_outlier / d_Mpc_er[6]) ** 2 
chi_square2_new = chi_square2 - individual_chi_square2_outlier
reduced_chi_square2_new = chi_square2_new / degree_of_freedom2


# # new method based on use median instead weighted mean
# median value, *logP2/logP2 to turn d_Mpc_median to a arrary of same dimension
excepted2_median = d_Mpc_median*(logP2/logP2)
# outlier value of obsvered - estimated 
gap2_outlier_median = d_Mpc_orign[6] - d_Mpc_median
# individual chi_square and new chi-square when remove outlier
individual_chi_square2_outlier_median = (gap2_outlier_median / d_Mpc_er[6]) ** 2 
# value of obsvered - estimated
gap2_median = observed2 - excepted2_median
# individual chi_square and chi-square
individual_chi_square2_median = (gap2_median / d_Mpc_er) ** 2
chi_square2_median = np.sum(individual_chi_square2_median) - individual_chi_square2_outlier
reduced_chi_square2_median = chi_square2_median / degree_of_freedom2

# print value
print("=====================================")
# print origin value of degree of freedom, X², reduced X²
print("< origin >")
print("degree of freedom =", degree_of_freedom2)
print("X² = ", chi_square2)
print("reduced Xᵥ²:", reduced_chi_square2)
print("=====================================")
# print value of degree of freedom, X², reduced X² when just removed outlier
print("< removed outlier >")
print("X²=", chi_square2_new)
print("reduced Xᵥ²:", reduced_chi_square2_new)
print("=====================================")
# print value of degree of freedom, X², reduced X² hen removed outlier and fit the best fit line with median of data
print("< removed outlier + fit with median >")
print("X²=", chi_square2_median)
print("reduced Xᵥ²:", reduced_chi_square2_median)
print("=====================================")
print()

# =============================================================================
# end of chi-square part
# =============================================================================

# set the grah size
plt.figure(figsize=(9, 6))
# plot actual graph, should be guass trend
plt.hist(d_Mpc, bins=100, density=True, alpha=0.6, color='blue', label='Simulated d Distribution')

# fit line, and plot fit line
x = np.linspace(min(d_Mpc), max(d_Mpc), 100)
# the description of best fit line and best fit line plot
plt.plot(x, norm.pdf(x, mu2, sigma2), 'r-', label=f'Gaussian Fit: μ={mu2:.3f}, σ={sigma2:.3f}')

# define uncertainity of d(Mpc) use gauss method
unc_d_Mpc = sigma2
# plot title
plt.title("Gaussian Fit of d(Mpc)")
# plot x-axis title
plt.xlabel("Distance(Mpc)")
# plot y-axis title
plt.ylabel("Probability Density(accumlated)")
# plot description bar on the top right, says what is 
plt.legend()
# background lines(grids)
plt.grid()
# plot the graph
plt.show()

# print origin value of d(Mpc)
print("< origin >")
print("mean d(Mpc) =",d_Mpc_orign_weighted_mean,"±", np.mean(d_Mpc_er))
# print new value of d(Mpc), based on gauss
print("< new >")
print("average d(Mpc) =",d_Mpc_avg,"±", unc_d_Mpc)
print("median d(Mpc) =", d_Mpc_median,"±", unc_d_Mpc)
print("__________________________________________________________________________")
print()


# import data from local file, name is same to the orign file name
Re, d, d_er  = pd.read_csv(
    'other_galaxies.dat',
    # Spyder give warning that dont use delim_whitespace=True, but use sep=r"\s+" instead due to the update of future
    sep=r"\s+",
    comment='#',
    skiprows=2,
    names=["Recession", "distance", "distance error"]
).T.values

# new list of Recession velocity, added new data point
n_Re = np.append(Re, 1152)
# new list of distance(Mpc),used avg of guass of d(Mpc) from data 2
n_d_avg = np.append(d, d_Mpc_avg)
# new list of distance(Mpc),used median of guass of d(Mpc) from data 2
n_d_med = np.append(d, d_Mpc_median)
# new list of d(error), added new data point
n_d_er = np.append(d_er, unc_d_Mpc)



# the detailed variable calculaiton part.


# create empty list of Hubble, hubble error, wait to filled later
Hubble = []
Hubble_error = []
# a for loop to do calculation with given order
for i in range((len(n_Re))):
    # define the order of data we going to have for each turn
    new_n_Re = n_Re[i]
    new_n_d_med = n_d_med[i]
    new_n_d_er = n_d_er[i]
    # guass fit

    # generate samples, only alpha, beta change, logP error too low
    d_sample = np.random.normal(new_n_d_med, new_n_d_er, n)
    
    
    # give function of H, H = v / D
    H = new_n_Re / d_sample
    # average value of H samples
    avg_H = st.mean(H)
    # average, uncertainty of current data point, depends on i
    mu, sigma = norm.fit(H)
    # add value of H, H(error) in corresponding list
    Hubble.append(avg_H)
    Hubble_error.append(sigma)
    # print(sigma)

# weighted mean, but use data from gauss method
Hb_constant_new_weighted = Hbnw = (st.mean(Hubble))

# print result of [Hubble constant] with uncertainity
print("Hubble constant =", Hbnw,"±", st.mean(Hubble_error),"")
print("Hubble constant(Origin) =", st.mean(n_Re/n_d_med),"")
print()

# plot the graph

# plot graph with adjusted size, origin was look too squeezed
plt.figure(figsize=(10, 6))
# plot error bar and data point also the description of "Data with Error Bars", and pick point, not line use fmt
plt.errorbar(n_d_med, n_Re, xerr = n_d_er, fmt="o", capsize=5, label='Data with Error Bars', color='blue', alpha=0.6)
# fit line definition, to make the line straight
fit_line3 = Hbnw * n_d_med

# plot fit line(straight line)
plt.plot(n_d_med, fit_line3, label=f"best Fit of Hubble law : v = {Hbnw:.2f} "+"* D", color= "red")

# Adjusted title size, too small originally
plt.title("v(Recession velocity)  VS  D(Distance)", fontsize=14)
# x axis title
plt.xlabel(r"Distance(Mpc)", fontsize=12)
# y axis title
plt.ylabel(r"Recession velocity(km/s)", fontsize=12)
# plot background grid, otherwise it is blank
plt.grid()
# plot explainition bar, where explain what is red line and blue dots with bars
plt.legend()
# showed the plot
plt.show()

# =============================================================================
# chi-square part
# =============================================================================

# define observed data = Recession velocity
observed3 = n_Re
# define exepted data = value of fit line
excepted3 = fit_line3
# observed value - excepted value
gap3 = observed3 - excepted3
# arry list of chi-square of each individual data point, base on gauss method
individual_chi_square3 = (gap3 / Hubble_error) ** 2
# arry list of chi-square of each individual data point, base on origin d(error), error on Distance
individual_chi_square3_old = (gap3 / n_d_er) ** 2
# sum of chi-square, base on gauss method
chi_square3 = np.sum(individual_chi_square3)
# sum of chi-square, base on origin d(error), origin error of distance
chi_square3_old = np.sum(individual_chi_square3_old)

# degree of freedom, -1 due to only 1 parameter
degree_of_freedom3 = len(observed3) - 1
# overall chi-square based on d(error), origin error of distance
reduced_chi_square3_origin = chi_square3_old / degree_of_freedom3
# overall chi-square based on gauss method
reduced_chi_square3 = chi_square3 / degree_of_freedom3


# print the degree of freedom
print("degree of freedom = ",degree_of_freedom3)
print("===========================================")
# print value of X², reduced X², base on d(error), origin error of distance
print("< origin >")
print("X² = ", chi_square3_old)
print("reduced Xᵥ²:", reduced_chi_square3_origin)
print("===========================================")
# print value of X², reduced X², base on gauss method
print("< new >")
print("X²(new error) = ", chi_square3)
print("reduced Xᵥ²(new error):", reduced_chi_square3)
print("===========================================")

# =============================================================================
# end of chi-square part
# =============================================================================
