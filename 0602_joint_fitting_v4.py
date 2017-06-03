import numpy as np
import pickle
import math
import scipy
import matplotlib.pyplot as plt
from termcolor import colored
import pickle
import matplotlib
import emcee
import scipy.optimize as op



def log10(x):
    if x > 0:
        return math.log10(x)
    else:
        return -np.inf

# Load index for fitting;
fb = 0.17

pkl_file = open('z_target.pkl', 'rb')
z_target = pickle.load(pkl_file)
pkl_file.close()

a_target = 1 / (1 + z_target)

a_target = a_target[::-1]

pkl_file = open('a_index_us.pkl', 'rb')
a_index_us = pickle.load(pkl_file)
pkl_file.close()


pkl_file = open('a_index_Behroozi.pkl', 'rb')
a_index_Behroozi = pickle.load(pkl_file)
pkl_file.close()



class MCMC():
    def __init__(self, kwargs):
        self.kwargs = kwargs

    def update_kwargs(self,kwargs):
        self.kwargs = kwargs


    def stellar_mass_per_interval(self, f_con, delta_Mh):

        fb = 0.15

        return f_con * fb * delta_Mh

    # f_con for Jeremy

    def f_con_Jeremy(self, z, Mh, Ms):

        kwargs = self.kwargs

        f0 = kwargs["f0"]

        z0 = 1
        gamma1 = -3
        gamma2 = 0

        if z > z0:

            return f0 * ((1 + z) / (1 + z0)) ** gamma1

        else:
            return f0 * ((1 + z) / (1 + z0)) ** gamma2

    # f_con for me

    def f_con_Jason(self, a, Mh, Ms):

        z = 1 / a - 1

        kwargs = self.kwargs

        f0 = kwargs["f0"]

        A1 = kwargs["A1"]
        A2 = kwargs["A2"]
        A3 = kwargs["A3"]
        mht = kwargs["mht"]
        mst = kwargs["mst"]
        zt = kwargs["zt"]
        zs = kwargs["zs"]


        z0 = 1
        gamma1 = -3
        gamma2 = 0

        # Remember the critical a/z0
        # Now we use a and a varies from small to big:

        if abs(z-z0)<0.05:

            self.f_con_critical = f0 * ((Mh / mht) ** A1) * ((Ms / mst) ** A2)* ((1. + z) ** A3)




        if z > z0:
            return f0 * ((Mh / mht) ** A1) * ((Ms / mst) ** A2) * ((1. + z) ** A3)

        else:
            # Let's return a function with fixed A1 and A2 but un-fixed A3:
            return f0 * ((Mh / mht) ** A1) * ((Ms / mst) ** A2) * ((1. + z) ** A3)*(math.exp(-(zt-z)**2/zs))
            # return self.f_con_critical * (math.exp(-(zt - z) ** 2 / zs))


##### Quenching: Ignore it for now

    def f_q(self):

        return 1


    def calculate_M_stellar(self):

        kwargs = self.kwargs

        index = str(self.index)


        # Read_Behroozi's Data

        B_path = "Behroozi_revised_M11.0.pkl"

        B_path = B_path.replace("M11.0",index)

        data_path = "/Users/caojunzhi/Desktop/NYU/Laboratory/My_code/Cosmology_2017.4-8_Jason/M10.0/"

        data_path = data_path.replace("M10.0",index)


        pkl_file = open(B_path, 'rb')
        Behroozi_fusion = pickle.load(pkl_file)
        pkl_file.close()


        # Only use 100 halos sometimes. Make it quicker
        n_halo = 300

        # result all is z + Mass_stellar

        result_all = np.array([[10,10,10,10]])



        for halo in range(1, n_halo):

            file_i = "output_fulltree" + str(halo) + ".txt"

            result_i = np.loadtxt(data_path + file_i)

            # calculate M_stellar

            z = result_i[:, 5]

            a = 1/(1+z)

            a = np.array(a)

            a = a[::-1]

            M_h = result_i[:, 1]

            M_h = np.array(M_h)

            M_h = M_h[::-1]

            # plot each halo:

            delta_Mh = [M_h[0]]

            Ms_now = 1

            Ms = [Ms_now]

            f_con_now = self.f_con_Jason(a[0],M_h[0],Ms_now)

            f_con_array = [f_con_now]


            for j in range(1, len(M_h)):
                delta_Mh_j = M_h[j] - M_h[j-1]
                delta_Mh.append(delta_Mh_j)


                # calculate M_stellar
                f_con_array_j = self.f_con_Jason(a[j], M_h[j], Ms_now)

                f_con_array.append(f_con_array_j)

                delta_Ms = f_con_array_j*fb*delta_Mh_j*self.f_q()

                Ms_now = Ms_now+delta_Ms

                Ms.append(Ms_now)



            delta_Mh = np.array(delta_Mh)

            Ms = np.array(Ms)

            f_con_array = np.array(f_con_array)

            # Now we already have small a to big a:

            # Our fusion has a+ Ms, Mh f_con_array

            fusion = np.c_[a,Ms,M_h,f_con_array]
            try:
                result_all = np.vstack((result_all, fusion))

            except:
                none = 1

        # calculate median

        # result_all : z + Mh
        result_all = np.array(result_all)

        a_all = result_all[:,0]
        Ms_all = result_all[:,1]

        median_Ms = []

        scatter = []

        for ii in range(0,len(a_target)):

            index_ii = np.where(a_all==a_target[ii])

            median_Ms.append(np.nanmedian(Ms_all[index_ii]))

            log_Ms_all = np.array([log10(x) for x in Ms_all])

            scatter_i = abs(np.percentile(log_Ms_all[index_ii], [84]) - np.percentile(log_Ms_all[index_ii], [50]))

            scatter.append(scatter_i)

        median_Ms = np.array(median_Ms)
        scatter = np.array(scatter)

        self.a_target = a_target
        self.median_Ms = median_Ms
        self.scatter = scatter
        self.scatter0 = scatter[-1]

        # let's calculate chi-squared between us and Behroozi

        log_median_Ms = [log10(x) for x in median_Ms]

        log_Behroozi_Ms = [log10(x) for x in Behroozi_fusion[:,1]]

        log_median_Ms = np.array(log_median_Ms)
        log_Behroozi_Ms = np.array(log_Behroozi_Ms)


        chi = np.sum((log_median_Ms[a_index_us]-log_Behroozi_Ms[a_index_Behroozi])**2)

        self.chi = chi

        if scatter[-1]>0.2:
            return chi
        else:
            return chi

    def fitting_separately(self,index):

        self.index = index



    def fitting_simultaneously(self):

        index_array = ["M11.0","M11.5","M12.0","M12.5"]

        chi_all = []
        scatter0_all = []

        for x in range(0,len(index_array)):
            self.fitting_separately(index=index_array[x])
            chi_all.append(self.calculate_M_stellar())
            scatter0_all.append(self.scatter[-1])
        chi_all = np.array(chi_all)
        scatter0_all = np.array(scatter0_all)

        if np.mean(scatter0_all)>0.2:

            return np.sum(chi_all)
        else:
            return np.sum(chi_all)









#### Let's do it!




kwargs = {"MCMC":None,"A1": None, "A2": None, "A3": None, "f0": None,"zt": None,"zs": None, "mht": None, "mst": None, "zc": None, "sigmaz": None,
          "alphaz": None, "mhc": None, "msc": None, "sigmah": None, "sigmag": None, "alphah": None, "alphag": None}

# Initial values:
# different f0!
kwargs["f0"] = 0.1

kwargs["A1"] = 0
kwargs["A2"] = 0
kwargs["A3"] = -3
kwargs["zt"] = 1
kwargs["zs"] = 1

kwargs["mht"] = 10 ** (8)
kwargs["mst"] = 10 ** (8)


"""
# Threshold

all_index = ["M11.0","M11.5","M12.0","M12.5"]

model = MCMC(kwargs=kwargs)
model.fitting_separately(index=all_index[1])
model.calculate_M_stellar()

"""


# Let's do a scipy fitting:

# First, let's fit separately
all_index = ["M11.0","M11.5","M12.0","M12.5","M13.0"]

model = MCMC(kwargs=kwargs)
model.fitting_separately(index=all_index[1])

counter = 0

para_chi_sca = np.array([0,0,0,0,0,0,0,0])


def lnlike(theta, x, y):


    global counter

    global para_chi_sca

    # introduce our model here.

    # Theta is a tuple
    f0, A1, A2, A3,zt,zs = theta

    kwargs["f0"] = f0
    kwargs["A1"] = A1
    kwargs["A2"] = A2
    kwargs["A3"] = A3

    kwargs["zt"] = zt
    kwargs["zs"] = zs

    # Only fit 4 parameters for now
    #kwargs["mht"] = mht
    #kwargs["mst"] = mst

    model.update_kwargs(kwargs=kwargs)


    # Let's change the model from simple linear to our complex model.

    # y_model is the value from our method


    chi = model.calculate_M_stellar()


    # record para_chi_sca

    para_chi_sca = np.vstack((para_chi_sca,[f0,A1,A2,A3,zt,zs,chi,model.scatter0]))

    # if you want to fit simultaneously. Use this

    yerr = 0.3

    inv_sigma2 = 1.0 / (yerr ** 2)


    counter = counter + 1
    print("Doing %d" % counter)
    print(chi,f0,A1,A2,A3,zt,zs)


    # Here things become simpler because we have a constant y_err
    # return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

    return -0.5 * (chi * inv_sigma2 - np.sum(np.log(inv_sigma2)))


# Set the range of the parameters
def lnprior(theta):
    f0, A1, A2, A3,zt,zs = theta

    if 0 < f0 < 9 and -5 < A1 < 5 and -5 < A2 < 5 and -15 < A3 < 15 and 0 < zt < 6 and 0 < zs < 5:
        return 0.0
    return -np.inf


# The final ln posterior probability is the sum of the ln prior and ln likelihood

def lnprob(theta, x, y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y)


# From scipy fit:


## This part is for scipy opt
## We need this because we need them as initial values!!
## That's what you discussed with Jeremy.


# nll = lambda *args: -lnlike(*args)
nll = lambda *args: -lnprob(*args)
print("doing scipy opt")
# Only fit f0,A1,A2,A3
# I adjust the likelihood function

# x is a and y is logm: But it doesn't matter since lnlike has nothing to do with x and y.
x = 0
y = 0

result = op.minimize(nll, [kwargs["f0"], kwargs["A1"], kwargs["A2"], kwargs["A3"], kwargs["zt"], kwargs["zs"]],
                     args=(x, y))

print("finish doing scipy opt")

print("result x")
print(result["x"])
f0, A1, A2, A3,zt,zs = result["x"]

kwargs["f0"] = f0
kwargs["A1"] = A1
kwargs["A2"] = A2
kwargs["A3"] = A3

kwargs["zt"] = zt
kwargs["zs"] = zs

print("final")
print(f0, A1, A2, A3,zt,zs)

save_path = "Separately_best_fit_M11.0_v4.pkl"
save_path = save_path.replace("M11.0",str(model.index))


output = open(save_path, 'wb')
pickle.dump(result["x"], output)
output.close()

# save para_chi
para_chi_sca = np.array(para_chi_sca)
print(para_chi_sca.shape)

save_path = save_path.replace("EMCEE","para_chi_sca")
output = open(save_path, 'wb')
pickle.dump(para_chi_sca, output)
output.close()



# Then, let's do some MCMC:


print("doing emcee")

# Define the initial condition of the MCMC chain: Position/initial values

ndim, nwalkers = 6, 40

kwargs["MCMC"] = 1

# Or you can replace result["x"] with your initial values
# pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
pos = [
    [kwargs["f0"], kwargs["A1"], kwargs["A2"], kwargs["A3"], kwargs["zt"], kwargs["zs"]] + 1e-4 * np.random.randn(
        ndim) for i in range(nwalkers)]

# Set up the MCMC chain
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y), threads=8)

print("running MCMC")
sampler.run_mcmc(pos, 200)

# Now we have an array with dimension of 100*500*3: 100 walkers, 500 steps and 3 parameters

result = sampler.chain
print(result)

# save it:

save_path = save_path.replace("best_fit_","EMCEE")
output = open(save_path, 'wb')
pickle.dump(result, output)
output.close()

# save para_chi_sca:

para_chi_sca = np.array(para_chi_sca)
print(para_chi_sca.shape)

save_path = save_path.replace("EMCEE","para_chi_sca")
output = open(save_path, 'wb')
pickle.dump(para_chi_sca, output)
output.close()
