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

        self.scatter0_all = np.mean(scatter0_all)

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


# save it:

path = "Separately_EMCEEM11.0.pkl"

pkl_file = open(path, 'rb')
result = pickle.load(pkl_file)
pkl_file.close()

print(result.shape)


model = MCMC(kwargs=kwargs)

walker = 20

step = 200

fusion= np.array([0,0,0,0,0,0,0,0])

for i in range(0,walker):

    for j in range(0,step):

        parameters = result[i,j,:]
        kwargs["f0"] = parameters[0]
        kwargs["A1"] = parameters[1]
        kwargs["A2"] = parameters[2]
        kwargs["A3"] = parameters[3]
        kwargs["zt"] = parameters[4]
        kwargs["zs"] = parameters[5]

        model.update_kwargs(kwargs=kwargs)

        chi_i = model.fitting_simultaneously()

        scatter0_i=model.scatter0_all
        print("doing%d"%(i*walker+j))

        print(parameters,chi_i,scatter0_i)

        fusion = np.vstack((fusion,[parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],chi_i,scatter0_i]))

fusion = np.array(fusion)
print(fusion.shape)
print(fusion)

# fusion = parameters+chi+scatter0

path = path.replace("EMCEE","EMCEE_fusion_")

output = open(path, 'wb')
pickle.dump(fusion, output)
output.close()


