import numpy as np
import emcee
import corner
import os


aux = np.loadtxt("../results/chains/chains_lcdm_1.dat")
file_name = 'lcdm'

# dimension of the matrix
ndim = 4
chain_size = len(aux)/ndim
nwalkers = 2**7
steps = chain_size/nwalkers
burnin = chain_size/2

# initilize the matrix
samples = np.zeros((chain_size,ndim))

# convert flat values to matrix form
i = 0
for j in range(chain_size):
	for k in range(ndim):
		samples[j][k] = aux[i]
		i = i+1

np.savetxt("chains_table_"+file_name+".dat", samples, fmt='%1.5e')

samples = samples[burnin:,:]

# np.savetxt("chains_table_"+file_name+".dat", samples, fmt='%1.6e')

fig = corner.corner(samples, labels=["M", "$h$", "$\omega_{b}$", "$\omega_{cdm}$", "$w$"],  
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12})
fig.suptitle('walkers: %s steps: %s' %(nwalkers, steps))


i=0
while os.path.exists("triangle_"+file_name+"_%s.png" % i) :
    i += 1

fig.savefig("triangle_"+file_name+"_%s.png" % i)

