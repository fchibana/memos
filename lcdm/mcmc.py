import numpy as np
import emcee
# import corner
import os
import time

from likelihood import lnprob

start_time = time.time()

#chute da posicao inicial dos walkers. a posicao de cada walker eh diferente. o p1_0 eh o centro da bola
p0 = [24.96, 0.69, 0.022, 0.12]

ndim = 4			#no. de parametros

nwalkers = 2**7	    #no. de walkers
steps = 10**1     #no. de passos
# steps = 10**2     #no. de passos
# steps = 6*10**3     #no. de passos
# steps = 10**4     #no. de passos
# steps = 10**5     #no. de passos
burnin = steps/2	#tempo (passos) para os walker passarem o burn in

nthreads = 4

# which data?
file_name = 'lcdm'
base_path = "/Users/fabio/code/fchibana/memos"

# MCMC =============================================================================================

pos = [p0 + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]	#condicoes iniciais dos walkers dentro da bola de centro p1_0

#inicializa as cadeias
sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob, threads=nthreads)

# j = 0
# while os.path.exists("results/chains/chains_"+file_name+"_%s.dat" % j) or os.path.exists("results/chains_"+file_name+"_%s.txt" % j) or  os.path.exists("results/triagle_scf_"+file_name+"_%s.png" % j) or os.path.exists("results/triangle_"+file_name+"_%s.png" % j) :
#     j += 1
#     print(j)

# f = open("results/chains/chains_"+file_name+"_%s.dat" %j, "w")
# f.close()

#roda MCMC
for i, result in enumerate(sampler.sample(pos, iterations=steps,rstate0=np.random.get_state())):
    progress = float(i)/float(steps)*100.
    if progress % 5. == 0:
        print("progress: %.f%%") %progress
        print("nsteps: %.f") %i

    position = result[0]
    # f = open("results/chains/chains_"+file_name+"_%s.dat" %j, "a")
    # for k in range(position.shape[0]):
    #     np.savetxt(f, position[k],fmt='%1.4e')
    # f.close()


#rearranja as cadeias
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

# Analysis ===========================================================================================

#Calcula as medias de cada parametro
medias = []
for i in range(ndim):
    medias.append(np.percentile(samples[:,i],50))

#Calcula os devios para cima e para baixo de cada parametro
desvios = []
for i in range(ndim):
    desvios.append([np.percentile(samples[:,i],84.13) - np.percentile(samples[:,i],50), np.percentile(samples[:,i],50) - np.percentile(samples[:,i],15.869999999999997)])

print("""MCMC result:
    M           = {0[0]} +{1[0][0]} -{1[0][1]}
    h          = {0[1]} +{1[1][0]} -{1[1][1]}
    omega0_b    = {0[2]} +{1[2][0]} -{1[2][1]}
    omega0_cdm  = {0[3]} +{1[3][0]} -{1[3][1]}
""".format(medias, desvios))

# fig = corner.corner(samples, labels=["M", "$h$", "$\Omega_{b} h^2$", "$\Omega_{c} h^2$"],quantiles=(0.16, 0.5, 0.84),show_titles=True, title_kwargs={"fontsize": 12})
# fig.suptitle('walkers: %s steps: %s' %(nwalkers, steps))

# create output files ==================================================================================
end_time = time.time() - start_time
execution_time = 'Execution time: '+str(end_time)+'s'

# np.savetxt("results/chains_"+file_name+"_%s.txt" % j, samples, fmt='%1.4e', header=execution_time)
# fig.savefig("results/triangle_"+file_name+"_%s.png" % j)
