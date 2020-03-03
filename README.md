# ICASAR
An algorithm for robustly appling sICA to InSAR data


ICASAR applies spatial independent component analysis (sICA) to interferograms in a robust manner
with the aim of retrieving the latent sources that combined to form them. Applying sICA to InSAR
data that covers volcanic centres is discussed in Ebmeier (2016), and Gaddes et al. (2018), whilst the
ICASO algorithm from which the ICASAR algorithm was derived is described in Himberg et al. (2004).


The ICASAR algorithm is discussed fully in Gaddes et al. (2019), but is summarised here. It
performs sICA many times with both bootstrapped samples of the data and dierent initiations of
the un-mixing matrix. This produces a suite of many repeated sources, which we project onto a 2 D
hyperplane using t-SNE and cluster using HDBSCAN. The source most representative of each cluster
is then found, and a simple inversion performed to calculate the time course for each source (i.e. how
strongly they are used to create each interferogram). If you use this software in your research, please
cite it as Gaddes et al. (2019)
