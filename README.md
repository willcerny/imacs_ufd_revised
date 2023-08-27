# imacs_ufd_revised
A revision to Ting Li's code for fitting IMACS spectra to get radial velocities and metallicities from the Calcium Triplet.
The revision focuses on building a version that can batch run lots of different masks worth of data, and thus shouldn't necessarily be used for all purposes. 
To do so, and to improve upon other areas, there are four main sets of new capabilities:


 (1) controlling input parameters via a config.yaml file,
 
 (2) running two different equivalent width measurements codes,
 
 (3) expanding the RV template grid to include a subgiant and a dwarf star (todo: make this flexible)
 
 (4) improved documentation / general cleaning


 A wish list feature is to implement fitting the Mg I $\lambda$ 8807 line 


 
