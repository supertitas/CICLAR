First, using "CEEMDAN.py", the decomposed components are obtained.

Then, use "CICLA.py" to predict each IMF component processed by CEEMDAN, which also obtains the hyperparameters of the model.

Finally, the results for each IMF are nonlinearly summed using "RF.py".

It is important to note that the above operations can be time-consuming.

In addition, if the required hyperparameters for the model are known, the previous steps are not necessary.

If you want to use other algorithms, the "Algorithm" folder provides six alternative algorithms for optimizing the hyperparameters.

To facilitate the use of the data, remote sensing, meteorological and hydrological data are integrated in the "Data" folder.
