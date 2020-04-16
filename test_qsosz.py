'''
scripts to plot QSOLoaderZ
'''
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from CDDF_analysis.qso_loader import QSOLoaderZ
from CDDF_analysis import make_zqso_plots

matplotlib.use('TkAgg')

delta_z = 1

qsos = make_zqso_plots.generate_qsos()

index = qsos.plot_z_map(delta_z=delta_z)
plt.show()

nspec = 15

qsos.plot_z_sample_posteriors(nspec, dla_samples=1)
plt.show()

# qsos.plot_this_mu(nspec=nspec, suppressed=True, 
#     num_voigt_lines=3, num_forest_lines=6, z_sample=qsos.z_map[nspec])
# plt.show()

# for nspec in np.where(index)[0]:
#     qsos.plot_z_sample_posteriors(nspec, dla_samples=False)
#     make_zqso_plots.save_figure(
#         "posterior_zqso_samples_delta_z_{}_thing_id_{}".format(
#             delta_z, qsos.thing_ids[nspec]))
#     plt.show()

#     qsos.plot_z_sample_posteriors(nspec, dla_samples=True)
#     plt.show()
