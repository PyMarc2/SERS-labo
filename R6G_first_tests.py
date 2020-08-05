from ramanGrapher import *

rmg1 = RamanGrapher(figsize=(9, 8))

#R6G 10-4 #1
#rmg1.load_image("data/04-08-2020/measure_OOSERSAu_R6G[10-4]_15min_30s_31p3.TIF")
#rmg1.modify_subtract_ref_image("data/04-08-2020/ref_OOSERSAu_empty_forR6G10-4M_30s_31p3.TIF")
#rmg1.modify_image_to_summed_plot()
#rmg1.modify_calibration_polynomial(1.67*10**-8, -4.89*10**-5, 0.164, 789)
#rmg1.modify_smoothen(2, 0.2)
#rmg1.add_plot(xunit='cm-1', normalized=False, label="04-08-2020, R6G, C=10$^{-4}$M, 100nm AuNP, 30s")
# rmg1.add_peaks(distance=4, height=0.2, threshold=0, prominence=0.08, width=1)

#R6G REFERENCE 1st measure
#rmg1.load_image("data/02-08-2020/measure_OOSERSAu_R6G_5min-dry_10s_31p3_relcm.TIF")
#rmg1.modify_subtract_ref_image("data/02-08-2020/ref_OOSERSAu_empty_noDescription_10s_31p3_nm.TIF")
#rmg1.modify_image_to_summed_plot()
#rmg1.modify_calibration_polynomial(1.67 * 10 ** -8, -4.89 * 10 ** -5, 0.164, 789)
#rmg1.modify_smoothen(2, 0.2)
#rmg1.add_plot(xunit='cm-1', normalized=False, label="02-08-2020, R6G, C=?, 100nmAuNP, 10s")
#rmg1.add_peaks(distance=4, height=0.2, threshold=0, prominence=0.1, width=1)

#R6G 20mg/ml(saturated) sur Thorlabs paper
#rmg1.load_image("data/05-08-2020/measure_ThorlabsPaper_300s_62mW_#1.tif")
#rmg1.modify_subtract_ref_image("data/05-08-2020/ref_ThorlabsPaper_300s_62mW_#1.tif")
#rmg1.modify_subtract_ref_image("data/05-08-2020/ref_ThorlabsPaper_300s_62mW_#1.tif")
#rmg1.modify_image_to_summed_plot()
#rmg1.modify_calibration_polynomial(1.67 * 10 ** -8, -4.89 * 10 ** -5, 0.164, 789)
#rmg1.modify_smoothen(2, 0.9)
#rmg1.add_plot(xunit='cm-1', normalized=True, label="05-08-2020, R6G, C=saturated, thorlabPaper, 300s")
#rmg1.add_peaks(distance=3, prominence=0.1, width=2,)

rmg1.load_image("data/05-08-2020/measure_KimwipePaper_300s_62mW_#2.tif")
rmg1.modify_subtract_ref_image("data/05-08-2020/ref_KimwipePaper_300s_62mW_#2.tif", 2.2)

rmg1.modify_image_to_summed_plot()
rmg1.modify_calibration_polynomial(1.67 * 10 ** -8, -4.89 * 10 ** -5, 0.164, 789)
rmg1.modify_smoothen(2, 0.9)
rmg1.add_plot(xunit='cm-1', normalized=False, label="05-08-2020, R6G, C=saturated, kimwipePaper, 300s")
#rmg1.add_peaks(distance=3, prominence=0.1, width=2, )

#R6G REFERENCE 1st measure
rmg1.load_image("data/02-08-2020/measure_OOSERSAu_R6G_5min-dry_10s_31p3_relcm.TIF")
rmg1.modify_subtract_ref_image("data/02-08-2020/ref_OOSERSAu_empty_noDescription_10s_31p3_nm.TIF")
rmg1.modify_image_to_summed_plot()
rmg1.modify_calibration_polynomial(1.67 * 10 ** -8, -4.89 * 10 ** -5, 0.164, 789)
rmg1.modify_smoothen(2, 0.2)
rmg1.add_plot(xunit='cm-1', normalized=False, label="02-08-2020, R6G, C=?, 100nmAuNP, 10s")
#rmg1.add_peaks(distance=4, height=0.2, threshold=0, prominence=0.1, width=1)

rmg1.show_plot()