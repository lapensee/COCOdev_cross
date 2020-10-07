import numpy
import hpd

a4 = numpy.load(open("pickles/sourceDev4COCOfreqVaa_rBeta/phiMus.npy", "rb"))
a7 = numpy.load(open("pickles/sourceDev8COCOfreqVaa_rBeta/phiMus.npy", "rb"))
aa = numpy.load(open("pickles/sourceDevAdultsCOCOfreqVaa_rBeta/phiMus.npy", "rb"))


for anAGE in [a4, a7, aa]:
	print "############## NEW AGE ################"
	for i in xrange(8):
		print hpd.HDI_from_MCMC(anAGE[:,:,i].flatten(), .95), numpy.mean(anAGE[:,:,i].flatten())


'''
####### Results
4yr	2.5	95	MEAN	7yrs	2.5	95	MEAN	adult	2.5	95	MEAN
 'logVti'	-5.1294689	-4.1011257	-4.60972		-8.0926905	-6.939806	-7.50202		-12.930324	-10.780352	-11.8796
 'logVsu'	-4.2694407	-3.8510981	-4.05971		-4.6292901	-4.0942063	-4.35615		-5.3168302	-5.0213308	-5.17095
 'logVaa'	-3.1139541	-2.4478118	-2.77687		-5.20017	-3.8562651	-4.52926		-5.7584457	-4.051404	-4.94839
 'logVac'	-12.485683	-4.2659049	-9.44055		-0.10974073	0.48723614	0.186574		0.42187765	0.88691205	0.657423
'crit'	-0.59825742	-0.22707096	-0.407795		0.5012008	0.84337997	0.671244		0.53116924	1.0297635	0.786763
 'critSource'	-0.27962014	-0.1310018	-0.205002		-0.058574751	0.11744962	0.0297276		0.22534637	0.55503148	0.388565
 'rWeak'	0.64505678	0.76607215	0.706161		0.72156072	0.83468443	0.779709		0.7204172	0.83904046	0.781757
 'logrStrong'	-1.0807823	-0.8729164	-0.975613		-0.6599462	-0.48614624	-0.572121		-0.54234761	-0.37782723	-0.461694
'''