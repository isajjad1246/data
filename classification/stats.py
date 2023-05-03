"""
Do about 5 experiments.
Get avg and stddev of 5 runs of 10%, 5 runs of 20%, etc.
this file will just use the data collected from dataClassifier and create the graph
"""
import matplotlib.pyplot as plt
import numpy as np

# ######DIGITS#########
# #graph averages- ACCURACY
# #naive bayes
# naiveX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# naiveY = np.array([0, 30.4,34.8,44.2,51.8,56.4,55.6,61.0,59.8,63.2,64])
# #perceptron
# percX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# percY = np.array([0, 37.8,  49.6,  61.4,  60.8,  60.2,  60.,   60.,   60.,   60.,   60. ])
# #knn
# knnX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# knnY = np.array([0, 26.8,  34.6,  45.6, 52.4,  58.,   57.,   58.4,  62.6,  63.2,  64.2])

# plt.plot(naiveX,naiveY,label="naive bayes")
# plt.plot(percX,percY,label="perceptron")
# plt.plot(knnX,knnY,label="knn")

# plt.legend(loc='lower center')
# plt.title("Accuracy Averages")
# plt.figure()

# #TIME AVGS
# #naive bayes
# naiveTimeX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# naiveTimeY = [0, 1.17369123,   3.35878901,   5.38366265,   7.23315296,   9.28426723,
#   10.52953358,  12.20957198,  14.52791963,  17.61717238,  18.90291986]

# #perceptron
# percTimeX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# percTimeY = [0, 0.23460641,  0.50340524,  0.78971162,  1.02826042,  1.24762115,  1.57119074,
#   1.83412104,  1.95437598,  2.27491326,  2.59559722]

# #knn
# knnTimeX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# knnTimeY = np.array([0, 0.00196438,  0.00377779,  0.00685062,  0.00915856,  0.01046,     0.01191621,
#   0.01472874,  0.01614575,  0.02301536,  0.01993303])

# plt.plot(naiveTimeX,naiveTimeY, label="naive bayes")
# plt.plot(percTimeX,percTimeY, label="perceptron")
# plt.plot(knnTimeX,knnTimeY,label="knn")
# plt.legend("lower center")
# plt.title("Time Averages")

# plt.figure()

# ###STDDEV accuracies
# #naive bayes
# stdNaiveX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# stdNaiveY = [0, 8.45221864,  6.30555311,  8.97552227,  2.99332591,  2.93938769,
#         4.12795349,  2.28035085,  2.03960781,  1.16619038,  0.        ]

# #perceptron
# stdPercX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# stdPercY = np.array([0, 2.92574777,  1.0198039 ,  1.62480768,  0.9797959 ,  0.9797959 ,
#         0.        ,  0.        ,  0.        ,  0.        ,  0.        ])

# #knn
# stdKnnX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# stdKnnY = np.array([0, 4.83321839,  3.97994975,  3.26190129,  4.54312668,  2.82842712,
#         3.63318042,  2.87054002,  1.8547237 ,  1.16619038,  0.4       ])

# plt.plot(stdNaiveX,stdNaiveY,label="naive bayes")
# plt.plot(stdPercX,stdPercY,label="perceptron")
# plt.plot(stdKnnX,stdKnnY,label="knn")

# plt.legend(loc='lower center')
# plt.title("Accuracy Standard Deviation")
# plt.figure()

# ##stddev TIME
# #naive bayes
# stdTimeNaiveX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# stdTimeNaiveY = [0, 0.16728958,  0.22529976,  0.11700574,  0.53014419,  0.24964613,
#         0.13441956,  0.12015557,  0.55914647,  2.16212005,  1.14906486]

# #perceptron
# stdTimePercX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# stdTimePercY = np.array([0, 0.02667395,  0.01183083,  0.03920343,  0.07225347,  0.05340041,
#         0.09517255,  0.12265426,  0.06394061,  0.10631427,  0.3029164 ])

# #knn
# stdTimeKnnX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# stdTimeKnnY = np.array([0, 0.0002143 ,  0.00020012,  0.00016267,  0.00043829,  0.00041392,
#         0.00047767,  0.00127011,  0.00028533,  0.01018988,  0.00061252])

# plt.plot(stdTimeNaiveX,stdTimeNaiveY, label="naive bayes")
# plt.plot(stdTimePercX,stdTimePercY, label="perceptron")
# plt.plot(stdTimeKnnX,stdTimeKnnY,label="knn")

# plt.legend(loc='upper center')
# plt.title("Time Standard Deviation")

# plt.show()
# #####END OF DIGITS#########

#####IMAGES########
###accuracy averages###
#naive bayes
InaiveX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
InaiveY = np.array([0, 53.2,  61.,   57.,   59.6,  64.,   62.,   64.8,  74.6,  74.4,  79. ])
#perceptron
IpercX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IpercY = np.array([0, 64.8,  73.,   76.,   84.,   84.,   84.,   84.,   84.,   84.,   84., ])
#knn
IknnX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IknnY = np.array([0, 49.2,  55.2,  54.8,  62.2,  52.8,  56.2,  57.6,  56.2,  57.6,  59. ])

plt.plot(InaiveX,InaiveY,label="naive bayes")
plt.plot(IpercX,IpercY,label="perceptron")
plt.plot(IknnX,IknnY,label="knn")

plt.legend(loc='lower center')
plt.title("Faces Accuracy Averages")
plt.figure()

#TIME AVGS
#naive bayes
InaiveTimeX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
InaiveTimeY = np.array([0,  1.98619642,   4.01011281,   6.03945518,   7.91391344,   9.923489,
  11.99412246,  15.12529516,  16.62578859,  17.87256899,  19.9901526 ])

#perceptron
IpercTimeX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IpercTimeY = np.array([0, 0.3321136,   0.58856072,  0.81678128,  1.04565897,  1.28396416,  1.54995084,
  1.86209459,  2.27592683,  2.31941109,  2.70506101])

#knn
IknnTimeX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IknnTimeY = np.array([0, 0.01135073,  0.02315707,  0.03565903,  0.04631381,  0.06858859,  0.07753592,
  0.07684302,  0.08820767,  0.09869843,  0.10904675])

plt.plot(InaiveTimeX,InaiveTimeY, label="naive bayes")
plt.plot(IpercTimeX,IpercTimeY, label="perceptron")
plt.plot(IknnTimeX,IknnTimeY,label="knn")
plt.legend("lower center")
plt.title("Faces Time Averages")

plt.figure()

###STDDEV accuracies
#naive bayes
IstdNaiveX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IstdNaiveY = np.array([0, 3.54400903,  7.69415362,  5.4405882 ,  4.40908154,  8.78635305,
        7.15541753,  6.20966988,  3.13687743,  5.46260011,  0.        ])

#perceptron
IstdPercX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IstdPercY = np.array([0, 4.35430821,  2.44948974,  6.13188389,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ])

#knn
IstdKnnX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IstdKnnY = np.array([0, 4.4       ,  2.56124969,  4.48998886,  5.30659966,  3.31058907,
        1.93907194,  3.2       ,  2.31516738,  1.356466  ,  0.        ])

plt.plot(IstdNaiveX,IstdNaiveY,label="naive bayes")
plt.plot(IstdPercX,IstdPercY,label="perceptron")
plt.plot(IstdKnnX,IstdKnnY,label="knn")

plt.legend(loc='lower center')
plt.title("Faces Accuracy Standard Deviation")
plt.figure()

##stddev TIME
#naive bayes
IstdTimeNaiveX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IstdTimeNaiveY = np.array([0, 0.01333548,  0.05472133,  0.1584163 ,  0.1956316 ,  0.13621774,
        0.23545172,  1.16664568,  0.5302358 ,  0.26517864,  0.3606288 ])

#perceptron
IstdTimePercX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IstdTimePercY = np.array([0, 0.01364573,  0.02703003,  0.0206775 ,  0.02005197,  0.00487027,
        0.00988644,  0.12450492,  0.17186805,  0.02921629,  0.27549273])

#knn
IstdTimeKnnX = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
IstdTimeKnnY = np.array([0, 0.00083863,  0.00214715,  0.00367065,  0.00390079,  0.02526463,
        0.01117091,  0.00233422,  0.00205491,  0.00130318,  0.0038369 ])

plt.plot(IstdTimeNaiveX,IstdTimeNaiveY, label="naive bayes")
plt.plot(IstdTimePercX,IstdTimePercY, label="perceptron")
plt.plot(IstdTimeKnnX,IstdTimeKnnY,label="knn")

plt.legend(loc='upper center')
plt.title("Face Time Standard Deviation")

plt.show()

