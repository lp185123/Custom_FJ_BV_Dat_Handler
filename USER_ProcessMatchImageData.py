#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html


import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import _3DVisLabLib
import statistics

#load up json file with all metrics saved after matching images
#make this into a dynamic 
ExportMetrics_filename=r"E:\NCR\TestImages\MatchOutput\Pairs\\ExportedMatchData.md"
ImageMatchMetrics=_3DVisLabLib.JSON_Open(ExportMetrics_filename)

#input metrics is organised as a dictionary with first index of 1, as key, with values as dictionary of image info
MetricArray=[]
MetricArray_DY=[]#for derivative
for Indexer in range (1,int(len(ImageMatchMetrics)/1)):
    MetricArray.append(ImageMatchMetrics[str(Indexer)]["MatchScore"])
    MetricArray_DY.append(Indexer)
MetricArray_np = np.asarray(MetricArray)
MetricArray_DxDY_np = np.asarray(MetricArray_DY)


# x = electrocardiogram()[2000:4000]
# peaks, _ = find_peaks(x, height=0)
# plt.plot(x)
# plt.plot(peaks, x[peaks], "x")
# plt.plot(np.zeros_like(x), "--", color="gray")
# plt.show()

# exit()

#x = electrocardiogram()[2000:4000]
peaks, properties = find_peaks(MetricArray_np)#,prominence=(None, 6.9))
#print(properties["prominences"].max())
plt.plot(MetricArray_np)
plt.plot(peaks, MetricArray_np[peaks], "x")
plt.plot(np.zeros_like(MetricArray_np), "--", color="gray")
plt.show()


#create moving mean/std deviation average -maybe can see where drop off is
MatchMetric_Filter_mean=[]
MatchMetric_Filter_std=[]
MatchMetric_Filter_Dx=[]
FilterSize=5
Buffer = [0] * FilterSize
Buffered_Metric=Buffer + MetricArray + Buffer
for OuterIndexer, Metric in enumerate(Buffered_Metric):
    #break out before hitting end
    if OuterIndexer==len(MetricArray)-FilterSize:break
    #get subset
    MatchMetric_all_subset=Buffered_Metric[OuterIndexer:OuterIndexer+FilterSize]
    MatchMetric_Filter_std.append(statistics.pstdev(MatchMetric_all_subset))
    MatchMetric_Filter_mean.append(statistics.mean(MatchMetric_all_subset))
    MatchMetric_Filter_Dx.append(OuterIndexer)

MatchMetricProdFilter = [a * b for a, b in zip(MatchMetric_Filter_std, MatchMetric_Filter_mean)]
dydx = np.diff(MatchMetricProdFilter)/np.diff(MatchMetric_Filter_Dx)

plt.cla()
plt.clf()
plt.plot(dydx)
#plt.plot(MatchMetricProdFilter)
#plt.plot(np.zeros_like(MatchMetricProdFilter), "--", color="gray")
plt.show()




#run some kind of 1D kernal over data to detect signature of peaks
kernel_edge=np.asarray([0.1,0.2,0.3,0.4,0.5,10,-5,0.1,0.2,0.3,0.4,0.5])#edge filter?
#kernel_edge=[1,4,7,4,1]#gaussian blur
FilterSize=len(kernel_edge)
#divide by

#create buffers either end of the data
Buffer = [0] * FilterSize
Buffered_Metric=Buffer + MetricArray + Buffer
FilteredOutput=[]
for OuterIndexer, Metric in enumerate(Buffered_Metric):
    #break out before hitting end
    if OuterIndexer==len(MetricArray)-FilterSize:break
     #get subset
    MatchMetric_all_subset=np.asarray(Buffered_Metric[OuterIndexer:OuterIndexer+FilterSize])
    #for Elem in MatchMetric_all_subset:
    SingleRes=(MatchMetric_all_subset*kernel_edge).sum()/kernel_edge.sum()
    FilteredOutput.append(SingleRes)





#try convolve filter
Filter=[1,2,3,4,5,4,3,2,1]
con_res1 = np.convolve(Filter, MetricArray_DY,mode='same')

plt.cla()
plt.clf()
plt.plot(FilteredOutput)
#plt.plot(MatchMetricProdFilter)
#plt.plot(np.zeros_like(MatchMetricProdFilter), "--", color="gray")
plt.show()