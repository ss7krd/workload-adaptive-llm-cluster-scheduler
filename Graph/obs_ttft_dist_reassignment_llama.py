import matplotlib
import numpy as np
import pandas as pd
# extra for mac
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 60})

from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines

# matplotlib.rcParams["figure.figsize"] = (8, 2)  # (4, 1.3)
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

class SymHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle,xdescent, ydescent, width, height, fontsize, trans):
        xx= 0.5*height
        return super(SymHandler, self).create_artists(legend, orig_handle,xdescent, xx, width, height, fontsize, trans)

N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.09      # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

color_list = []
with open('color_pallete_for_exp_lines.txt','r') as color_file:
	for eachLine in color_file:
		color_list.append(eachLine.strip())

stepX=[0.527,0.81, 1.093, 1.376, 1.659, 1.942, 2.225, 2.508, 2.791, 3.074, 3.357, 3.64, 3.923]
for i in range (0, len(stepX)):
	stepX[i] = stepX[i]/10
	stepX[i] = stepX[i] - .02
#DRAWING VERTICAL CHARTS
# vals = pd.read_csv('rmse_vs_serverNo.csv')
# vals_eachComp = pd.read_csv('training_time_data_each_comp.csv')

# optimization = [8,16,32]
# sixtykValuesModified = []
# for i in sixtykValues:
#     j = (i/100.0)*6415
#     sixtykValuesModified.append(j) 

# rects1 = ax.bar(ind, our_system, width, color='r', edgecolor='black', hatch="+", label="Our System")

# ax.plot(stepX,optimization,c='red',marker="x",mew=4,markersize=26,ls='-',label="Optimization",fillstyle='full', linewidth = 4)

exhaustive = [6, 8, 22, 17, 17, 2, 1, 1, 1, 3, 10, 12, 1]#pdf
exhaustive = [6, 14, 36, 53, 70, 72, 73, 74, 75, 78, 88, 99, 100]#cdf ##(a good result for SLO=0.16s)
llumnix_plus = [2, 3, 6, 21, 28, 43, 71, 89, 91, 91, 91, 95, 100] #(a good result for SLO=0.25s, 90% reqs were within the SLO.)

# exhaustive = [9, 12, 22, 36, 45, 62, 80, 90, 91, 95, 98, 100, 100]
# exhaustive = [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7,0.8,0.8,1,1]#prev-wrong
   
# ax.plot(stepX,np.array(llumnix_plus)/100,c=color_list[0],ls='-',label="Llumnix+",fillstyle='none', linewidth = 4)

sarathi = [4, 1, 17, 11, 8, 16, 6, 5, 6, 16, 3, 7, 1]
sarathi = [4, 9, 15, 19, 23, 31, 39, 48, 61, 78, 83, 94, 100]
sarathi = [3, 3, 5, 9, 13, 21, 29, 38, 51, 68, 73, 85, 100]
# ax.plot(stepX, np.array(static)-5, c = 'blue', marker = 'o', markersize = 26, ls = '-', label = "Scrooge", fillstyle = 'full', linewidth = 4)
# ax.plot(stepX, np.array(sarathi)/100, c = color_list[3], ls = ':', label = "Round-Robin", fillstyle = 'none', linewidth = 4)
# fortykValuesModified = []
# for i in fortykValues:
#     j = (i/100.0)*7222
#     fortykValuesModified.append(j)

# rects2 = ax.bar(ind+width, federated_learning, width, color='b', edgecolor='black', hatch="*", label='Federated Learning')
# 8, 16, 32, 64, 128, 256
# shepherd = [0, 0, 18, 11, 10, 17, 1, 1, 1, 26, 4, 7, 3]
# ax.plot(stepX, np.array(static)-5, c = 'blue', marker = 'o', markersize = 26, ls = '-', label = "Scrooge", fillstyle = 'full', linewidth = 4)
# ax.plot(stepX, np.array(shepherd), c =color_list[1], ls = '-', label = "S++R-s", fillstyle = 'none', linewidth = 4)
# [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7,0.8,0.8,1,1]
llumnix = [9, 1, 12, 11, 8, 16, 6, 5, 7, 16, 3, 7, 0]
llumnix = [9, 10, 22, 33, 41, 57, 63, 68, 75, 91, 94, 100, 100] ##(a good result for SLO=0.16s)
llumnix = [9, 10, 22, 33, 41, 47, 53, 55, 68, 94, 100, 100, 100] #(a good result for SLO=0.25s, 90% reqs were within the SLO for exhaustive.)
# llumnix = [9, 10, 22, 33, 41, 47, 53, 55, 70, 83, 91, 100, 100]
# llumnix = [0.1, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.5, 0.6, 0.7, 1, 1]#prev-wrong-right
# ax.plot(stepX, np.array(static)-5, c = 'blue', marker = 'o', markersize = 26, ls = '-', label = "Scrooge", fillstyle = 'full', linewidth = 4)
ax.plot(stepX, np.array(llumnix)/100, c = color_list[1], ls = '-.', label = "Llumnix", fillstyle = 'none', linewidth = 4)
ax.plot(stepX,np.array(llumnix_plus)/100,c=color_list[0],ls='-',label="Llumnix+SD",fillstyle='none', linewidth = 4)

plt.axvline(x = 0.24, color = 'green', linestyle = '--', linewidth = 2)
plt.axhline(y = 0.9, color = color_list[0], linestyle = '--', linewidth = 2)
plt.axhline(y = 0.595, color = color_list[1], linestyle = '--', linewidth = 2)

ax.set_ylabel('CDF')
ax.set_xlabel('TTFT(s)')
xvalues = [1, 2, 3, 4]
# ax.set_xticks(xvalues)
# ax.set_xticklabels([0.1, 0.2, 0.3, 0.4])
# ax.set_ylim(0,4500)
# ax.set_xticks(ind+4*width)
#ax.set_xticklabels(vals['error_type'].tolist())

# xvalues = [5, 10, 15, 20, 25, 30]
# xvalues = [8, 10, 12]
# plt.xticks(xvalues)
# plt.xticks(xvalues)
# ax.set_xticklabels(["%d%%" % x for x in xvalues], fontsize=36)
# ax.set_xticklabels(["200", "300", "400"])

# plt.yticks([10,20,30,40,50])
# ax.set_yticklabels(["10k","20k","30k","40k","50k"])

ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=2)

# ax.set_ylim(70, 100)

# ytickvalues = []

# ax.set_ylim(75, 101)
# ytickvalues = []
# for i in range(75, 101, 10):
# 	ytickvalues.append(i)
# plt.yticks(ytickvalues)
# ax.set_yticklabels(["%d%%" % x for x in ytickvalues])

# for i in range(70, 105, 10):
# 	ytickvalues.append(i)
# plt.yticks(ytickvalues)
# ax.set_yticklabels(["%d%%" % x for x in ytickvalues], fontsize=36)
# ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('iWash', 'WristWash', 'H2DTR-NN', 'H2DTR-kNN'), loc=1, fontsize=28 )

# ax.legend( (rects1[0], rects2[0], rects3[0]), ('Our System', 'Federated Learning', 'Malicious Device Detection+Trust-aware Reassignment'), loc=1, fontsize=28)

# ax.legend(loc=1)
# plt.title("Categorization of Errors in Critical Cases")
# def autolabel(rects):
#     for rect in rects:
#         h = rect.get_height()
#         ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
#                 ha='center', va='bottom')

# # autolabel(rects1)
# # autolabel(rects2)
# # autolabel(rects3)

# plt.show()

#DRAWING HORIZONTAL CHARTS
# our_lexicon_bangla = [9.8, 1.33]
# rects1 = ax.barh(ind, our_lexicon_bangla, width, color='r', edgecolor='black', hatch=patterns[0])
# our_lexicon_romanized = [9.9, 1.27]
# rects2 = ax.barh(ind+width, our_lexicon_romanized, width, color='g', edgecolor='black', hatch=patterns[1])
# google_lexicon = [14.8, 1.88]
# rects3 = ax.barh(ind+width*2, google_lexicon, width, color='b', edgecolor='black', hatch=patterns[9])

# ax.set_xlabel('Percentage')
# ax.set_yticks(ind+width)
# ax.set_yticklabels( ('WER %', 'PER %') )
# # ax.set_xlim(0,25)
# ax.legend( (rects1[0], rects2[0], rects3[0]), ('Our lexicon Bangla', 'Our lexicon romanized', 'Google lexicon') )

# # def autolabel(rects):
# #     for rect in rects:
# #         h = rect.get_height()
# #         ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
# #                 ha='center', va='bottom')

# # autolabel(rects1)
# # autolabel(rects2)
# # autolabel(rects3)

box = ax.get_position()
#box.height*0.75
#box.y0 + box.height * 0.32
ax.set_position([box.x0 + box.width*0.01, box.y0 + box.height*0.09, box.width, box.height*0.75])
#bbox_to_anchor=(0.5, 1.6)
leg = plt.legend(loc='upper center', bbox_to_anchor=(0.44, 1.5), handler_map={matplotlib.lines.Line2D: SymHandler()}, 
           fontsize='60', ncol=2, handleheight=1.5, labelspacing=0.0, columnspacing=0.4, handletextpad = 0.2, frameon=False) 
# leg = plt.legend(loc='upper center', bbox_to_anchor=(0.3, 1.55), handler_map={matplotlib.lines.Line2D: SymHandler()}, 
            # fontsize='36', ncol=2, handleheight=1.5, labelspacing=0.0, frameon=False)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          # fancybox=True, shadow=True, ncol=5)
# plt.legend(frameon=False)
# leg.get_frame().set_linewidth(0.0)
# fig.tight_layout()
# plt.tight_layout()

# plt.tight_layout()
# # fig.savefig(f'utilization_latency_tradeoff.png', bbox_inches='tight', dpi=500)
# fig.savefig(f'goodput_varyingLoads_fixedCluster_realDeployment_MAF1.pdf', bbox_inches='tight', dpi=500)
# plt.close()

manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

# figure = plt.gcf()  # get current figure
# figure.set_size_inches(50, 30)
# plt.savefig("/home/sudipta/Desktop/image_filename_test.png")

plt.show()