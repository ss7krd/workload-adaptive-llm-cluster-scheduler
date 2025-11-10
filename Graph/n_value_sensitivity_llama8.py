import matplotlib
import numpy as np
import pandas as pd
# extra for mac
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 60})

from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines

matplotlib.rcParams['hatch.linewidth'] = 2

color_list = []
with open('color_pallete_for_obs_bars.txt','r') as color_file:
	for eachLine in color_file:
		color_list.append(eachLine.strip())

class SymHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle,xdescent, ydescent, width, height, fontsize, trans):
        xx= 0.5*height
        return super(SymHandler, self).create_artists(legend, orig_handle,xdescent, xx, width, height, fontsize, trans)

N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.25      # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()

patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

# stepX=[1,2,3,4,5,6]
#DRAWING VERTICAL CHARTS
dummy_1 = [94.1,94.9,96,95.2,94.8]
dummy = [0,0,0,0,0]
dummy_2 = [0.58,0.79,0.97,1.9,2.41] #FINALLY X2 KORECHI.
#detection: cpu, reassignment: memory
# vals_detection = pd.read_csv('cpu_data_until_converge_1.csv')
# vals_reassignment = pd.read_csv('cpu_data_until_converge_1.csv')
# vals_bw = pd.read_csv('cpu_data_until_converge_1.csv')

# vals_eachComp_detection = pd.read_csv('cpu_data_each_comp_1.csv')
# vals_eachComp_reassignment = pd.read_csv('cpu_data_each_comp_1.csv')
# vals_eachComp_bw = pd.read_csv('cpu_data_each_comp_1.csv')

# our_system_detection = np.array(vals_detection['our_system'].tolist())/100
# our_system_reassignment = np.array(vals_reassignment['our_system'].tolist())/100-0.06
# our_system_bw = np.array(vals_bw['our_system'].tolist())/100+0.05

ax.bar(ind,dummy_1,width,color=color_list[0],hatch='+',edgecolor=color_list[0],fill=False,linewidth=2)
# ax.bar(ind,our_system_reassignment,width,color='yellow',edgecolor='black',hatch="O",bottom=our_system_detection)
# ax.bar(ind,our_system_bw,width,color='yellow',edgecolor='black',hatch="-",bottom=our_system_detection+our_system_reassignment)
# ax.bar(ind,dummy,width,color='yellow',edgecolor='black')

ax.bar(ind,dummy,width,color=color_list[0],edgecolor=color_list[0],hatch='+',fill=False,label="SLO\nattainment",linewidth=2)
ax.bar(ind,dummy,width,color=color_list[1],edgecolor=color_list[1],hatch='',fill=True,label="Scheduling\ntime",linewidth=2)
# ax.bar(ind,dummy,width,color='white',edgecolor='black',label="Bandwidth",hatch="-")
# ax.bar(ind,dummy,width,color='yellow',edgecolor='black',label="TrustMe")




ax2.bar(ind+1*width, np.array(dummy_2)*2, width, color = color_list[1], hatch='',edgecolor = color_list[1],fill=True,linewidth=2)
# ax.bar(ind+width,federated_learning_reassignment,width,color='lime',edgecolor='black',hatch="O",bottom=federated_learning_detection)
# ax.bar(ind+width,federated_learning_bw,width,color='lime',edgecolor='black',hatch="-",bottom=federated_learning_detection+federated_learning_reassignment)

ax.set_ylim(90, 100)
# ax2.set_ylim(0.45, 1.45)
# ax2.set_ylim(0.9, 5)

ax.set_yticks([90, 95, 100])
ax.set_yticklabels(["90%", "95%", "100%"])

# ytickvalues = []
# for i in range(90, 101, 3):
# 	ytickvalues.append(i)
# ax.set_yticks(ytickvalues)
# ax.set_yticklabels(["%d%%" % x for x in ytickvalues])

# ytickvalues = []
# for i in range(95, 101, 3):
# 	ytickvalues.append(i)
# ax2.set_yticks(ytickvalues)
# ax2.set_yticklabels(["%d%%" % x for x in ytickvalues])

ax2.set_ylabel('Scheduling\ntime (s)')
ax.set_ylabel('SLO\nattainment')
ax.set_xlabel(r'$|\mathcal{S}_t|$')
ax.set_xticks(ind+0.5*width)

# ax.set_ylim(0,.5)


# ax.set_ylim(0,4500)
# ax.set_xticks(ind+4*width)
#ax.set_xticklabels(vals['error_type'].tolist())

# xvalues = [5, 10, 15, 20, 25, 30]
xvalues = [50, 55, 60, 65, 70]
# plt.xticks(xvalues)
# plt.xticks(xvalues)
ax.set_xticklabels(xvalues)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=2)

ax2.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=2)
# ax.set_ylim(70, 100)
ytickvalues = []

box = ax.get_position()
#box.height*0.75
#box.y0 + box.height * 0.32
ax.set_position([box.x0 + box.width*0.15, box.y0 + box.height*0.1, box.width*0.75, box.height*0.7])
#bbox_to_anchor=(0.5, 1.6)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6), handler_map={matplotlib.lines.Line2D: SymHandler()}, 
            fontsize='60', ncol=2, handleheight=1, labelspacing=0.2, frameon=False) 

manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

plt.show()
