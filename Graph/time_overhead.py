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

N = 4
ind = np.arange(N)  # the x locations for the groups
ind = np.array([0, 1.5, 3])
width = 0.25      # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
# ax2 = ax.twinx()

patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

# stepX=[1,2,3,4,5,6]
#DRAWING VERTICAL CHARTS
predictor = [0.051,0.053,0.053]
PDLD = [0.9,0.91,0.91]
SD = [1,1.1,1.1]
dummy = [0,0,0]
execution_time = [3.2,4.1,5.4]
on_demand = [0.12, 0.11, 0.11]
communication = [0.022,0.03,0.028]
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
ax.bar(ind,predictor,width,color='white',hatch='O',edgecolor='black',fill=False,linewidth=2)
ax.bar(ind,PDLD,width,bottom=predictor,color='white',hatch='+',edgecolor='black',fill=False,linewidth=2)
#STACKING:
ax.bar(ind,SD,width,bottom=np.array(predictor)+np.array(PDLD), color='white',hatch='*',edgecolor='black',fill=False,linewidth=2)
# ax.bar(ind,our_system_reassignment,width,color='yellow',edgecolor='black',hatch="O",bottom=our_system_detection)
# ax.bar(ind,our_system_bw,width,color='yellow',edgecolor='black',hatch="-",bottom=our_system_detection+our_system_reassignment)
# ax.bar(ind,dummy,width,color='yellow',edgecolor='black')

ax.bar(ind,dummy,width,color='white',edgecolor='black',hatch='O',fill=False,label="DLP",linewidth=2)
ax.bar(ind,dummy,width,color='white',edgecolor='black',hatch='+',fill=False,label="PDLD",linewidth=2)
ax.bar(ind,dummy,width,color='white',edgecolor='black',hatch='*',fill=False,label="SD",linewidth=2)
ax.bar(ind,dummy,width,color=color_list[2],edgecolor=color_list[2],hatch='',fill=True,label="Execution",linewidth=2)
ax.bar(ind,dummy,width,color='green',edgecolor='green',hatch='/',fill=False,label="OR",linewidth=2)
ax.bar(ind,dummy,width,color='white',edgecolor='blue',hatch='',fill=False,label="Comm. downtime",linewidth=2)
# ax.bar(ind,dummy,width,color='white',edgecolor='black',label="Bandwidth",hatch="-")
# ax.bar(ind,dummy,width,color='yellow',edgecolor='black',label="TrustMe")




ax.bar(ind+1*width, execution_time, width, color = color_list[2], hatch='',edgecolor = color_list[2],fill=True,linewidth=2)
ax.bar(ind+2*width, on_demand, width, color = 'green', hatch='/',edgecolor = 'green',fill=False,linewidth=2)
ax.bar(ind+3*width, communication, width, color = 'white', hatch='',edgecolor = 'blue',fill=False,linewidth=2)
# ax.bar(ind+width,federated_learning_reassignment,width,color='lime',edgecolor='black',hatch="O",bottom=federated_learning_detection)
# ax.bar(ind+width,federated_learning_bw,width,color='lime',edgecolor='black',hatch="-",bottom=federated_learning_detection+federated_learning_reassignment)

# ax.set_ylim(95, 100)
# ax2.set_ylim(95, 100)

# ytickvalues = []
# for i in range(95, 101, 2):
# 	ytickvalues.append(i)
# ax.set_yticks(ytickvalues)
# ax.set_yticklabels(["%d%%" % x for x in ytickvalues])

# ytickvalues = []
# for i in range(95, 101, 3):
# 	ytickvalues.append(i)
# ax2.set_yticks(ytickvalues)
# ax2.set_yticklabels(["%d%%" % x for x in ytickvalues])

ax.set_ylabel('Time (s)')
# ax2.set_ylabel('Time (ms)')
# ax.set_xlabel(r'$\tau$ (ms)')
ax.set_xticks(ind+width)

# ax.set_ylim(0,.5)


# ax.set_ylim(0,4500)
# ax.set_xticks(ind+4*width)
#ax.set_xticklabels(vals['error_type'].tolist())

# xvalues = [5, 10, 15, 20, 25, 30]
# xvalues = [0.0006, 0.0008, 0.001, 0.0012, 0.0014]
# plt.xticks(xvalues)
# plt.xticks(xvalues)
system_names = ["Llama3-8B", "Mixtral", "Llama3-70B"]
ax.set_xticklabels(system_names, rotation=30)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=2)

# ax2.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=2)
# ax.set_ylim(70, 100)
ytickvalues = []

box = ax.get_position()
#box.height*0.75
#box.y0 + box.height * 0.32
ax.set_position([box.x0 + box.width*0.05, box.y0 + box.height*0.25, box.width, box.height*0.50])
#bbox_to_anchor=(0.5, 1.6)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.95), handler_map={matplotlib.lines.Line2D: SymHandler()}, 
            fontsize='60', ncol=2, handleheight=1, labelspacing=0.2, frameon=False) 

manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

plt.show()
