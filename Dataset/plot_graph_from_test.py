import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette='Set2')
# save the metrics for the run to a csv file
metrics_dataframe = pd.read_csv('./evaluation_GAN.csv')

ax = plt.gca()
metrics_dataframe = metrics_dataframe.sort_values(by='score_threshold')
metrics_dataframe[['score_threshold', 'map_person', 'map_cars', 'map_cyclist']].set_index('score_threshold').plot.bar(ax = ax)
plt.legend(ncol=1, loc='upper right');
ax.yaxis.grid(True) 
ax.xaxis.grid(False)
ax.set_xlabel('Soglia')
ax.set_ylabel('Mean Average Precision')
ax.margins()
ax.set_ylim(bottom=0)
sns.despine()
#plt.show()
plt.savefig('./graphics_map.pdf', format='pdf', bbox_inches='tight')