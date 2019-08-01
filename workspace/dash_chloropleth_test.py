#%%
import plotly.graph_objects as go

# Load data frame and tidy it.
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

#%%

df0 = xy.archetypes(4).fn
orientation = {'arch' : 'right', 'other' : 'bottom'}
df = df0.set_index(df0.idxmax(axis = 1).map(lambda string: string + ' Archetype'))
dendro_arch = ff.create_dendrogram(df, orientation= orientation['arch'], labels=df.index)
dendro_arch_leaves = dendro_arch['layout']['yaxis']['ticktext']
dendro_other = ff.create_dendrogram(df.T, orientation= orientation['other'], labels=df.T.index)
dendro_other_leaves = dendro_other['layout']['xaxis']['ticktext']
clustered_df = df[dendro_other_leaves].loc[dendro_arch_leaves]
clustermap = go.Heatmap(z = clustered_df.values,
                        y = clustered_df.index,
                        x = clustered_df.columns)
fig = go.Figure(clustermap)
fig.show()