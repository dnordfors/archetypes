import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

input_data = pd.read_csv('Final.csv')
data = []
for i in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]:
    for j in range(24):
        data.append([i,j])
data = pd.DataFrame(data,columns=['weekday','hour'])


POS = input_data['POS'].unique()
POS_default = input_data['POS'][0]
Menu_default = input_data['MenuItem'][0]
df_default = input_data[(input_data['POS'] == POS_default) & (input_data['MenuItem'] == Menu_default)]


app.layout = html.Div([
	html.H1('Food and Beverages - Day Part Analytics'),
    html.Div([
    	
    	html.Div([
    		html.H4('Select POS'),
            dcc.Dropdown(
                id='POS_dropdown',
                options=[{'label': i, 'value': i} for i in POS],
                value = POS_default
             
            ),
          
        ],
        style={'width': '48%', 'display': 'inline-block'}),
    	
        html.Div([
        	html.H4('Select MenuItem'),
            dcc.Dropdown(
                id='Menu_dropdown',
    			value = Menu_default
            ),
        
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    	dcc.Graph(id='heatmap',  
    	figure = {
    		'data': [go.Heatmap(
                x=df_default['weekday'],
                y=df_default['hour'],
                z=df_default['Qty'],
                name = 'first legend group',
                
                colorscale='Viridis')],
        	'layout': go.Layout(
        		xaxis = dict(title = 'Weekday'),
              	yaxis = dict( title = 'Hours'),
              )

    	})
    ]),

  
])
@app.callback(
    dash.dependencies.Output(component_id='Menu_dropdown',component_property='options'),
    [dash.dependencies.Input(component_id='POS_dropdown',component_property='value')]
)


def update_Menu_dropdown(selected_POS):
    return [{'label': i, 'value': i} for i in input_data[input_data['POS'] == selected_POS]['MenuItem'].unique()]

@app.callback(
    dash.dependencies.Output(component_id='heatmap',component_property='figure'),
    [dash.dependencies.Input(component_id='POS_dropdown',component_property='value'),
	 dash.dependencies.Input(component_id='Menu_dropdown',component_property='value')]
)

def update_graph(POS_dropdown,Menu_dropdown):
    heatmap_data = input_data[(input_data['POS'] == POS_dropdown) & (input_data['MenuItem'] == Menu_dropdown)][['weekday','hour','Qty']]
    heatmap_data = pd.merge(data, heatmap_data, on=['weekday', 'hour'],how='outer').fillna(0)
    print (POS_dropdown,Menu_dropdown)
    maxsale = heatmap_data[heatmap_data['Qty']==heatmap_data['Qty'].max()]
    maxsale = maxsale.reset_index()
    print(maxsale)
    return {
        'data': [go.Heatmap(
                x=heatmap_data['weekday'],
                y=heatmap_data['hour'],
                z=heatmap_data['Qty'],
                xgap = 2,
  				ygap = 2,
                colorscale='Viridis')],
        'layout': go.Layout(
             	title = 'MAJORITY OF '+Menu_dropdown+' SOLD AT '+str.upper(POS_dropdown)+' IS ON '+ str.upper(maxsale['weekday'][0])+' '+str(maxsale['hour'][0])
            )
        
    }

   

if __name__ == '__main__':
    app.run_server(debug=True)