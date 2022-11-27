########## Library ##########
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import pandas as pd, numpy as np
import base64, io
from io import BytesIO
import dash_daq as daq
import plotly.express as px, plotly.graph_objects as go, plotly.figure_factory as ff
from plotly.subplots import make_subplots
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import more_itertools


########## Data loading ##########
df_pred = pd.read_csv("/home/geouser05/geo/data/07_model_output/prediction_dts_full2.csv")
filenames = list(df_pred['filename'].unique())
has_dts = [s.replace('/', '_').replace(' ', '_') + '_logs.las' for s in [
        '16/10-3', '16/10-5', '16/2-11 A', '16/2-16', '16/2-6', '16/5-3', '25/10-10', '25/11-24',
        '25/6-3', '25/8-5 S', '31/2-19 S', '31/3-4', '32/2-1', '33/6-3 S', '34/10-35', '34/11-1',
        '34/12-1', '34/3-1 A', '34/3-3 A', '34/4-10 R', '34/5-1 A', '34/5-1 S', '34/6-1 S', '35/11-10',
        '35/11-11', '35/11-12', '35/11-13', '35/11-15 S', '35/11-6', '35/3-7 S', '35/4-1', '35/6-2 S',
        '35/8-6 S', '35/9-10 S', '35/9-2', '35/9-5', '35/9-8', '7/1-2 S']]
no_dts = [name for name in filenames if name not in has_dts]

lithology_numbers = {30000: {'lith':'Sandstone', 'lith_num':1, 'hatch': '..', 'color':'#ffff00'},
                 65030: {'lith':'Sandstone/Shale', 'lith_num':2, 'hatch':'-.', 'color':'#ffe119'},
                 65000: {'lith':'Shale', 'lith_num':3, 'hatch':'--', 'color':'#bebebe'},
                 80000: {'lith':'Marl', 'lith_num':4, 'hatch':'', 'color':'#7cfc00'},
                 74000: {'lith':'Dolomite', 'lith_num':5, 'hatch':'-/', 'color':'#8080ff'},
                 70000: {'lith':'Limestone', 'lith_num':6, 'hatch':'+', 'color':'#80ffff'},
                 70032: {'lith':'Chalk', 'lith_num':7, 'hatch':'..', 'color':'#80ffff'},
                 88000: {'lith':'Halite', 'lith_num':8, 'hatch':'x', 'color':'#7ddfbe'},
                 86000: {'lith':'Anhydrite', 'lith_num':9, 'hatch':'', 'color':'#ff80ff'},
                 99000: {'lith':'Tuff', 'lith_num':10, 'hatch':'||', 'color':'#ff8c00'},
                 90000: {'lith':'Coal', 'lith_num':11, 'hatch':'', 'color':'black'},
                 93000: {'lith':'Basement', 'lith_num':12, 'hatch':'-|', 'color':'#ef138a'}}
df_lith = pd.DataFrame.from_dict(lithology_numbers, orient='index')
df_lith.index.name = "FORCE_2020_LITHOFACIES_LITHOLOGY"

df_pred["lith"] = df_pred["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(df_lith["lith"])
df_pred["lith_hatch"] = df_pred["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(df_lith["hatch"])
df_pred["lith_color"] = df_pred["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(df_lith["color"])

list_filename = df_pred.dropna(subset= ["x_loc", "y_loc", "z_loc"])["filename"].unique()
df_field_plot = pd.DataFrame(columns= ["x_loc", "y_loc", "z_loc", "field", "filename"])

for names in list_filename:
    tmp = df_pred[df_pred.filename == names].describe().iloc[1][["x_loc", "y_loc", "z_loc"]]
    tmp["field"] = df_pred[df_pred.filename == names].field.unique()[0]
    tmp["filename"] = names
    tmp = pd.DataFrame(tmp).T.reset_index(drop= True)
    df_field_plot = pd.concat([df_field_plot, tmp])

tmp_df = df_pred.groupby('filename')['DTS'].agg({'count'}).reset_index()
tmp_df["dts_available"] = np.where(tmp_df["count"] > 0, "Available", "Not-available")
df_field_plot = df_field_plot.merge(tmp_df[["filename", "dts_available"]], on= "filename", how= "left")

df_field_plot.filename.values

########## Create image ##########
image_filename = "/home/geouser05/geo/data/01_raw_data/Team5_banner_resized.png"
team5_logo = base64.b64encode(open(image_filename, 'rb').read())

########## Generate overview data table ##########
def generate_overview(field):
    if field == 'All':
        df = df_pred.copy()
    else:
        df = df_pred[df_pred['field'] == field]

    # df = df[df["lith"].str.contains("nan") == False]
    most_lith = Counter(df[df["lith"].str.contains("nan") == False].lith).most_common(1)[0][0]

    firstcol = ['Well log data', 'Formations available']
    seccol   = [df.filename.nunique(), df.lith.nunique()]
    thirdcol = [Counter(df.filename).most_common(1)[0][0], most_lith,]
    dfout = pd.DataFrame({
        'Info': firstcol,
        'Count': seccol,
        'Max entries': thirdcol
        })
    
    return(dfout)

########## Header & Borders ##########
app = JupyterDash(__name__)

app = JupyterDash(__name__)

app.layout = html.Div([

    ##### Header #####
    html.Div([
        html.Img(src= 'data:image/png;base64,{}'.format(team5_logo.decode()), style= {'height':'12%', 'width':'12%', 'display': 'inline-block', 'margin-left': '10px', 'float': 'left'}),
        html.H1('Well Inlog Fill', style= {'display': 'inline-block', 'margin-left': '15px', 'color': 'white', 'textAlign': 'center', 'width' : '80%'}),
        ##### Night mode #####
        html.Div([daq.ToggleSwitch(
            id = 'night-mode',
            label ='Night mode',
            value = False,
            labelPosition = 'bottom')], style= {'display': 'inline-block', 'float': 'right', 'margin-top': '5px', 'margin-right': '5px', 'color': 'white'})
    ], style={'backgroundColor': 'rgb(46, 64, 83)', 'width': '100%'}),                     

    ##### Main body #####
    html.Div(id = 'background-selection'),

    ##### End #####
    html.H1('Developed by Team 5: Paul Yew, Agnes Lee, Hailey Thai Yuan Jiun, Anselm Adrian, Aini Mokhdhari. ACGPE GeoHackathon Nov 2022', style= {'textAlign': 'right', 'width': '100%', 'font-size': '10px', 'color': '#FFFFFF'})
], style = {'backgroundColor' : 'rgb(46, 64, 83)'})


light_bg = html.Div([           
    html.Div([
        html.Div([html.Label('Field selection'),], style={'font-size': '15px', 'margin-top': '30px'}),
        dcc.Loading(
            type = 'default',
            children = dcc.Dropdown(
              id='field_select',
              options=["Poseidon", "Force"],
              value='Force',
              clearable= True
          ))], style={'width': '35%', 'margin-bottom': '5px', 'margin-left': '30px', 'margin-top' : '15px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='field_plot',
                    clickData={'points': [{'customdata': df_pred.filename[0]}]},
                ))], style={'width': '65%', 'height':'50%', 'display': 'inline-block', 'margin-right': '15px', 'margin-left': '15px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='data_overview'
                ))], style= {'width': '30%', 'height':'100%', 'display': 'inline-block', 'padding': '50', 'margin-bottom': '30px', 'position': 'center'}),
        html.Div([
            dcc.RadioItems(
                id= 'dts_legend_checklist', 
                options=[
                    {'label': html.Div(['Hide DTS legend'], style={'color': 'black', 'display': 'inline-block'}), 'value': 'Hide DTS legend'},
                    {'label': html.Div(['Show DTS legend'], style={'color': 'black', 'display': 'inline-block'}), 'value': 'Show DTS legend'},
                    ], 
                    value= 'Hide_DTS_legend'),
            ], style = {'position': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='well_log_plot',
                )),], style={'width': '40%', 'height':'120%', 'display': 'inline-block'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = [
                    dcc.Graph(id='dts_pred_plot')
                    ]),], style={'width': '25%', 'height':'120%', 'display': 'inline-block', 'margin-right': '15px', 'margin-left': '30px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='lithography_plot'
                )),], style={'width': '10%', 'height':'80%', 'display': 'inline-block', 'margin-left': '60px', 'margin-top': '10px'}),
        html.Div(style={'margin-left': '60px', 'margin-top': '10px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='lithography_legend'
                )),], style={'width': '100%', 'height':'60%', 'margin-left': '60px'}),
], style= {
    'background-image': 'url("https://img.freepik.com/free-vector/yellow-background-with-dynamic-abstract-shapes_1393-144.jpg?w=1060&t=st=1669463501~exp=1669464101~hmac=958ff1ce5c74d4931165cd01a3509b2ea5ce4562c5333c379ba0c65aa1f4c32c")',
    'background-position': 'center-top',
    'background-attachment': 'scroll',
    'margin-top' : '20px'})


dark_bg = html.Div([            
    html.Div([
        html.Div([html.Label('Field selection'),], style={'font-size': '15px', 'color': 'white'}),
        dcc.Loading(
            type = 'default',
            children = dcc.Dropdown(
                id='field_select',
                options=["Poseidon", "Force"],
                value='Force',
                clearable= False
        ))], style={'width': '35%', 'margin-bottom': '5px', 'margin-left': '30px', 'margin-top' : '30px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='field_plot',
                    clickData={'points': [{'customdata': df_pred.filename[0]}]},
            ))], style={'width': '65%', 'height':'50%', 'display': 'inline-block', 'margin-right': '15px', 'margin-left': '15px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='data_overview'
            ))], style= {'width': '30%', 'height':'80%', 'display': 'inline-block', 'padding': '50', 'margin-bottom': '20px', 'position': 'center'}),
        html.Div([
            dcc.RadioItems(
                id= 'dts_legend_checklist', 
                options=[
                    {'label': html.Div(['Hide DTS legend'], style={'color': 'white', 'display': 'inline-block'}), 'value': 'Hide DTS legend'},
                    {'label': html.Div(['Show DTS legend'], style={'color': 'white', 'display': 'inline-block'}), 'value': 'Show DTS legend'},
                    ], 
                    value= 'Hide_DTS_legend'),
            ], style = {'position': 'center', 'margin-top': '20px', 'margin-bottom': '20px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='well_log_plot',
                )),], style={'width': '40%', 'height':'120%', 'display': 'inline-block'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = [
                    dcc.Graph(id='dts_pred_plot')
                    ]),], style={'width': '25%', 'height':'120%', 'display': 'inline-block', 'margin-right': '15px', 'margin-left': '15px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='lithography_plot',
                )),], style={'width': '10%', 'height':'80%', 'display': 'inline-block', 'margin-left': '30px'}),
        html.Div(style={'margin-left': '60px', 'margin-top': '10px'}),
        html.Div([
            dcc.Loading(
                type = 'default',
                children = dcc.Graph(
                    id='lithography_legend'
                )),], style={'width': '10%', 'height':'80%', 'margin-left': '60px', 'margin-top': '10px'}),
], style= {
    'background-image': 'url("https://images.unsplash.com/photo-1517999144091-3d9dca6d1e43?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2127&q=80")',
    'background-position': 'center',
    'background-attachment': 'scroll',
    'margin-top' : '20px'})

@app.callback(
    dash.dependencies.Output('background-selection', 'children'),
    [
        dash.dependencies.Input('night-mode', 'value')
    ]
)
def bg_page(selection):
  if selection == True:
    return(dark_bg)
  else:
    return(light_bg)

########## Field plot ##########
@app.callback(
    dash.dependencies.Output('field_plot', 'figure'),
    [
      dash.dependencies.Input('field_select', 'value'),
      dash.dependencies.Input('night-mode', 'value')
    ]
)
def update_graph(field, mode):
      df = df_field_plot[df_field_plot['field'] == field]
      hover_names = [f"Filename: {s}" for s in df["filename"]]
      
      fig = px.scatter(
        df,
        x= "x_loc", y= "y_loc",
        opacity= 0.8,
        hover_name= hover_names,
        color = "dts_available"
        )
      fig.update_traces(customdata= df.filename.values)
      fig.update_layout(
        coloraxis_colorbar= {'title': f'{hover_names}'},
        legend_title_text= 'DTS data availability',
        height= 350,
        margin= {'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest',
        paper_bgcolor= 'white' if mode == False else 'rgb(46, 64, 83)',
        plot_bgcolor=  'white' if mode == False else 'rgb(46, 64, 83)',
        template= 'plotly_dark' if mode == True else 'none')
      return fig

########## Data overview ##########
@app.callback(
    dash.dependencies.Output('data_overview', 'figure'),
    [
        dash.dependencies.Input('field_select', 'value'),
        dash.dependencies.Input('night-mode', 'value')
    ]
)
def update_data_overview(field, mode):
  df = generate_overview(field)
  if mode == False:
    fig = ff.create_table(df.round(3))
  else:
    fig = ff.create_table(df.round(3), colorscale= [[0, '#17202A'],[.5, '#34495E'],[1, '#212F3D']], font_colors= ['#FFFFFF'])
  return(fig)

########## Well log visualization ##########
@app.callback(
    dash.dependencies.Output('well_log_plot', 'figure'),
    [
            dash.dependencies.Input('field_plot', 'clickData'),
            dash.dependencies.Input('night-mode', 'value')
    ]
)
def update_well_log_plot(clickData, mode):
    indexclick = clickData['points'][0]['customdata']
    df_click = df_pred[df_pred.filename == indexclick]
    y_axis = "z_loc"
    
    logplot = make_subplots(rows=1, cols=3, shared_yaxes=True)

    logplot.add_trace(go.Scatter(x = df_click["GR"], y = df_click[y_axis], name = "Gamma", line_color = "green"), row= 1, col = 1)
    logplot.update_xaxes(row = 1, col= 1, title_text = "Gamma", tickfont_size = 12)

    logplot.add_trace(go.Scatter(x = df_click["NPHI"], y = df_click[y_axis], opacity=0.5, name = "NPHI", line_color = "cyan"), row= 1, col = 2)
    logplot.add_trace(go.Scatter(x = df_click["RHOB"], y = df_click[y_axis], opacity=0.5, name = "RHOB", line_color = "grey"), row= 1, col = 2)
    logplot.update_xaxes(col = 2, title_text = "DENSITY_NEUTRON")

    logplot.add_trace(go.Scatter(x = df_click["DTC"], y = df_click[y_axis], name = "DTC", line_color = "red"), row= 1, col = 3)
    logplot.update_xaxes(row = 1, col= 3, title_text = "DTC", tickfont_size = 12)

    
    logplot.update_layout(
        title_text = "Well log data for " + indexclick,
        title_font_color = 'black' if mode == False else 'white',
        paper_bgcolor = 'white' if mode == False else 'rgb(46, 64, 83)',
        plot_bgcolor =  'white' if mode == False else 'rgb(46, 64, 83)',
        template = 'plotly_dark' if mode == True else 'none',
        margin=dict(l=50, r=50, t=100, b=100)
    )

    return logplot


@app.callback(
    dash.dependencies.Output('lithography_plot', 'figure'),
    [
            dash.dependencies.Input('field_plot', 'clickData'),
            dash.dependencies.Input('night-mode', 'value')
    ]
)
def update_lithography_plot(clickData, mode):
    indexclick = clickData['points'][0]['customdata']
    df_click = df_pred[df_pred.filename == indexclick]

    fig, ax4 = plt.subplots(figsize=(10,30))
    ax4.plot(df_click["FORCE_2020_LITHOFACIES_LITHOLOGY"], df_click['z_loc'], color = "black", linewidth = 0.5)
    ax4.set_xlabel("Lithology")
    ax4.set_xlim(0, 1)
    ax4.xaxis.label.set_color("black")
    ax4.tick_params(axis='x', colors="black")
    ax4.spines["top"].set_edgecolor("black")

    for key in lithology_numbers.keys():
        color = lithology_numbers[key]['color']
        hatch = lithology_numbers[key]['hatch']
        ax4.fill_betweenx(df_click['z_loc'], 0, df_click['FORCE_2020_LITHOFACIES_LITHOLOGY'], where=(df_click['FORCE_2020_LITHOFACIES_LITHOLOGY']==key),
                            facecolor=color, hatch=hatch)
    
    ax4.set_xticks([0, 1])

    path = "/home/geouser05/geo/data/tmp.jpg"
    plt.savefig(path, format = "jpg")
    img = path.replace('data:image/png;base64,', '')
    img = np.array(Image.open(path))
    fig_final = px.imshow(img, title= "Lithology view of "+indexclick).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
    fig_final.update_layout(width = 500, height = 500,
        paper_bgcolor = 'white' if mode == False else 'rgb(46, 64, 83)',
        plot_bgcolor =  'white' if mode == False else 'rgb(46, 64, 83)',
        template = 'plotly_dark' if mode == True else 'none',
        margin=dict(l=0, r=0, t=30, b=0))
        
    return fig_final


@app.callback(
    dash.dependencies.Output('dts_pred_plot', 'figure'),
    [
            dash.dependencies.Input('field_plot', 'clickData'),
            dash.dependencies.Input('dts_legend_checklist', 'value'),
            dash.dependencies.Input('night-mode', 'value')
    ]
)
def update_dts_plot(clickData, check_yes, mode):
    indexclick = clickData['points'][0]['customdata']
    well = df_pred[df_pred.filename == indexclick]
    indices = well.index[well['DTS_BOOL_PRED'].diff().fillna(False)]
    pairs = more_itertools.chunked(indices, 2)
    fig = px.scatter(well, x=['DTS', 'DTS_PRED'], y='z_loc', width=500)
    fig.update_traces(marker={'size': 2})    

    if check_yes == "Show DTS legend":
        fig.add_trace(go.Scatter(
            x=[250],
            y=[1500],
            fill='tonexty', # fill area between trace0 and trace1
            mode='lines',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line_width=0,
            name='Within Train Domain'
            ))
        for s, e in pairs:
            fig.add_hrect(y0=well['z_loc'][s], y1=well['z_loc'][e], line_width=0, fillcolor="rgba(0, 255, 0, 0.2)")
    
    fig.add_trace(go.Scatter(x=well['DTS_PRED_025'], y=well['z_loc'],
    # fig.add_trace(go.Scatter(x=well.DTS_PRED_05.ewm(span=span).mean(), y=well.DEPT,
        fill=None,
        mode='lines',
        line_width=0,
        showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=well['DTS_PRED_975'],
        y=well['z_loc'],
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line_width=0,
        name='95% Pred Interval'
        ))
    fig.update_layout(
        title = f'DTS prediction for {indexclick}',
        title_font_color = 'black' if mode == False else 'white',
        paper_bgcolor = 'white' if mode == False else 'rgb(46, 64, 83)',
        plot_bgcolor =  'white' if mode == False else 'rgb(46, 64, 83)',
        template = 'plotly_dark' if mode == True else 'none',
        margin=dict(l=30, r=0, t=100, b=100)
    )

    return fig

@app.callback(
    dash.dependencies.Output('lithography_legend', 'figure'),
    [
            dash.dependencies.Input('night-mode', 'value')
    ]
)
def view_lith_legend(mode):
    path = "/home/geouser05/geo/data/lith_legend.jpg"
    img = path.replace('data:image/png;base64,', '')
    img = np.array(Image.open(path))
    fig_final = px.imshow(img, title= "Lithology legend").update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
    fig_final.update_layout(width = 1000, height = 800,
        paper_bgcolor = 'white' if mode == False else 'rgb(46, 64, 83)',
        plot_bgcolor =  'white' if mode == False else 'rgb(46, 64, 83)',
        template = 'plotly_dark' if mode == True else 'none')
    return fig_final

########## Run server ##########
app.run_server(host="0.0.0.0", port="8000")