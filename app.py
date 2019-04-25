import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import pydotplus

from dash.dependencies import Input, Output, State

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

def unique_values(df, value):
    return list(dict.fromkeys(df[value]))
access_token = 'YOUR MAPBOX ACCESS TOKEN HERE'
css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load CSV into Dataframe, Data cleansing
sampling = 200000 # Remove sampling to load all data
traning_sample = 60000 # Set how much data is used for training, must not exceed sampling
vehicle_info = pd.read_csv('Vehicle_Information.csv', engine='python', nrows=sampling) # If sampling is removed, remove nrows
accident_info = pd.read_csv('Accident_Information.csv', engine='python',nrows=sampling) # If sampling is removed, remove nrows
df = accident_info.join(vehicle_info, lsuffix='Accident_Index', rsuffix='Accident_index')
df.rename(columns={'Accident_IndexAccident_Index':'Accident_Index', 'YearAccident_index':'Year'}, inplace=True)

# Filter unused columns, remove unknown data
df = df[['Day_of_Week', 'Year', 'Vehicle_Manoeuvre', 'Date', 'Longitude', 'Latitude', 'Light_Conditions', 'Number_of_Vehicles', 'Road_Surface_Conditions', 'Speed_limit', 'Weather_Conditions', 'Accident_Severity', 'Number_of_Casualties', 'Time']]
df = df[df.Light_Conditions != 'Darkness - lighting unknown']
df = df[df.Weather_Conditions != 'Unknown']
df = df[df.Weather_Conditions != 'Other']
df = df[df.Weather_Conditions != 'Data missing or out of range']
df = df[df.Road_Surface_Conditions != 'Data missing or out of range']
df = df[df.Time.notnull()]
df = df[df.Longitude.notnull()]
df = df[df.Latitude.notnull()]
severity_list = df['Accident_Severity']

# replace string to numbers
for c in ['Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Accident_Severity']:
    df[c] = LabelEncoder().fit(df[c]).transform(df[c])

# sample data for learning
learning_data = df[['Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Longitude', 'Latitude', 'Accident_Severity']].sample(traning_sample)
x = learning_data[['Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Longitude', 'Latitude']] 
y = learning_data.Accident_Severity

# create decision tree
decision_tree = DecisionTreeClassifier(min_samples_leaf = 100, min_samples_split = 100)
decision_tree.fit(x, y)

# create decision tree image
dot_data = StringIO()
export_graphviz(decision_tree, out_file = dot_data, filled = True, rounded = True, special_characters = True, feature_names = ['Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Longitude', 'Latitude'], class_names = ['Fatal', 'Serious', 'Slight'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
print("creating image")
graph.write_png('tree.png')

# create confusion matrix
prediction = decision_tree.predict(df[['Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Longitude', 'Latitude']])
cm = confusion_matrix(df.Accident_Severity.tolist(), prediction) # true value, predicted value
accuracy = (cm[0][0] + cm[1][1] + cm[2][2]) / df.shape[0] # amount of correct predictions divided by the amount of data (rows)

app = dash.Dash(external_stylesheets = css)
app.config['suppress_callback_exceptions'] = True

# contents of the web app
app.layout = html.Div(children=[
    html.H1("UK Traffic Accident Severity Predictor"),
    dcc.Tabs(id = "tabs", value='tab-1', children=[
        dcc.Tab(label='Decision Tree Statistics',
                value='tab-1'),
        dcc.Tab(label='Accident Map',
                value='tab-2',),
    ]),
    html.Div(id='tabs-content')
])

# Value updater
@app.callback(Output('longitude_output', 'children'),
              [Input('lon', 'value')])
def update_output(value):
    return html.P("Longitude is set to " + str(value))
@app.callback(Output('latitude_output', 'children'),
              [Input('lat', 'value')])
def update_output(value):
    return html.P("Latitude is set to " + str(value))

# Real time map updating
@app.callback(Output('map', 'figure'),
              [Input('lon', 'value'),
               Input('lat', 'value')],)
def update_map(value, value2):
    return {
        "data": [
            go.Scattermapbox(
                            lon = [value],
                            lat = [value2],
                            text = ["Current location"],
                            mode = 'markers',
                            marker=go.scattermapbox.Marker(size=15, color = 'rgb(66, 129, 255)'),
                            name = 'Current location'
                        ),
            go.Scattermapbox(
                            lon = df.Longitude[:2000],
                            lat = df.Latitude[:2000],
                            name = 'Recorded accident',
                            text = severity_list[:2000]
                        )],
        "layout": go.Layout(
                        title = 'Accident Location',
                        width = 900,
                        height = 600,
                        mapbox = go.layout.Mapbox(
                            accesstoken = access_token,
                            center = go.layout.mapbox.Center(
                                lat = value2,
                                lon = value,
                            ),
                            style="streets",
                            zoom = 12
                        )
                    )
    }

# Tab renderer
@app.callback(Output('tabs-content', 'children'),
            [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Graph(
                figure = go.Figure(
                    data = [go.Heatmap(
                        z = cm,
                        x = ['Predicted Fatal', 'Predicted Serious', 'Predicted Slight'],
                        y = ['Actual Fatal', 'Actual Serious', 'Actual Slight']
                    )]
                )
            ),
            html.P("Number of correct predictions is " + str(cm[0][0] + cm[1][1] + cm[2][2])),
            html.P("Accuracy of decision tree is " + str(accuracy))
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.H1("Decision Tree Predicting"),
                html.P("Using a decision tree, we can roughly predict the severity of an accident in a certain area, given road, weather and light conditions as well as the coordinates."),
                html.P("We cannot detect if the coordinates point to a road!"),
                html.P("We can only ask you to specify Longitude and Latitude. Your coordinates will be displayed on the map."),
                html.H2("Longitude"),
                dcc.Slider(
                    id='lon',
                    min = df.Longitude.min(),
                    max = df.Longitude.max(),
                    step = 0.0001,
                    value = -0.21
                ),
                html.Div(id='longitude_output'),
                html.H2("Latitude"),
                dcc.Slider(
                    id='lat',
                    min = df.Latitude.min(),
                    max = df.Latitude.max(),
                    step = 0.0001,
                    value = 51.5
                ),
                html.Div(id='latitude_output'),
                dcc.Dropdown(
                    id='Lighting',
                    options=[
                        {'label': 'Daylight', 'value': 3},
                        {'label': 'Darkness with lights', 'value': 0},
                        {'label': 'Darkness with not working lights', 'value': 1},
                        {'label': 'Darkness without lights', 'value': 2},
                    ],
                    placeholder='Set lighting condition',
                    value='lighting_value'
                ),
                dcc.Dropdown(
                    id='Weather',
                    options=[
                        {'label': 'Raining', 'value': 4},
                        {'label': 'Clear', 'value': 1},
                        {'label': 'Snowing', 'value': 6},
                        {'label': 'Wind', 'value': 0},
                        {'label': 'Rain + Wind', 'value': 3},
                        {'label': 'Fog/Mist', 'value': 2},
                        {'label': 'Snow + Wind', 'value': 5},
                    ],
                    placeholder='Set weather condition',
                    value='weather_value'
                ),
                dcc.Dropdown(
                    id='Road',
                    options=[
                        {'label': 'Wet', 'value': 4},
                        {'label': 'Dry', 'value': 0},
                        {'label': 'Frost', 'value':2},
                        {'label': 'Snow', 'value': 3},
                        {'label': 'Flood over 3cm', 'value': 1},
                    ],
                    placeholder='Set road condition',
                    value='road_value'
                ),
            ], style={'float': 'left', 'width': '50%'}),
            html.Div([
                dcc.Graph(id='map'),
                html.Button('Predict accident severity', id='button'),
                html.Div(id='prediction_output')
            ], style={'float': 'right'}),
        ])

# Take arguments and use predict
@app.callback(
    Output('prediction_output', 'children'),
    [Input('button', 'n_clicks')],
    [State('Lighting', 'value'),
    State('Weather', 'value'),
    State('Road', 'value'),
    State('lon', 'value'),
    State('lat', 'value')],
)
def update_output(*args):
    values = list(args)
    values.pop(0)
    text_out = None
    
    try:
        ans = decision_tree.predict([values])
        if ans[0] == 2:
            text_out = "Accident will be slight"
        elif ans[0] == 1:
            text_out = "Accident will be serious"
        else:
            text_out = "Accident will be fatal"
    except:
        text_out = ""
    return html.P(text_out)

if __name__ == '__main__':
    app.run_server(debug=True)
