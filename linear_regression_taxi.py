import io 

import numpy as np
import pandas as pd 

import keras

import matplotlib.pyplot as plt

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)

print('total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.describe(include='all')

# # What is the maximum fare?
# max_fare = training_df['FARE'].max()
# print("What is the maximum fare? \t\t\t\tAnswer: ${fare:.2f}".format(fare = max_fare))

# # What is the mean distance across all trips?
# mean_distance = training_df['TRIP_MILES'].mean()
# print("What is the mean distance across all trips? \t\tAnswer: {mean:.4f} miles".format(mean = mean_distance))

# # How many cab companies are in the dataset?
# num_unique_companies =  training_df['COMPANY'].nunique()
# print("How many cab companies are in the dataset? \t\tAnswer: {number}".format(number = num_unique_companies))

# # What is the most frequent payment type?
# most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
# print("What is the most frequent payment type? \t\tAnswer: {type}".format(type = most_freq_payment_type))

# # Are any features missing data?
# missing_values = training_df.isnull().sum().sum()
# print("Are any features missing data? \t\t\t\tAnswer:", "No" if missing_values == 0 else "Yes")

# the correlation matric between the labels 
training_df.corr(numeric_only = True)

# # visualiizing the relationships in dataset
# sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
# # plt.show()


def make_plots(df, feature_names, label_name, model_output, sample_size=200):

  random_sample = df.sample(n=sample_size).copy()
  random_sample.reset_index()
  weights, bias, epochs, rmse = model_output

  is_2d_plot = len(feature_names) == 1
  model_plot_type = "scatter" if is_2d_plot else "surface"
  fig = make_subplots(rows=1, cols=2,
                      subplot_titles=("Loss Curve", "Model Plot"),
                      specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

  plot_data(random_sample, feature_names, label_name, fig)
  plot_model(random_sample, feature_names, weights, bias, fig)
  plot_loss_curve(epochs, rmse, fig)

  fig.show()
  return

def plot_loss_curve(epochs, rmse, fig):
  curve = px.line(x=epochs, y=rmse)
  curve.update_traces(line_color='#ff0000', line_width=3)

  fig.append_trace(curve.data[0], row=1, col=1)
  fig.update_xaxes(title_text="Epoch", row=1, col=1)
  fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

  return

def plot_data(df, features, label, fig):
  if len(features) == 1:
    scatter = px.scatter(df, x=features[0], y=label)
  else:
    scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

  fig.append_trace(scatter.data[0], row=1, col=2)
  if len(features) == 1:
    fig.update_xaxes(title_text=features[0], row=1, col=2)
    fig.update_yaxes(title_text=label, row=1, col=2)
  else:
    fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

  return

def plot_model(df, features, weights, bias, fig):
  df['FARE_PREDICTED'] = bias[0]

  for index, feature in enumerate(features):
    df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

  if len(features) == 1:
    model = px.line(df, x=features[0], y='FARE_PREDICTED')
    model.update_traces(line_color='#ff0000', line_width=3)
  else:
    z_name, y_name = "FARE_PREDICTED", features[1]
    z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
    y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
    x = []
    for i in range(len(y)):
      x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

    plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

    light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
    model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                      colorscale=light_yellow))

  fig.add_trace(model.data[0], row=1, col=2)

  return

def model_info(feature_names, label_name, model_output):
  weights = model_output[0]
  bias = model_output[1]

  nl = "\n"
  header = "-" * 80
  banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header

  info = ""
  equation = label_name + " = "

  for index, feature in enumerate(feature_names):
    info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
    equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

  info = info + "Bias: {:.3f}\n".format(bias[0])
  equation = equation + "{:.3f}\n".format(bias[0])

  return banner + nl + info + nl + equation

print("SUCCESS: defining plotting functions complete.")


def build_model(my_learning_rate , num_features):
  inputs = keras.Input(shape=(num_features,))
  outputs = keras.layers.Dense(units=1)(inputs)
  model = keras.Model(inputs=inputs , outputs = outputs)

  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=my_learning_rate),loss="mean_squared_error",metrics=[keras.metrics.RootMeanSquaredError()])

  return model

def train_model(model, features , label , epochs , batch_size):
  
  history = model.fit(features, label , batch_size = batch_size , epochs=epochs)

  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]


  epochs = history.epoch

  hist = pd.DataFrame(history.history)

  rmse = hist["root_mean_squared_error"]

  return trained_weight , trained_bias , epochs , rmse


def run_experiment(df , features_names , label_name , learning_rate , epochs , batch_size):
  print('INFO: starting training experiment with features={} and label={}\n'.format(features_names,label_name))
  num_features = len(features_names)

  features = df.loc[:, features_names].values
  label = df[label_name].values

  model = build_model(learning_rate , num_features)
  model_output = train_model(model , features , label , epochs , batch_size)

  print('\nSUCCESS: training experiment complete\n')
  print('{}'.format(model_info(features_names, label_name , model_output)))
  make_plots(df , features_names , label_name , model_output)

  return model

print("SUCCESS : defining linear regression function complete.")




# expirement 1

# learning_rate = 0.01
# epochs = 20 
# batch_size  = 50 
 
# features = ['TRIP_MILES']
# label = 'FARE'

# model_1 = run_experiment(training_df , features , label , learning_rate , epochs , batch_size)



# expirement 2
# learning_rate = 0.001
# epochs = 20 
# batch_size  = 50 
 
# features = ['TRIP_MILES']
# label = 'FARE'

# model_1 = run_experiment(training_df , features , label , learning_rate , epochs , batch_size)




# expirement 3
learning_rate = 0.001
epochs = 20 
batch_size  = 50 

training_df.loc[:, 'TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60 
features = ['TRIP_MILES', 'TRIP_MINUTES']
label = 'FARE'

model_2 = run_experiment(training_df , features , label , learning_rate , epochs , batch_size)



def format_currency(x):
  return "${:,.2f}".format(x)

def build_batch(df , batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace = True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))
  

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return


features = ['TRIP_MILES', 'TRIP_MINUTES']
output = predict_fare(model_2, training_df , features , label)
show_predictions(output)