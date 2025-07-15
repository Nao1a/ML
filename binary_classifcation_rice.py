import io
import keras
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px


pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

rice_dataset = rice_dataset_raw[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class',
]]

rice_dataset.describe()


# for x_axis_data, y_axis_data in [
#     ('Area', 'Eccentricity'),
#     ('Convex_Area', 'Perimeter'),
#     ('Major_Axis_Length', 'Minor_Axis_Length'),
#     ('Perimeter', 'Extent'),
#     ('Eccentricity', 'Major_Axis_Length'),
# ]:
#   px.scatter(rice_dataset, x=x_axis_data, y=y_axis_data, color='Class').show()


# px.scatter_3d(
#     rice_dataset,
#     x='Eccentricity',
#     y='Area',
#     z='Major_Axis_Length',
#     color='Class',
# ).show()

feature_mean = rice_dataset.mean(numeric_only = True)
feature_std = rice_dataset.std(numeric_only=True)

numerical_features = rice_dataset.select_dtypes('number').columns
normalized_dataset = (
    rice_dataset[numerical_features] - feature_mean
) / feature_std

normalized_dataset['Class'] = rice_dataset['Class']

normalized_dataset.head

keras.utils.set_random_seed(42)


normalized_dataset['Class_Bool'] = (
    normalized_dataset['Class'] == 'Cammeo'
).astype(int)
normalized_dataset.sample(10)


number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th =index_80th + round(number_samples * 0.1)

shuffeld_dataset = normalized_dataset.sample(frac=1 , random_state= 100)
train_data = shuffeld_dataset.iloc[0:index_80th]
validation_data = shuffeld_dataset.iloc[index_80th:index_90th]
test_data = shuffeld_dataset.iloc[index_90th:]


label_columns = ['Class' , 'Class_Bool']

train_features = train_data.drop(columns=label_columns)
train_label = train_data['Class_Bool'].to_numpy()
valdiation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data["Class_Bool"].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Area',
]



def create_model(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric],
) -> keras.Model:
    model_inputs = [
        keras.Input(name=feature , shape=(1,))
        for feature in settings.input_features
    ]



    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    model_output = keras.layers.Dense(
        units=1, name='dense_layer', activation=keras.activations.sigmoid
    )(concatenated_inputs)
    model = keras.Model(inputs=model_inputs , outputs = model_output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(
            settings.learning_rate
        ),
        loss = keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model


def train_model(
        experiment_name:str,
        model: keras.Model, 
        dataset: pd.DataFrame,
        labels: np.ndarray,
        settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.ExperimentSettings:
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x = features,
        y = labels,
        batch_size= settings.batch_size,
        epochs = settings.number_epochs,
    )
    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )


settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features = input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold = settings.classification_threshold
    ),
    keras.metrics.Precision(
        name='percision', thresholds = settings.classification_threshold
    ), 
    keras.metrics.Recall(
        name='recall' , thresholds = settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

model = create_model(settings , metrics)

experiment = train_model(
    'baselline', model , train_features, train_label , settings
)

ml_edu.results.plot_experiment_metrics(experiment,['accuracy' ,'percision' , 'recall'])
plt.show()
ml_edu.results.plot_experiment_metrics(experiment , ['auc'])
plt.show()