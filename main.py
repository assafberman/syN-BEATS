import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.cm import get_cmap
from darts.models import NBEATSModel, RegressionEnsembleModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import TimeSeries
import torch
from torch.nn import MSELoss, HuberLoss
from torchmetrics import MeanSquaredError
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import r2_score
import warnings
from datetime import datetime
from skopt.space.space import Integer
from skopt import gp_minimize

warnings.filterwarnings('ignore')


def input_output_preparation(input_df: pd.DataFrame, output_series: pd.Series, split: int):
    ts_input = TimeSeries.from_dataframe(input_df, fill_missing_dates=True, freq='H')
    ts_input = TimeSeries.from_dataframe(ts_input.pd_dataframe().fillna(method='ffill'))
    ts_output = TimeSeries.from_series(output_series, fill_missing_dates=True, freq='H')
    ts_output = TimeSeries.from_dataframe(ts_output.pd_dataframe().fillna(method='ffill'))
    ts_input_train, ts_input_test = ts_input.split_after(split)
    ts_input_train, ts_input_val = ts_input_train.split_after(0.8)
    scaler_input = Scaler()
    scaler_input.fit(ts_input_train)
    # ts_input_train = scaler_input.transform(ts_input_train)
    # ts_input_test = scaler_input.transform(ts_input_test)
    ts_output_train, ts_output_test = ts_output.split_after(split)
    ts_output_train, ts_output_val = ts_output_train.split_after(0.8)
    scaler_output = Scaler()
    scaler_output.fit(ts_output_train)
    # ts_output_train = scaler_output.transform(ts_output_train)
    # ts_output_test = scaler_output.transform(ts_output_test)
    return ts_input_train, ts_input_val, ts_input_test, ts_output_train, ts_output_val, ts_output_test, scaler_input, scaler_output


def create_model(input_chunk=120, output_chunk=24, num_stacks=20, num_blocks=3, num_layers=4, layer_width=512,
                 batch_size=8):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    model = NBEATSModel(input_chunk_length=input_chunk, output_chunk_length=output_chunk, num_stacks=num_stacks,
                        num_blocks=num_blocks,
                        num_layers=num_layers, layer_widths=layer_width, batch_size=batch_size, n_epochs=5,
                        log_tensorboard=True,
                        generic_architecture=True, force_reset=True, save_checkpoints=False,
                        loss_fn=HuberLoss(), torch_metrics=MeanSquaredError(squared=False), expansion_coefficient_dim=5,
                        pl_trainer_kwargs={'accelerator': 'gpu', 'devices': -1}, optimizer_kwargs={'lr': 1e-5})
    return model


def create_ensemble(list_blocks, list_stacks):
    model = RegressionEnsembleModel(
        [create_model(num_blocks=i, num_stacks=j) for i, j in zip(list_blocks, list_stacks)],
        regression_train_n_points=24)
    return model


def fit_model(model, input_train, input_val, output_train, output_val):
    model.fit(series=output_train, past_covariates=input_train, val_series=output_val, val_past_covariates=input_val,
              verbose=False)
    return model


def predict_output(model, input_series, output_series, time_steps=24):
    prediction = model.predict(time_steps, series=output_series, past_covariates=input_series, verbose=False)
    return prediction


def optimize_weights(weight_list):
    global predictions_lin_com, out_val
    weight_list = [x / np.sum(np.abs(weight_list)) for x in weight_list]
    prediction = np.sum(ts.values() * weight for ts, weight in
                        zip(predictions_lin_com, weight_list))
    prediction = TimeSeries.from_times_and_values(
        times=predictions_lin_com[0].time_index, values=prediction)
    mse = np.mean((prediction.values() - out_val[:PREDICTION_PERIOD].values()) ** 2)
    return mse


print(f'{datetime.now()}: Starting data load.')

met_coast_df = pd.concat([pd.read_excel('Data\IMS_data_timorim_2020.xlsx'),
                          pd.read_excel('Data\IMS_data_timorim_2021.xlsx'),
                          pd.read_excel('Data\IMS_data_timorim_2022.xlsx')])
met_south_df = pd.concat([pd.read_excel('Data\IMS_data_yotveta_2020.xlsx'),
                          pd.read_excel('Data\IMS_data_yotveta_2021.xlsx'),
                          pd.read_excel('Data\IMS_data_yotveta_2022.xlsx')])
aq_coast_df = pd.read_excel('Data\AQ_data_timorim.xlsx')
aq_south_df = pd.read_excel('Data\AQ_data_yotveta.xlsx')

TIME_LAG_MET_POLL = 3  # hours

# Timorim
met_coast_df['Datetime'] = pd.to_datetime(met_coast_df['Datetime'], format='%d/%m/%Y %H:%M')
met_coast_df['Datetime'] = met_coast_df['Datetime'] + pd.Timedelta(hours=TIME_LAG_MET_POLL)
met_coast_df.set_index('Datetime', drop=True, inplace=True)
met_coast_df.replace(to_replace='-', value=np.nan, inplace=True)
met_coast_df.interpolate(method='akima', inplace=True)
aq_coast_df[['time', 'date']] = aq_coast_df['Datetime'].str.split(expand=True)
aq_coast_df['Datetime'] = (pd.to_datetime(aq_coast_df.pop('date'), format='%d/%m/%Y') +
                           pd.to_timedelta(aq_coast_df.pop('time') + ':00'))
aq_coast_df['Datetime'] = pd.to_datetime(aq_coast_df['Datetime'], format='%d/%m/%Y %H:%M')
aq_coast_df.set_index('Datetime', drop=True, inplace=True)
aq_coast_df.replace(to_replace=0, value=np.nan, inplace=True)
aq_coast_df.interpolate(method='akima', inplace=True)
coast_df = met_coast_df.merge(aq_coast_df, how='inner', left_index=True, right_index=True)
coast_df = coast_df[coast_df['PM2.5 [ug/m^3]'] < 1000]
coast_df = coast_df[coast_df['PM2.5 [ug/m^3]'] > 0]
for col in coast_df.columns:
    coast_df[col] = pd.to_numeric(coast_df[col], errors='coerce')
coast_df.dropna(inplace=True, how='any')

# Yotveta
met_south_df['Datetime'] = pd.to_datetime(met_south_df['Datetime'], format='%d/%m/%Y %H:%M')
met_south_df['Datetime'] = met_south_df['Datetime'] + pd.Timedelta(hours=TIME_LAG_MET_POLL)
met_south_df.set_index('Datetime', drop=True, inplace=True)
met_south_df.replace(to_replace='-', value=np.nan, inplace=True)
met_south_df.interpolate(method='akima', inplace=True)
aq_south_df[['time', 'date']] = aq_south_df['Datetime'].str.split(expand=True)
aq_south_df['Datetime'] = (pd.to_datetime(aq_south_df.pop('date'), format='%d/%m/%Y') + pd.to_timedelta(
    aq_south_df.pop('time') + ':00'))
aq_south_df['Datetime'] = pd.to_datetime(aq_south_df['Datetime'], format='%d/%m/%Y %H:%M')
aq_south_df.set_index('Datetime', drop=True, inplace=True)
aq_south_df.replace(to_replace=0, value=np.nan, inplace=True)
aq_south_df.interpolate(method='akima', inplace=True)
south_df = met_south_df.merge(aq_south_df, how='inner', left_index=True, right_index=True)
for col in south_df.columns:
    south_df[col] = pd.to_numeric(south_df[col], errors='coerce')
south_df.dropna(inplace=True, how='any')

print(f'{datetime.now()}: Data loading complete. Starting calculation.')

stack_list = [1, 2, 3, 5, 6, 10, 15, 30]
block_list = [30 // i for i in stack_list]
pollutant = 15
PREDICTION_PERIOD = 12
with open('best_weights.csv', 'w') as f:
    f.write('time iterator, weights\n')
rrmse_list = []
for time_iterator in range(0, 5):
    in_train, in_val, in_test, out_train, out_val, out_test, _, _ = input_output_preparation(
        coast_df, coast_df.iloc[:, pollutant],
        south_df.index.get_loc('2022-06-01 00:00:00') + time_iterator * PREDICTION_PERIOD)
    predictions_lin_com = []
    predictions_final = []
    for i, (s, b) in enumerate(zip(stack_list, block_list)):
        model = create_model(num_stacks=s, num_blocks=b)
        print(f'Model {i + 1} created for time step {time_iterator + 1}. Starting fit process.')
        model = fit_model(model, in_train, in_val, out_train, out_val)
        print(
            f'{datetime.now()}: Model {i + 1} fitting complete for time step {time_iterator + 1}. Starting prediction process.')
        predictions_lin_com.append(predict_output(model, in_train, out_train, PREDICTION_PERIOD))
        predictions_final.append(predict_output(model, in_val, out_val, PREDICTION_PERIOD))
        print(f'{datetime.now()}: Data prediction complete for model {i + 1}.')
    print(f'{datetime.now()}: Weight optimization started.')
    search_space = [Integer(-10, 10, name='Model_1'), Integer(-10, 10, name='Model_2'),
                    Integer(-10, 10, name='Model_3'), Integer(-10, 10, name='Model_4'),
                    Integer(-10, 10, name='Model_5'), Integer(-10, 10, name='Model_6'),
                    Integer(-10, 10, name='Model_7'), Integer(-10, 10, name='Model_8')]
    res = gp_minimize(optimize_weights, search_space, n_calls=100, verbose=False)
    print(f'{datetime.now()}: Weight optimization complete. Saving data to files.')
    best_weights = res['x']
    best_weights = [x / np.sum(np.abs(best_weights)) for x in best_weights]
    prediction_final = np.sum(ts.values() * weight for ts, weight in zip(predictions_final, best_weights))
    prediction_final = TimeSeries.from_times_and_values(times=predictions_final[0].time_index, values=prediction_final)
    prediction_final.pd_dataframe().to_csv(f'prediction_{time_iterator}.csv')
    mse = np.mean((prediction_final.values() - out_test.values()[:PREDICTION_PERIOD]) ** 2)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.mean(out_test.values()[:PREDICTION_PERIOD])
    rrmse_list.append(rrmse)
    print('RRMSE:', rrmse)
    with open('best_weights.csv', 'a') as f:
        f.write(f'{time_iterator},')
        for i in best_weights:
            f.write(f'{i},')
        f.write('\n')
print('Average RRMSE:', np.mean(rrmse_list))
