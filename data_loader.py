import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import tensorflow as tf
import wandb 
import cp_detection_timeseries as cpdt
import random

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

import timeseries_dataset_by_time_indices


from pathlib from Path
FP_ROOT = Path(__file__).parent


#--------PREPARING THE DATA------

# function to load and average the raw 1-min output dataset to the desired averaging interval
def load_and_average_dataset(fp_1min, start_date='2019-07-01 00:00:00', end_date='2019-11-01 00:00:00', averaging_interval='15min'):
    fp_csv = FP_ROOT / f'WRF-{averaging_interval}.csv'
    
    
    ds = xr.open_dataset(fp_1min).sel(Time=slice(start_date,end_date))

    
    datasets = []
    for station_name in ["Pout", "Dakar"]:
        ds_station = ds.sel(station_name=station_name)
        var_names = list(ds.data_vars)
        ds_station = ds_station.rename({v: f"{v}__{station_name}" for v in var_names})
        for c in ["lat", "lon"]:
            da_coord = ds_station[c].item()
            ds_station = ds_station.drop_vars(c)
            new_name = f"{c}__{station_name}"
            ds_station[new_name] = da_coord

        ds_station = ds_station.drop("station_name")
        datasets.append(ds_station)

    ds_renamed_vars = xr.merge(datasets)
    ds_renamed_vars
    
    if fp_csv.exists():
        df = pd.read_csv(fp_csv,parse_dates=True, date_parser=pd.to_datetime,index_col='Time')
    else:
        ds_averaged = ds_renamed_vars.resample(Time=averaging_interval).mean('Time')
        df = ds_averaged.to_dataframe()
        df.to_csv(fp_csv)
    
    return ds_renamed_vars, df


# function to simply load the raw 1-min output dataset and the averaged dataset (faster than the previous)
def load_datasets(fp_1min, fp_csv_averaged, start_date='2019-07-01 00:00:00', end_date='2019-11-01 00:00:00'):
    
    ds = xr.open_dataset(fp_1min).sel(Time=slice(start_date,end_date))
    
    df_averaged = pd.read_csv(fp_csv_averaged,parse_dates=True, date_parser=pd.to_datetime,index_col='Time')

    return ds, df_averaged


def find_cps(ds_1min, df_averaged, station_name):
    dtdata = pd.to_datetime(ds_1min.Time)
    ttdata = ds_1min.T2.sel(station_name=station_name).to_numpy()
    rrdata = ds_1min.RAIN_1MIN.sel(station_name=station_name).to_numpy()
    ppdata = ds_1min.PSFC.sel(station_name=station_name).to_numpy()
    uudata = ds_1min.U10.sel(station_name=station_name).to_numpy()

    cps = cpdt.cp_detection(dtdata, ttdata, rrdata)  # Perform cold-pool detection                    
    cp_times = cps.datetimes()                                         
    idx_cps = df_averaged.index.get_indexer(cp_times, method='nearest')
    return idx_cps


def sample_surrounding_indices(start_index, end_index, selected_indexes, num_steps):
    surrounding_indices = []

    for idx in selected_indexes:
        # Calculate the range of indices, ensuring they stay within [start_index, end_index]
        start_idx = max(start_index, idx - num_steps)
        end_idx = min(end_index, idx + num_steps)
        
        surrounding_indices.extend(range(start_idx, end_idx + 1))
        # Remove duplicates
        surrounding_indices_unique = list(set(surrounding_indices))
    return  (surrounding_indices_unique)

def generate_random_indices(start_index, end_index, size, subset_to_exclude):
    # Convert the subset to exclude into a set for faster lookup
    exclude_set = set(subset_to_exclude)
    
    # Create a list of all possible numbers
    all_numbers = np.arange(start_index, end_index)
    
    # Create a set of all possible numbers excluding the subset
    all_numbers = [num for num in all_numbers if num not in exclude_set]
    
    # Sample random numbers from the set of all possible numbers
    random_numbers = random.sample(all_numbers, size)
    
    return (random_numbers)

def split_indices(time_indices, train_fraction=0.7, val_fraction=0.2):
    np.random.shuffle(time_indices)
    n_time_indices = len(time_indices)

    n_train = int(n_time_indices * train_fraction)
    n_val = int(n_time_indices * val_fraction)
    n_test = n_time_indices - n_train - n_val

    train_time_indices = time_indices[0:n_train]
    val_time_indices = time_indices[n_train : n_train + n_val]
    test_time_indices = time_indices[n_train + n_val :]

    return {
        "train": train_time_indices,
        "val": val_time_indices,
        "test": test_time_indices,
    }


def sample_indices_includingCPs_vs_random(ds_1min, df_averaged, n_samples_beforeafterCPs):
    n, num_features = df_averaged.shape
    
    # Find CPs in 'Pout' and 'Dakar'
    pout_cps = set(find_cps(ds_1min, df_averaged, 'Pout'))
    dakar_cps = set(find_cps(ds_1min, df_averaged, 'Dakar'))
    
    # Sample surrounding indices based on 'Pout' CPs
    sampled_CPs = sample_surrounding_indices(0, n-100, pout_cps, n_samples_beforeafterCPs)
    
    # Generate random indices excluding sampled CPs
    random_indices_minus_CPs = generate_random_indices(0, n-100, size=len(sampled_CPs), subset_to_exclude=sampled_CPs)
    
    # Combine sampled CPs with random indices and exclude 'Dakar' CPs
    CPs_plus_random = sampled_CPs + random_indices_minus_CPs
    CPs_plus_random_set = set(CPs_plus_random)
    CPs_plus_random_minus_dakar = CPs_plus_random_set - dakar_cps
    
    # Convert CPs_plus_random_minus_dakar back to a list
    CPs_plus_random_minus_dakar = list(CPs_plus_random_minus_dakar)
    
    # Generate random indices for the remaining samples
    time_indices_random = generate_random_indices(0, n-100, size=len(CPs_plus_random_minus_dakar), subset_to_exclude=[])
    
    # Split indices by data split
    time_indices_by_data_split = split_indices(time_indices_random)
    time_indices_by_data_split_CPs = split_indices(CPs_plus_random_minus_dakar)
    
    return (time_indices_by_data_split, time_indices_by_data_split_CPs)


def sample_indices_3sets(ds_1min, df_averaged, n_samples_beforeafterCPs):
    
    n, num_features = df_averaged.shape
    
    # Find CPs in 'Pout' and 'Dakar'
    pout_cps = set(find_cps(ds_1min, df_averaged, 'Pout'))
    dakar_cps = set(find_cps(ds_1min, df_averaged, 'Dakar'))
    
    
    #set 1 [PROPAGATING CPs]: sample CPs that propagate from Pout to Dakar:
    Propagating_CPs = sample_surrounding_indices(0, n-100, dakar_cps, n_samples_beforeafterCPs)
    Propagating_CPs_set = set(Propagating_CPs)


    
    #set 2 [NON PROPAGATING CPs]: sample CPs that are measured in Pout, that do not propagate to Dakar
    sample_dakar_CPs = sample_surrounding_indices(0, n-100, pout_cps, n_samples_beforeafterCPs)
    sample_dakar_CPs_set = set(sample_dakar_CPs)
    Not_propagating_CPs_set = sample_dakar_CPs_set.difference(Propagating_CPs_set)
    
    # Determine the minimum length between Propagating_CPs and Non_propagating_CPs
    min_length = min(len(Propagating_CPs_set), len(Not_propagating_CPs_set))

    # Randomly sample from both sets to create new sets with the minimum length
    random_propagating_cps = random.sample(Propagating_CPs_set, min_length)
    random_non_propagating_cps = random.sample(Not_propagating_CPs_set, min_length)
    
    # set 3 [NO CPs]: Generate random indices excluding all sampled CPs
    all_cps = Propagating_CPs_set.union(Not_propagating_CPs_set)
    
    Not_CPs = generate_random_indices(0, n-100, size=min_length, subset_to_exclude=all_cps)
    Not_CPs_set= set(Not_CPs)
    
    
    all_indices_set = Propagating_CPs_set.union(Not_propagating_CPs_set, Not_CPs_set)
    all_indices = list(all_indices_set)
    # Split indices by data split
    time_indices_by_data_split = split_indices(all_indices)
    
    return time_indices_by_data_split, Propagating_CPs_set, Not_propagating_CPs_set, Not_CPs_set





def create_windows(station_input, all_data, time_indices,IN_WIDTH,OUT_STEPS,SHIFT,input_columns=None, label_columns=None):
    
    
    # Dictionary mapping station data types to input column lists
    station_input_columns = {
        "pout": ["T2__Pout","Q2__Pout", "U10__Pout", "V10__Pout", "PSFC__Pout"],
        "dakar": ["T2__Dakar","Q2__Dakar", "U10__Dakar", "V10__Dakar", "PSFC__Dakar"],
        "dakar_and_pout": [
            "T2__Dakar","Q2__Dakar", "U10__Dakar", "V10__Dakar", "PSFC__Dakar",
            "T2__Pout","Q2__Pout", "U10__Pout", "V10__Pout", "PSFC__Pout"
        ]
    }

    if station_input not in station_input_columns:
        raise ValueError("Invalid station. Please choose 'pout', 'dakar', or 'dakar_and_pout'.")
    
    label_columns = ["T2__Dakar"]
    windows = WindowGeneratorByTimeIndices(
        input_width=IN_WIDTH,
        label_width=OUT_STEPS,
        shift=SHIFT,
        input_columns=station_input_columns[station_input] if input_columns is None else input_columns,
        label_columns=label_columns,
        time_indices=time_indices,
        df_all=all_data
    )
    num_features=len(station_input_columns[station_input]) if input_columns is None else len(input_columns)

    return (windows, num_features)


class WindowGeneratorByTimeIndices:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        time_indices,
        df_all,
        input_columns=None,
        label_columns=None,
        batch_size=128,
        normed_plot=False
    ):
        """
        Specialised window generator which ensures that only windows starting at
        specific time indecies are sampled.

        Parameters
        ----------
        input_width : int
            Number of input time steps.
        label_width : int
            Number of output time steps.
        shift : int
            Number of time steps to shift the window.
        time_indices : dict
            Dictionary with keys "train", "val", "test" and values which are
            lists of time indecies to sample windows from.
        df_all : pandas.DataFrame
            DataFrame containing all data (for both train, val and test)
        label_columns : list, optional
            List of column names to use as labels, by default None, so that all
            columns are used as labels.
        """
        # Store the raw data.
        self.df_all = df_all
        self.mean = self.df_all.mean()
        self.std = self.df_all.std()
        self.df_norm = (self.df_all - self.mean) / self.std
        self.batch_size = batch_size
        
        required_time_indices = ["train", "val", "test"]
        if any(kind not in time_indices for kind in required_time_indices):
            raise ValueError(
                f"time_indices must include all of {required_time_indices}"
            )
        # check here that for all time indices none of the windows run over the end of the data
        if any(
            any(
                time_indices[kind][i] + input_width + shift > len(self.df_all)
                for i in range(len(time_indices[kind]))
            )
            for kind in required_time_indices
        ):
            raise ValueError(
                "Some of the windows run over the end of the data, please reduce the input_width or shift"
                " or remove the time indecies which cause this."
            )

        self.time_indices = time_indices

        # Work out the input column indices.
        self.input_columns = input_columns
        if input_columns is not None:
            self.input_columns_indices = {
                name: i for i, name in enumerate(input_columns)
            }
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(self.df_all.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift + label_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        


    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Input column name(s): {self.input_columns}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.input_columns is not None:
            inputs = tf.stack(
                [
                    inputs[:, :, self.column_indices[name]]
                    for name in self.input_columns
                ],
                axis=-1,
            )
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, time_indices):
        data = np.array(self.df_norm, dtype=np.float32)

        # create a specialised "timeseries_dataset_from_array" function which
        # ensures that only sampling windows only start at specific time indecies
        timeseries_dataset_from_array = timeseries_dataset_by_time_indices.create_timeseries_dataset_from_array_sampler(
            time_indices=time_indices
        )
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(time_indices=self.time_indices["train"])

    @property
    def val(self):
        return self.make_dataset(time_indices=self.time_indices["val"])

    @property
    def test(self):
        return self.make_dataset(time_indices=self.time_indices["test"])

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset - changed to '.test'
            result = next(iter(self.test)) #self.train
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col_input='T2__Dakar', plot_col_label='T2__Dakar',
             ymin=298, ymax=305,normed_plot=False, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_input_index =self.column_indices[plot_col_input] #self.column_indices[plot_col_input]
        plot_col_label_index = self.column_indices[plot_col_label] #self.column_indices[plot_col_label]
        mean=self.mean
        std=self.std
        max_n = min(max_subplots, len(inputs))
        if normed_plot==True:
            for n in range(max_n):
                plt.subplot(max_n, 1, n + 1)
                plt.ylabel(f"{plot_col_input} [normed]")


                input_col_index = plot_col_input_index

                if input_col_index is None:
                    continue

                label_col_index = plot_col_label_index

                if label_col_index is None:
                    continue

                
                
                plt.plot(
                    self.input_indices,
                    inputs[n, :, input_col_index],
                    label="Inputs",
                    marker=".",
                    zorder=-10,
                )


                plt.scatter(
                    self.label_indices,
                    labels[n, :, label_col_index],
                    edgecolors="k",
                    label="Labels",
                    c="#2ca02c",
                    s=64,
                )
                if model is not None:
                    predictions = model(inputs)
                    plt.scatter(
                        self.label_indices,
                        predictions[n, :, label_col_index],
                        marker="X",
                        edgecolors="k",
                        label="Predictions",
                        c="#ff7f0e",
                        s=64,
                    )
        

                if n == 0:
                    plt.legend()
                    

                
        elif normed_plot==False:
            for n in range(max_n):
                plt.subplot(max_n, 1, n + 1)
                plt.ylabel(f"{plot_col_input}")
                plt.ylim(ymin,ymax)
                if self.input_columns:
                    input_col_index = self.input_columns_indices.get(plot_col_input, None)
                else:
                    input_col_index = plot_col_input_index

                if input_col_index is None:
                    continue

                if self.label_columns:
                    label_col_index = self.label_columns_indices.get(plot_col_label, None)
                else:
                    label_col_index = plot_col_label_index

                if label_col_index is None:
                    continue
                
                # Extract the time indices corresponding to the input and prediction window
                
                # Calculate the starting index of the plotted window
                start_index = self.time_indices['test'][n] 
                
                # Calculate the ending index of the plotted window
                end_index = start_index + self.total_window_size
                
                # Slice the relevant portion of df_all based on the calculated indices
                df_all_subset = self.df_all.iloc[start_index:end_index]

                # Plot raw df_all data subset
                plt.plot(
                    np.arange(self.total_window_size),
                    (df_all_subset[plot_col_input])  ,
                    label="Raw Data",
                    linestyle="-",
                    color="gray",
                    alpha=0.5
                )
                
                plt.plot(
                    self.input_indices,
                    (inputs[n, :, input_col_index]* std[plot_col_input]  + mean[plot_col_input] ) ,
                    label="Inputs",
                    marker=".",
                    zorder=-10,
                )
                
                plt.scatter(
                    self.label_indices,
                    (labels[n, :, label_col_index] * std[plot_col_label]+ mean[plot_col_label]) ,
                    edgecolors="k",
                    label="Labels",
                    c="#2ca02c",
                    s=64,
                )
                
                if model is not None:
                    predictions = model(inputs)
                    plt.scatter(
                        self.label_indices,
                        (predictions[n, :, label_col_index]* std[plot_col_label] + mean[plot_col_label])  ,
                        marker="X",
                        edgecolors="k",
                        label="Predictions",
                        c="#ff7f0e",
                        s=64,
                    )
                

                if n == 0:
                    plt.legend()
                

            plt.xlabel("Timestep")
        
   
       



