import sys
import os

sys.path.append(os.path.join(os.path.pardir, 'util'))

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
import pynumdiff

import figurefirst as fifi
import figure_functions as ff
from utils import wrapTo2Pi, wrapToPi
from setdict import SetDict

pd.set_option('mode.chained_assignment', None)


class Trajectory:
    def __init__(self, traj, props=None, time_range=(-0.1, 1.1), norm=True, fc=20, filter_order=3, find_heading=False):
        """ Proces trajectory data.

            Inputs
                traj: data frame with trajectory data
                props: default mass & drag properties
                time_range: (min_time, max_time) time window to analyze (ms)
                w: ambient wind magnitude
                zeta: ambient wind direction
                norm: (boolean) if True, normalize position
                fc: cutoff frequency for low-pass filter in hz
                filter_order: low-pass filter order
                find_heading: (boolean) to find heading use model

        """

        # Default mass & drag properties
        mass = 0.25e-6
        self.props = {'mass': mass,  # mass [kg]
                      'inertia': 5.2e-13,  # [N*m*s^2] yaw mass moment of inertia: 10.1242/jeb.02369
                      'drag': mass / 0.170,  # [N*s/m] calculate using the mass and time constant: 10.1242/jeb.098665
                      'ambient_wind_speed': 0.4,  # [m/s] ambient wind speed
                      'ambient_wind_direction': np.pi}  # [rad] ambient wind direction

        # Overwrite properties
        if props is not None:
            SetDict().set_dict_with_overwrite(self.props, props)

        # Get raw trajectory
        self.traj_raw = traj.copy()

        # Create new data-frame
        self.traj = self.traj_raw.rename(columns={'x': 'position_x', 'y': 'position_y', 'z': 'position_z',
                                                  'xvel': 'velocity_x', 'yvel': 'velocity_y', 'zvel': 'velocity_z'})

        # Add data
        self.traj['time'] = self.traj_raw['time stamp'] / 1000  # time in seconds
        self.traj['pulse'] = (self.traj['time'] >= 0) & (self.traj['time'] < 0.680)
        self.traj['w'] = self.props['ambient_wind_speed'] * np.ones_like(self.traj['time'])
        self.traj['zeta'] = self.props['ambient_wind_direction'] * np.ones_like(self.traj['time'])

        # Get trajectory window
        time_window = (self.traj['time'] >= time_range[0]) & (self.traj['time'] <= time_range[1])
        self.traj = self.traj.loc[time_window, ['time', 'pulse', 'w', 'zeta', 'position_x', 'position_y', 'position_z']]

        # Normalize position
        if norm:
            self.traj.position_x = self.traj.position_x - self.traj.position_x.iloc[0]
            self.traj.position_y = self.traj.position_y - self.traj.position_y.iloc[0]

        # Sampling rate
        self.dt = np.round(np.squeeze(np.mean(np.diff(self.traj.time.values))), 5)  # sampling time
        self.fs = 1 / self.dt  # sampling frequency
        self.fc_norm = fc / (self.fs / 2)

        # Calculate velocity
        vel_x = pynumdiff.smooth_finite_difference.butterdiff(self.traj.position_x.values, self.dt, [filter_order, self.fc_norm])[1]
        vel_y = pynumdiff.smooth_finite_difference.butterdiff(self.traj.position_y.values, self.dt, [filter_order, self.fc_norm])[1]
        vel_z = pynumdiff.smooth_finite_difference.butterdiff(self.traj.position_z.values, self.dt, [filter_order, self.fc_norm])[1]

        self.traj['velocity_x'] = vel_x
        self.traj['velocity_y'] = vel_y
        self.traj['velocity_z'] = vel_z
        self.traj['g'] = np.sqrt(self.traj['velocity_x'] ** 2 + self.traj['velocity_y'] ** 2)
        self.traj['psi_global'] = wrapToPi(np.arctan2(self.traj['velocity_y'], self.traj['velocity_x']))

        # Calculate acceleration
        accel_x = pynumdiff.smooth_finite_difference.butterdiff(self.traj.velocity_x.values, self.dt, [filter_order, self.fc_norm])[1]
        accel_y = pynumdiff.smooth_finite_difference.butterdiff(self.traj.velocity_y.values, self.dt, [filter_order, self.fc_norm])[1]
        accel_z = pynumdiff.smooth_finite_difference.butterdiff(self.traj.velocity_z.values, self.dt, [filter_order, self.fc_norm])[1]

        self.traj['acceleration_x'] = accel_x
        self.traj['acceleration_y'] = accel_y
        self.traj['acceleration_z'] = accel_z

        # Calculate air velocity
        self.traj['air_velocity_x'] = self.traj['velocity_x'] - self.traj['w'] * np.cos(self.traj['zeta'])
        self.traj['air_velocity_y'] = self.traj['velocity_y'] - self.traj['w'] * np.sin(self.traj['zeta'])
        self.traj['gamma_global'] = np.arctan2(self.traj['air_velocity_y'], self.traj['air_velocity_x'])

        # Calculate heading/orientation angle as thrust angle from acceleration & mass/drag properties
        thrust_x = self.props['mass'] * self.traj['acceleration_x'] + self.props['drag'] * self.traj['air_velocity_x'].values
        thrust_y = self.props['mass'] * self.traj['acceleration_y'] + self.props['drag'] * self.traj['air_velocity_y'].values
        thrust_angle = wrapToPi(np.arctan2(thrust_y, thrust_x))

        if find_heading:
            self.traj['phi'] = thrust_angle
        else:
            self.traj['phi'] = self.traj['psi_global'].copy()

        self.traj['phi_unwrap'] = np.unwrap(self.traj['phi'])

        # Calculate velocity in body reference frame
        self.traj['psi'] = wrapToPi(self.traj['psi_global'] - self.traj['phi'])
        self.traj['v_para'] = self.traj['g'] * np.cos(self.traj['psi'])
        self.traj['v_perp'] = self.traj['g'] * np.sin(self.traj['psi'])

        # Calculate air velocity in body reference frame
        self.traj['a_para'] = self.traj['v_para'] - self.traj['w'] * np.cos(self.traj['phi'] - self.traj['zeta'])
        self.traj['a_perp'] = self.traj['v_perp'] + self.traj['w'] * np.sin(self.traj['phi'] - self.traj['zeta'])
        self.traj['a'] = np.sqrt(self.traj['a_para'] ** 2 + self.traj['a_perp'] ** 2)
        self.traj['gamma'] = wrapToPi(np.arctan2(self.traj['a_perp'], self.traj['a_para']))

        # Calculate derivatives
        self.traj['gdot'] = pynumdiff.smooth_finite_difference.butterdiff(self.traj['g'].values, self.dt, [2, 0.95])[1]
        self.traj['phidot'] = pynumdiff.smooth_finite_difference.butterdiff(self.traj['phi_unwrap'].values, self.dt, [2, 0.95])[1]
        self.traj['phi2dot'] = pynumdiff.smooth_finite_difference.butterdiff(self.traj['phidot'].values, self.dt, [2, 0.95])[1]

    def plot_pulse_trajectory(self, size=7.0):
        fig, ax = plt.subplots(1, 1, figsize=(size, size), dpi=100)
        ax.plot(self.traj['x'], self.traj['y'], 'k', linewidth=2)
        ax.plot(self.traj['x'][self.traj['pulse']], self.traj['y'][self.traj['pulse']], 'r', linewidth=2)
        ax.set_aspect('equal', adjustable='box')
        fifi.mpl_functions.adjust_spines(ax, [])

    def plot_heading_trajectory(self, size=7.0, color=None, arrow_size=None, nskip=0, data_range=None, cmap=None,
                                colornorm=None):
        if arrow_size is None:
            arrow_size = 0.07 * np.mean(np.abs(np.hstack((self.traj['x'], self.traj['y']))))

        if cmap is None:
            if color is None:
                crange = 0.2
                cmap = cm.get_cmap('bone_r')
            else:
                crange = 0.1
                cmap = cm.get_cmap('RdPu')

            cmap = cmap(np.linspace(crange, 1, 1000))
            cmap = ListedColormap(cmap)

        if color is None:
            color = self.traj['time_seconds'].values

        x = self.traj['x'].values
        y = self.traj['y'].values
        phi = self.traj['phi_filt'].values

        if data_range is not None:
            index = np.arange(data_range[0], data_range[-1], 1)
            x = x[index]
            y = y[index]
            phi = phi[index]
            color = color[index]

        fig, ax = plt.subplots(1, 1, figsize=(size, size), dpi=100)
        ff.plot_trajectory(x, y, phi,
                           color=color,
                           ax=ax,
                           nskip=nskip,
                           size_radius=arrow_size,
                           colormap=cmap,
                           colornorm=colornorm)

        fifi.mpl_functions.adjust_spines(ax, [])

    def plot_velocity(self, size=5.0):
        fig, ax = plt.subplots(2, 2, figsize=(size, size), dpi=100)

        time = self.traj['time_seconds']

        ax[0, 0].plot(*ff.circplot(time, self.traj['phi_filt_wrap'].values), '-')
        ax[0, 0].plot(*ff.circplot(time, self.traj['phi_wrap'].values), '.', markersize=3)

        ax[0, 1].plot(time, self.traj['g_filt'].values, '-')
        ax[0, 1].plot(time, self.traj['g'].values, '.', markersize=3)

        ax[1, 0].plot(time, self.traj['phidot'].values)

        ax[1, 1].plot(time, self.traj['gdot'].values)

        ff.pi_yaxis(ax[0, 0], tickpispace=0.5, lim=None)

        data_labels = [r'$\phi$ (rad)', r'$g$ (m/s)', r'$\dot{\phi}$ (rad/s)', r'$\dot{g}$ (m/$s^2$)']
        p = 0
        for r in range(ax.shape[0]):
            for c in range(ax.shape[1]):
                ax[r, c].grid()
                ax[r, c].set_ylabel(data_labels[p], fontsize=10)

                if r > 0:
                    ax[r, c].set_xlabel('time (s)', fontsize=10)
                else:
                    ax[r, c].xaxis.set_tick_params(labelbottom=False)

                p = p + 1

        plt.subplots_adjust(wspace=0.5, hspace=0.1)


# For batch processing fly trajectories
def run_process(fpath, props=None, g_thresh_high=1.5, g_thresh_low=0.1, time_thresh=(-0.1, 1.1)):
    # Load
    print('Loading:', fpath)
    data = pd.read_csv(fpath)

    # Find unique trajectories
    trajectory_idx = data.obj_id_unique.unique()
    n_trajectory = len(trajectory_idx)
    print(n_trajectory, 'trajectories')

    # Put each separate trajectory in list
    control_list = []
    full_list = []
    half_list = []
    for i in trajectory_idx:
        # Get trajectory
        traj = data[data.obj_id_unique == i]
        traj = traj.reset_index()

        # Normalize time & position
        traj.x = traj.x - traj.x.iloc[0]
        traj.y = traj.y - traj.y.iloc[0]
        traj.z = traj.z - traj.z.iloc[0]

        # Add to list
        exp_type = traj.duration.values[0]
        if exp_type == 100:
            full_list.append(traj)
        elif exp_type == 50:
            half_list.append(traj)
        elif exp_type == 0:
            control_list.append(traj)

    print('Full intensity pulse:', len(full_list))
    print('Half intensity pulse:', len(half_list))
    print('Control:', len(control_list))

    # Use all intensity levels
    trajectory_list = full_list + half_list + control_list

    # Process all trajectories
    DATA = []
    processed_traj = []
    print('\nRejecting:')
    count = 0
    for n, traj in enumerate(trajectory_list):
        traj_data = Trajectory(traj, props=props, time_range=time_thresh, norm=True, fc=20)

        # Clean trajectories
        if traj_data.traj.time.values[0] > (time_thresh[0]):
            print('time-start:', n, end=', ')
        elif traj_data.traj.time.values[-1] < (time_thresh[1]):
            print('time-end:', n, end=', ')
        elif np.any(traj_data.traj.g.values > g_thresh_high):
            print('g-high:', n)
        elif np.mean(traj_data.traj.g.values) < g_thresh_low:
            print('g-low:', n, end=', ')
        else:
            traj_data_id = traj_data.traj.copy()
            traj_data_id.insert(0, 'ID', count)
            count = count + 1

            DATA.append(traj_data)
            processed_traj.append(traj_data_id)

    print('\n\nRemoving ', n_trajectory - len(DATA), 'trajectories')
    print('Keeping ', len(DATA), 'trajectories')

    return DATA, processed_traj
