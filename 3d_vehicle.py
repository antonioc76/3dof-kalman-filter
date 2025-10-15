import matplotlib.pyplot as plt
import random
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

def plot_state(ax, state, color, linstyle='', title='', label=''):

  ax.plot(state[0, :], state[1, :], state[2, :], linstyle, label=label + ' position', color=color)

  ax.plot(state[3, :], state[4, :], state[5, :], linstyle, label=label + ' velocity')

  if title != '':
    ax.set_title(title)


def plot_3d_state(ax, state, color, title='', label=''):

  ax.plot(state[0, :], state[1, :], state[2, :], '--', label=label, color=color)


def compute_noisy_state(process_noise_sigma, initial_state):
  noisy_state = np.zeros(shape=(6, 50))

  noisy_state[:, 0] = initial_state

  disturbance = np.zeros(shape=(6,50))

  for i in range(np.shape(inputs)[1] - 1):
    disturbance[:, i] = G @ inputs[:, i] + np.random.normal(0, process_noise_sigma, size=(6,))
    noisy_state[:, i+1] = F @ noisy_state[:, i] + disturbance[:, i]

  return noisy_state, disturbance


def compute_kf(disturbance, process_noise_sigma, measurement_noise_sigma, Q_override, R_override):
  process_noise_sigma, process_noise_func, t_samples_process = compute_noise(process_noise_sigma)
  measurement_noise_sigma, measurement_noise_func, t_samples_measurement = compute_noise(measurement_noise_sigma)

  observations = true_state[:3, :] + np.random.normal(0, measurement_noise_sigma, size=(3, 50))

  estimated_state = np.zeros(shape=(6, 50))

  initial_estimate = np.array([0, 0, 0, 0, 0, 0])

  estimated_state[:, 0] = initial_estimate

  estimate_covariance = np.zeros(shape=(6, 6, 50))

  initial_covariance = np.array([[process_noise_sigma**2, 0, 0, process_noise_sigma**2, 0, 0],
                                  [0, process_noise_sigma**2, 0, 0, process_noise_sigma**2, 0],
                                  [0, 0, process_noise_sigma**2, 0, 0, process_noise_sigma**2],
                                  [process_noise_sigma**2, 0, 0, process_noise_sigma**2, 0, 0],
                                  [0, process_noise_sigma**2, 0, 0, process_noise_sigma**2, 0],
                                  [0, 0, process_noise_sigma**2, 0, 0, process_noise_sigma**2]])

  estimate_covariance[:, :, 0] = initial_covariance

  kalman_gain = np.zeros(shape=(6, 3, 50))

  measurement_covariance = np.array([[measurement_noise_sigma**2, 0, 0],
                                              [0, measurement_noise_sigma**2, 0],
                                              [0, 0, measurement_noise_sigma**2]])
  
  measurement_covariance = R_override * np.eye(3)

  Q = process_noise_sigma**2 * np.eye(6)

  print(Q)

  Q = Q_override * np.eye(6)

  print(Q)

  H = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0]])

  for i in range(0, inputs.shape[1]-1):
    # predict
    estimated_state[:, i+1] = F @ estimated_state[:, i] + disturbance[:, i]
    estimate_covariance[:, :, i+1] = F @ estimate_covariance[:, :, i] @ F.T + Q

    # correct
    kalman_gain[:, :, i+1] = estimate_covariance[:, :, i+1] @ H.T @ np.linalg.inv(H @ estimate_covariance[:, :, i+1] @ H.T + measurement_covariance)
    estimated_state[:, i+1] = estimated_state[:, i+1] + kalman_gain[:, :, i+1] @ (observations[:, i+1] - H @ estimated_state[:, i+1])
    estimate_covariance[:, :, i+1] = (np.eye(6) - kalman_gain[:, :, i+1] @ H) @ estimate_covariance[:, :, i+1] @ (np.eye(6) - kalman_gain[:, :, i+1] @ H).T + kalman_gain[:, :, i+1] @ measurement_covariance @ kalman_gain[:, :, i+1].T

  return observations, estimated_state


def plot_noise(ax, noise, t_samples, sigma, label):
  ax.hist(t_samples, bins=30, density=True)

  ax.plot(np.arange(-4*sigma, 4*sigma, 8*sigma/100), noise(np.arange(-4*sigma, 4*sigma, 8*sigma/100)), label=label)

  ax.set_title('noise models')


def compute_noise(sigma):
  process_noise_sigma = sigma
  noise_func = lambda t: 1/(process_noise_sigma * np.sqrt(2 * np.pi)) * np.e ** (-1/2 * (t / process_noise_sigma) ** 2)
  t_samples = np.random.normal(0, process_noise_sigma, 1000)

  return sigma, noise_func, t_samples


def plot_noisy_states(noisy_state, observations):
  
  plot_state(axes[1], state=true_state, label='true state', color='green')
  plot_state(axes[1], noisy_state, title='unfiltered model & observations', label='process noise state', color='red')
  plot_3d_state(axes[1], state=observations, label='observations', color='purple')


def plot_filtered_state(estimated_state):
  plot_state(axes[2], true_state, label='true state', color='green')  
  plot_state(axes[2], estimated_state, title='kalman filtered model', linstyle='--', label='estimated_state', color='blue')


def noise_update(val):
  for ax in axes:
    ax.cla()

  process_noise_sigma, process_noise_func, t_samples_process = compute_noise(process_noise_slider.val)
  measurement_noise_sigma, measurement_noise_func, t_samples_measurement = compute_noise(measurement_noise_slider.val)
  plot_noise(axes[0], noise=process_noise_func, t_samples=t_samples_process, sigma=process_noise_sigma, label='process noise')
  plot_noise(axes[0], noise=measurement_noise_func, t_samples=t_samples_measurement, sigma=measurement_noise_sigma, label='measurement noise')

  noisy_state, disturbance = compute_noisy_state(process_noise_slider.val, initial_state=initial_state)
  observations, estimated_state = compute_kf(disturbance, process_noise_slider.val, measurement_noise_slider.val, Q_override=Q_diagonal.val, R_override=R_diagonal.val)

  plot_noisy_states(noisy_state=noisy_state, observations=observations)

  plot_filtered_state(estimated_state=estimated_state)

  fig.canvas.draw_idle()


def optimize(event):
    new_q_val = process_noise_slider.val ** 2
    new_q_max = new_q_val * 2
    Q_diagonal.valmax = new_q_max
    Q_diagonal.ax.set_xlim(Q_diagonal.valmin, Q_diagonal.valmax)
    Q_diagonal.set_val(new_q_val)

    new_r_val = measurement_noise_slider.val ** 2
    new_r_max = new_r_val * 2
    R_diagonal.valmax = new_r_max
    R_diagonal.ax.set_xlim(R_diagonal.valmin, R_diagonal.valmax)
    R_diagonal.set_val(new_r_val)

    fig.canvas.draw_idle()


if __name__ == "__main__":
  fig = plt.figure(figsize=(8, 6))

  axes = []

  axes.append(fig.add_subplot(2, 2, 1))
  axes.append(fig.add_subplot(2, 2, 2, projection='3d'))
  axes.append(fig.add_subplot(2, 2, 3, projection='3d'))

  process_noise_sigma, process_noise_func, t_samples_process = compute_noise(2)

  initial_process_noise_Sigma = process_noise_sigma

  n_samples = 50
  t = np.linspace(0, 2 * np.pi, n_samples)

  radius = 8 + 2 * np.sin(2 * t)
  x_accs = radius * np.cos(t) + 0.5 * np.sin(3 * t)
  y_accs = radius * np.sin(t) + 0.3 * np.cos(2 * t)

  z_accs = np.concatenate([
    np.arange(0, 0.5, 0.5/10),
    np.arange(0, 1, 1/10),
    np.arange(1, 3, 2/10),
    np.arange(3, -1, -4/10),
    np.arange(0, 1, 1/10)
  ])[:50]

  inputs = np.vstack([x_accs, y_accs, z_accs])

  delta_t = 0.5

  F = np.array([[1, 0, 0, delta_t, 0, 0],
                 [0, 1, 0, 0, delta_t, 0],
                 [0, 0, 1, 0, 0, delta_t],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]])
  
  G = np.array([[0.5 * delta_t ** 2, 0, 0],
                 [0, 0.5*delta_t**2, 0],
                 [0, 0, 0.5*delta_t**2],
                 [delta_t, 0, 0],
                 [0, delta_t, 0],
                 [0, 0, delta_t]])
  
  true_state = np.zeros(shape=(6, 50))

  initial_state = np.array([50, -50, 0, 0, 0, 0])

  true_state[:, 0] = initial_state
  
  for i in range(0, np.shape(inputs)[1] - 1):
    true_state[:, i+1] = F @ true_state[:, i] + G @ inputs[:, i]

  noisy_state, disturbance = compute_noisy_state(process_noise_sigma=process_noise_sigma, initial_state=initial_state)

  # kalman filter

  measurement_noise_sigma, measurement_noise_func, t_samples_measurement = compute_noise(5)

  initial_measurement_noise_sigma = measurement_noise_sigma

  observations, estimated_state = compute_kf(disturbance=disturbance, process_noise_sigma=process_noise_sigma, measurement_noise_sigma=measurement_noise_sigma, Q_override=process_noise_sigma**2, R_override=measurement_noise_sigma**2)
  
  plot_noise(axes[0], noise=process_noise_func, t_samples=t_samples_process, sigma=process_noise_sigma, label='process noise')
  plot_noise(axes[0], noise=measurement_noise_func, t_samples=t_samples_measurement, sigma=measurement_noise_sigma, label='measurement noise')

  plot_noisy_states(noisy_state=noisy_state, observations=observations)

  plot_filtered_state(estimated_state=estimated_state)

  process_noise_axis = fig.add_axes([0.6, 0.5, 0.3, 0.01])
  measurement_noise_axis = fig.add_axes([0.6, 0.4, 0.3, 0.01])
  Q_scale_axis = fig.add_axes([0.6, 0.3, 0.3, 0.01])
  R_scale_axis = fig.add_axes([0.6, 0.2, 0.3, 0.01])

  optimize_button_axis = fig.add_axes([0.5, 0.05, 0.2, 0.1])
  resample_button_axis = fig.add_axes([0.75, 0.05, 0.2, 0.1])

  process_noise_slider = Slider(
    ax=process_noise_axis,
    label="process noise",
    valmin=0.1,
    valmax=25,
    valinit=process_noise_sigma,
    orientation='horizontal'
  )

  measurement_noise_slider = Slider(
    ax=measurement_noise_axis,
    label="measurement noise",
    valmin=0.1,
    valmax=25,
    valinit=measurement_noise_sigma,
    orientation='horizontal'
  )

  Q_diagonal = Slider(
    ax=Q_scale_axis,
    label="Q matrix diagonal",
    valmin=0.1,
    valmax=process_noise_sigma**2*2,
    valinit=process_noise_sigma**2,
    orientation='horizontal'
  )

  R_diagonal = Slider(
    ax=R_scale_axis,
    label="R matrix diagonal",
    valmin=0.1,
    valmax=measurement_noise_sigma**2*2,
    valinit=measurement_noise_sigma**2,
    orientation='horizontal'
  )

  optimize_button = Button(
    ax=optimize_button_axis,
    label="optimize"
  )

  resample_button = Button(
    ax=resample_button_axis,
    label='resample'
  )

  process_noise_slider.on_changed(noise_update)
  measurement_noise_slider.on_changed(noise_update)

  Q_diagonal.on_changed(noise_update)
  R_diagonal.on_changed(noise_update)

  optimize_button.on_clicked(optimize)
  resample_button.on_clicked(noise_update)
  
  plt.show()

  