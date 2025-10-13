import matplotlib.pyplot as plt
import random
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

def plot_state(state):
  plt.plot(np.arange(0, 50, 1), state[0, :], label='x', color='red')
  plt.plot(np.arange(0, 50, 1), state[1, :], label='y', color='blue')
  plt.plot(np.arange(0, 50, 1), state[2, :], label='z', color='green')

  plt.plot(np.arange(0, 50, 1), state[3, :], '--', label='vx', color='red')
  plt.plot(np.arange(0, 50, 1), state[4, :], '--', label='vy', color='blue')
  plt.plot(np.arange(0, 50, 1), state[5, :], '--', label='vz', color='green')

  plt.legend()

  fig = plt.figure()

  axes = fig.add_subplot(111, projection='3d')

  axes.plot(state[0, :], state[1, :], state[2, :], label='position')

  axes.plot(state[3, :], state[4, :], state[5, :], label='velocity')

  plt.legend()

  plt.show()


def plot_noise(noise, t_samples, sigma):
  plt.hist(t_samples, bins=30, density=True)

  plt.plot(np.arange(-4*sigma, 4*sigma, 8*sigma/100), noise(np.arange(-4*sigma, 4*sigma, 8*sigma/100)))

  plt.show()


if __name__ == "__main__":
  process_noise_sigma = 5
  noise = lambda t: 1/(process_noise_sigma * np.sqrt(2 * np.pi)) * np.e ** (-1/2 * (t / process_noise_sigma) ** 2)

  t_samples = np.random.normal(0, process_noise_sigma, 1000)

  n_samples = 50
  t = np.linspace(0, 2 * np.pi, n_samples)

  plot_noise(noise=noise, t_samples=t_samples, sigma=process_noise_sigma)

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

  plot_state(state=true_state)

  noisy_state = np.zeros(shape=(6, 50))

  noisy_state[:, 0] = initial_state

  for i in range(np.shape(inputs)[1] - 1):
    noisy_state[:, i+1] = F @ noisy_state[:, i] + G @ inputs[:, i] + np.random.normal(0, process_noise_sigma, size=(6,))

  plot_state(noisy_state)

  # kalman filter

  measurement_noise_sigma = 2

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

  Q = process_noise_sigma**2 * np.eye(6)

  H = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0]])

  for i in range(0, inputs.shape[1]-1):
    # predict
    estimated_state[:, i+1] = F @ estimated_state[:, i] + G @ inputs[:, i]
    estimate_covariance[:, :, i+1] = F @ estimate_covariance[:, :, i] @ F.T + Q

    # correct
    kalman_gain[:, :, i+1] = estimate_covariance[:, :, i+1] @ H.T @ np.linalg.inv(H @ estimate_covariance[:, :, i+1] @ H.T + measurement_covariance)
    estimated_state[:, i+1] = estimated_state[:, i+1] + kalman_gain[:, :, i+1] @ (observations[:, i+1] - H @ estimated_state[:, i+1])
    estimate_covariance[:, :, i+1] = (np.eye(6) - kalman_gain[:, :, i+1] @ H) @ estimate_covariance[:, :, i+1] @ (np.eye(6) - kalman_gain[:, :, i+1] @ H).T + kalman_gain[:, :, i+1] @ measurement_covariance @ kalman_gain[:, :, i+1].T

  plot_state(estimated_state)