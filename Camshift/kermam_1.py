# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg
import cv2

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)  # Fk,状态转移矩阵
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)  # Hk，测量矩阵

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement  # [x,y,a,h]
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]  # [x,y,a,h,dx,dy,da,dh]
        std = [10, 10, 10, 10, 10, 10, 10, 10]

        covariance = np.diag(np.square(std))  # P_k，误差协方差
        return mean, covariance  # X（k-1）,P(k-1)

    def predict(self, mean, covariance):
        motion_cov = np.eye(8, 8) * 10  # Q_k

        # mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)  # X_k = X_k-1 *F_k
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov  # Pk

        return mean, covariance

    def project(self, mean, covariance):
        std = [
            1,
            1,
            1,
            1]
        innovation_cov = np.diag(np.square(std))  # Rk

        mean = np.dot(self._update_mat, mean)  # X_k*H_k
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))  # Hk*Pk*HK.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        #  H_k * x_k ,  H_k *P_k*H_k + R_k
        projected_mean, projected_cov = self.project(mean, covariance)  # 8x8 --> 4x1,4x4

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean  # Z_k - H_k * x_k

        new_mean = mean + np.dot(innovation, kalman_gain.T)  # X_k+ (Z_k - H_k * x_k)*K
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))  # P_k' = P_k - K'*H_k*P_k

        return new_mean, new_covariance
