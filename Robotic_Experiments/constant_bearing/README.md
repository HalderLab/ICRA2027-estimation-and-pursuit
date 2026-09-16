# Modular LiDAR / Estimation / Control pipeline

This folder is a refactor of `2_agents_lidar_ekf_cbp_optitrack_bias.py` into replaceable blocks.

## Files

- `main_modular.py` — ROS2 orchestrator only.
- `lidar_block.py` — target selection, bearing sign, LiDAR bias correction.
- `estimator_ekf.py` — current 3-state EKF.
- `controller_cbp.py` — constant-bearing controller.
- `cmd_vel_block.py` — publishes `/cmd_vel` only.
- `optitrack_block.py` — OptiTrack parsing, history, timestamp interpolation.
- `experiment_logger.py` — collects data from the other blocks and writes synchronized CSV rows.
- `models.py` — shared data containers.
- `utils.py` — angle/time helper functions.

## Data flow
LiDAR -> LidarBlock -> Estimator -> Controller -> CmdVelBlock -> robots
   |                       |            |             |
   +-----------------------+------------+-------------+--> ExperimentLogger

OptiTrack -> OptiTrackBlock --------------------------------> ExperimentLogger
```

## Why the estimator interface is important

`EKFEstimator` exposes:

```python
step(t, rho, alpha2, v_self, u_self)
predict(t, v_self, u_self)
close()
```

A future MATLAB-MHE wrapper can expose the same three methods. Then the main ROS2 pipeline and controller do not need to know whether the estimator is EKF, PF, or MHE.

## Behavior intentionally preserved

The original source uses `alpha1_hat = -x_hat[1]` inside the constant-bearing controller. This refactor preserves it through the parameter:

controller_alpha1_sign = -1.0

This makes the sign convention explicit instead of hiding it inside the controller equation.
