# 两台 Burger 的本地 MMC + growMHE

这是 Numba 加速版，先阅读 README_NUMBA_zh.md 完成依赖安装和离线测速。

本包替代之前的中央式 mmc_mhe_ol_python.py。不要同时运行旧中央控制器。

## 每台机器人上放什么

把整个 mmc_onboard 文件夹复制到两台机器人的 home 目录：~/mmc_onboard。
两台使用同一份代码，分别通过 mmc_burger1.py 和 mmc_burger3.py 指定身份。

- growMHE.py：每台本地一个独立 MHE，算法和参数保持上次转换版本。
- mmc_burger1.py / mmc_burger3.py：对应机器人的入口。
- mmc_local.py：共同的本地 ROS2 控制器，只有一个本机 cmd_vel 发布器。
- mmc_worker.py：本机独立 MHE 求解进程，串行处理 step/predict，避免阻塞 ROS 控制回调。
- mmc_math.py：从原 MMC 原样提取的控制公式。
- mmc_protocol.py：启动、切换闭环和停车的协调状态机。
- start_mmc_experiment.py：只运行一份，放在 burger1 或 burger3 上均可。
- run_local.sh / start_experiment.sh：设置环境并启动对应程序。
- test_onboard.py：完全离线测试，不发布 ROS 速度指令。

本地控制器启动后自动创建本机 MHE 工作进程，但保持 WAIT 和零速度。
启动程序需要持续运行；Ctrl+C 发送两台 STOP。没有可靠硬件时钟同步的零误差承诺。

## 依赖和环境

使用现有 Ubuntu 22.04 / ROS2 Humble / Python3 环境，growMHE 需要 NumPy、SciPy、Numba（及其 llvmlite 依赖）。
ROS2 rclpy、geometry_msgs、std_msgs，以及构建好的 lidar_object_detection_ros2（ObjectsArray消息定义）必须可导入。
可选 mocap4r2_msgs 只用于记录真值，缺失时仍能运行。无需 MATLAB、matlabengine 或新的 MHE 求解器。
双LiDAR、merger、TF、目标检测和底盘继续使用已经调好的版本。

run_local.sh 默认 source /opt/ros/humble/setup.bash 和 ~/turtlebot3_ws/install/setup.bash。
若消息包在另一个 workspace，请在脚本中追加 source 那个 workspace 的 install/setup.bash。
两个脚本在没有现有设置时使用 ROS_DOMAIN_ID=30；两台必须使用同一个值，ROS_LOCALHOST_ONLY=0。
两台需要能通过局域网互相发现 ROS2 节点。

启动前先在两台同步系统时间（例如现有 NTP/chrony），不要在实验中突然校时。
代码用约定的未来 UTC 时间安排启动，再转换成本机 monotonic 截止时间；心跳超时用 monotonic。
默认要求收到的状态/命令时间戳与本机时间相差不超过0.25秒。这个检查包含网络延迟，不能代替准确校时；建议两台时钟偏差控制在10ms以内并用日志核对实际启动偏差。

先在每台执行不会驱动机器人的离线测试：

```bash
cd ~/mmc_onboard
python3 test_onboard.py
```

## 你的实际启动顺序

笔记本仅作为SSH终端。以下步骤都在对应机器人的SSH窗口执行。
确保 start_burger#_all.sh 已启动底盘、双LiDAR、TF及merger，且没有重复启动同一个检测器/控制器。

### burger1

窗口1：

```bash
~/start_burger1_all.sh
```

窗口2（每个新窗口都需要ROS环境）：

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
python3 ~/burger_target_detector_parallel_rho_smooth_locked_adaptive.py \
  --ros-args \
  -r __ns:=/burger1/merged \
  -r scan:=/burger1/scan_merged \
  -p flip_y_axis:=false
```

窗口3：

```bash
bash ~/mmc_onboard/run_local.sh burger1
```

显示 WAIT，保持零速度；等待 Numba ready 日志后再启动实验。

### burger3

窗口1：

```bash
~/start_burger3_all.sh
```

窗口2：

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
python3 ~/burger_target_detector_parallel_rho_smooth_locked_adaptive.py \
  --ros-args \
  -r __ns:=/burger3/merged \
  -r scan:=/burger3/scan_merged \
  -p flip_y_axis:=false
```

窗口3：

```bash
bash ~/mmc_onboard/run_local.sh burger3
```

也显示 WAIT。growMHE.py 不需要另开窗口。

### 真正开始实验（任选一台，另开一个窗口，只启动一份）

```bash
bash ~/mmc_onboard/start_experiment.sh
```

执行这条后，如果两台就绪，会自动安排开始运动。程序持续运行直到 Ctrl+C。
也可限定从运动开始算起60秒后自动停车：

```bash
bash ~/mmc_onboard/start_experiment.sh --duration 60
```

仅发送停车指令：

```bash
bash ~/mmc_onboard/start_experiment.sh --stop
```

自动到时/故障后协调程序保持发送 STOP，Ctrl+C 退出即可。
下一次实验需要退出并重新启动两个本地控制器，然后重新执行启动程序。底盘和检测器可保持运行。

## 参数

本次默认值取自上传的 mmc_mhe_ol(1).py，而非其他实验版本：

| 参数 | 默认值 |
|---|---|
| v1 / v3 | 0.15 / 0.15 m/s |
| mu | 1.46225 |
| desired_energy_E0 | 0.00284423 |
| kd | 1000 |
| sync_gain | 0.1 |
| sync_initial_fraction | 0.01 |
| sync_ramp_start / duration | 15 / 5 s |
| rho_bias（每台各自） | +0.05 m |
| bearing_sign / bearing_bias | 1 / 0 rad |
| warmup angular command | +0.30 rad/s |
| warmup timeout | 3 s（包含准备切换MMC的时间） |
| control_hz | 30 |
| max_measurement_age / max_estimate_age | 0.50 / 0.50 s |
| rho_stop / u_max | 0.15 m / 2.50 rad/s |
| heartbeat_timeout | 0.60 s |

可以用 ROS 参数覆盖本地设置，例如已确认的某台距离修正：

```bash
bash ~/mmc_onboard/run_local.sh burger1 --ros-args -p rho_bias:=0.045
```

上面只是覆盖方式示例，不代表重新标定。本包保持上传代码的+0.05m默认值。
共同的控制/速度参数需要在两台一致设置；启动程序比较参数签名，不一致时拒绝启动。
不使用 OptiTrack 日志：

```bash
bash ~/mmc_onboard/run_local.sh burger3 --ros-args -p enable_optitrack_logging:=false
```

## 协调行为

双方 LiDAR 新鲜、底盘cmd_vel有订阅者、MHE工作进程就绪，持续0.75s后进入准备。
协调器给出默认2秒后的启动时间；双方确认准备，再确认提交。到时双方进入warmup。
静止等待期间不向 MHE 填入静止测量。
双方在运动中积累数据、MHE就绪后，约定默认0.6秒后的MMC切换时间，双方确认后才切换。
本地计算只使用自己的测量和已知自身输入；网络交换启动和健康状态，不交换估计状态用于MMC控制。

任一本地测量过期、MHE工作进程失败/队列积压、估计过期、距离过近、对方或协调器心跳超时，会锁定停车。
该流程降低普通丢包/启动不同步风险，但ROS无线通信不是硬实时同步；不能保证网络分区时两台绝对同时启停。
控制进程被强制杀死时还需要现有底盘的cmd_vel超时停车能力；本程序无法在自身进程已经消失后发送零速度。

## Topics与日志

| 用途 | burger1 | burger3 |
|---|---|---|
| 本地检测输入 | /burger1/merged/lod_objects | /burger3/merged/lod_objects |
| 本地速度输出 | /burger1/cmd_vel | /burger3/cmd_vel |
| 状态心跳 | /burger1/mmc/status | /burger3/mmc/status |

协调命令：/mmc/experiment/command，std_msgs/String JSON，可靠且非持久历史QoS；有run_id、序列号和时间戳。

每台CSV和配置快照保存在 ~/mmc_logs。每条日志有共同run_id、wall_time、elapsed、状态、测量、估计、指令、MHE耗时和MMC量。
实际首次WARMUP/非零指令的wall_time可用于比较两台启动偏差。
OptiTrack可选：两台均可记录两个rigid body的原始位置/航向；它不进入估计、控制或启动判断。
CSV现在是每台各一份的新格式，不能直接当成旧中央式CSV喂给原绘图脚本；后续需要按run_id和时间合并或适配绘图。

MHE在独立进程里顺序求解和预测，主控制循环读取最近有效结果；因此存在可记录的estimate_age。
保留原代码按接收时刻给检测结果打时间戳的约定。未恢复传感器硬件时间戳；检测或ROS传输积压的原始延迟仍需单独检查。
原NumPy MHE算法和RK4/PMP参数未修改。由于进程调度，控制循环不承诺每个tick一定得到精确传播到该tick的估计。

## 验证

已运行12组离线测试：无指令不动、双方启动/切换、单边指令、过期/重放、不同run、参数不一致、心跳丢失、测量失效、MHE超时、MMC公式、实际MHE子进程。
已检查Python语法、shell语法，以及本地速度输出仅指向本机的实现。
当前环境没有ROS2、机器人硬件和MATLAB；尚未做ROS实机联调、Pi实时性测试、MATLAB与Python逐点对比。
