# Numba 加速版：部署与验证

适用于现有两台机器人的本地 MMC + MHE 架构。每台都放整份 mmc_onboard；实验启动程序仍只运行一份。

## 这次改动

- growMHE.py：沿用原版缓冲区、热启动和输出接口，调用编译后的求解内核。
- mhe_numba_kernels.py：RK4、连续协态反向积分、梯度和线搜索使用 Numba；一次求解内复用工作数组。float64，未开启 fastmath。
- growMHE_reference.py：原版 Python MHE，同时提供公共类和缓冲逻辑，必须保留。
- mmc_worker.py：待机阶段用独立临时估计器完成预编译；不会把合成数据写入真实估计器。编译失败不会报告 ready。
- mmc_local.py：预编译完成打印 Numba ready；停车原因细分测量过期、估计输出过期、MHE校正过期或非有限值。停车后不重复打印旧耗时。
- benchmark_mhe.py：不依赖 ROS 的合成数据测速，可与参考版比较。
- test_numba.py：轨迹、协态、终端项、预测和预编译隔离测试。

Nh=30、nSub=5、启动最大120次/运行最大10次迭代、控制公式、启动协议、0.5秒估计阈值均保持原值。仍串行处理所有输入，不跳过排队测量。没有更换求解算法。

## 1. 停止旧实验与控制器

先退出实验启动程序和两台 MMC 控制器，再覆盖文件。传感器与 detector 可以保持运行。保留现有自定义 ROS 环境设置；本包 run_local.sh 默认 source ~/turtlebot3_ws/install/setup.bash。

## 2. 安装依赖（两台各执行）

面向 Ubuntu 22.04 的系统 Python3，先尝试发行版匹配的包：

```bash
sudo apt update
sudo apt install -y python3-numba python3-scipy unzip
python3 -c "import numpy, scipy, numba; print('numpy',numpy.__version__,'numba',numba.__version__)"
```

如果包找不到或导入失败，保留报错，并提供 `python3 --version`、`uname -m` 输出；不要盲目升级 ROS 使用的 NumPy。这里测试环境为 x86_64 / Python3.12 / Numba0.67，尚未在你的 Pi 上验证版本兼容性。

## 3. 从笔记本传输

在 zip 所在目录打开笔记本终端，将地址替换为实际机器人 IP（若 burger1/burger3 主机名能解析，可直接用主机名）：

```bash
scp mmc_onboard_numba_bundle.zip burger1@BURGER1_IP:~/
scp mmc_onboard_numba_bundle.zip burger3@BURGER3_IP:~/
```

然后在每台机器人上备份并解压：

```bash
cp -a ~/mmc_onboard "${HOME}/mmc_onboard_backup_$(date +%Y%m%d_%H%M%S)"
unzip -o ~/mmc_onboard_numba_bundle.zip -d ~/
```

若旧目录不叫 mmc_onboard，先把备份命令中的源路径换成实际目录。zip 内已经包含 mmc_onboard 这一层目录，不要再解压到 ~/mmc_onboard 下。若旧 run_local.sh 增加过消息工作空间 source，请把相同设置加回新脚本。

## 4. 先离线测速（两台各执行）

保留通常的雷达与 detector 负载，但 MMC 控制器暂不运行：

```bash
cd ~/mmc_onboard
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmark_mhe.py --compare
python3 test_numba.py
python3 test_onboard.py
```

首次编译可能较久，后续在同一机器使用缓存；包中不附带电脑的编译缓存。--compare 会额外运行慢参考版，可耗时数十秒或更久。以上程序均不发布速度。

把 growing/full window 的 median、p95、max，以及 comparison PASS 输出保存下来。合成数据性能不能保证现场所有输入的耗时；最终需同时核对实际测量、worker输出和最后校正的年龄。

## 5. 按原流程启动

两台各自运行自己的 all.sh 与 detector（namespace 和 scan 用各自的 burger1/burger3）。detector 与 all.sh 不包含在此包内。

burger1：

```bash
bash ~/mmc_onboard/run_local.sh burger1
```

burger3：

```bash
bash ~/mmc_onboard/run_local.sh burger3
```

两台都出现 `Numba ready; precompile ...; waiting for experiment start` 后，且 detector 有有效目标，再在其中一台运行：

```bash
bash ~/mmc_onboard/start_experiment.sh
```

保持启动程序运行；Ctrl+C 结束实验。完整传感器与命名空间命令见 README_zh.md。

若仍停车，请提供两台 STOPPED 行及新 CSV；现在日志会区分到底哪一种年龄超限，不要仅通过放大阈值掩盖积压。

## 验证记录（2026-09-13，本地电脑，非 Pi）

- 原协议/控制公式/真实子进程12项测试通过；新增2项数值测试通过。
- 65次合成测量，间隔0.13秒，每次测量间3次预测，覆盖增长窗口与31样本满窗口。
- 满窗口 Numba 中位2.343ms，p95 4.811ms，最大5.345ms；参考版中位349.103ms。
- 此次回放状态输出与cost最大绝对差0；随机输入、终端交叉项、完整轨迹及协态也通过容差比较。浮点计算不承诺跨平台逐位一致。
- 尚未验证 Pi 性能、现场 ROS 消息和机器人闭环实验。安装后请先完成上述测速。
