## 当前内容

- `camera_tactile_zmq_sender.py`
  相机与触觉数据采集，并通过 ZMQ 发布。
- `launch_nodes.py`
  机器人节点启动脚本，支持多种机器人类型。
- `recorder_tactile.py`
  订阅相机和触觉流，并记录 episode 数据。

## 建议目录约定

- `configs/`
  放后续的参数配置、设备 IP、实验 YAML 或 JSON。
- `data/`
  放本地采集结果、临时录制数据，不建议直接上传到 Git。
- `docs/`
  放实验说明、接线记录、接口说明。
- `logs/`
  放运行日志和调试输出。

## 依赖说明

当前脚本涉及以下 Python 依赖：

- `numpy`
- `pyzmq`
- `opencv-python`
- `pyserial`
- `pyrealsense2`
- `grpcio`
- `tyro`

另外，代码还依赖你本地环境中的以下项目或生成文件：

- `gello`
- `polymetis_pb2`
- `polymetis_pb2_grpc`
- 部分机器人/相机硬件驱动环境

