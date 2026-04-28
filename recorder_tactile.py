import time
import threading
from pathlib import Path
import numpy as np
import zmq
import pickle
import signal
import sys

import grpc
import polymetis_pb2
import polymetis_pb2_grpc

print("[IMPORT] recorder_tactile.py loaded")
# ============================================================
# Unified ZMQ subscriber for camera + tactile
# ============================================================
class MultiModalZMQSubscriber:
    """
    Sender format (camera and tactile are both):
        recv_multipart(): [topic, ts(float64), payload]

    camera payload:
        jpeg bytes

    tactile distributed payload:
        flattened float32 bytes, usually shape (77, 3)
    """

    def __init__(
        self,
        addr: str,
        camera_topics=("global", "left", "right"),
        tactile_topics=(
            "tactile/distributed/sensor_0",
            "tactile/distributed/sensor_1",
            "tactile/distributed/sensor_2",
            "tactile/distributed/sensor_3",
        ),
    ):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(addr)

        self.camera_topics = tuple(camera_topics)
        self.tactile_topics = tuple(tactile_topics)

        for t in self.camera_topics + self.tactile_topics:
            self.sock.setsockopt_string(zmq.SUBSCRIBE, t)

        self._latest_camera = {t: None for t in self.camera_topics}
        self._latest_camera_ts = {t: None for t in self.camera_topics}

        self._latest_tactile = {t: None for t in self.tactile_topics}
        self._latest_tactile_ts = {t: None for t in self.tactile_topics}

        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _decode_tactile_payload(self, payload: bytes):
        arr = np.frombuffer(payload, dtype=np.float32).copy()
        if arr.size == 0:
            return None
        if arr.size % 3 == 0:
            return arr.reshape(-1, 3)
        return arr

    def _loop(self):
        while True:
            try:
                topic_b, ts_bytes, payload = self.sock.recv_multipart()
                topic = topic_b.decode()
                ts = float(np.frombuffer(ts_bytes, dtype=np.float64)[0])

                with self._lock:
                    if topic in self._latest_camera:
                        self._latest_camera[topic] = payload
                        self._latest_camera_ts[topic] = ts
                    elif topic in self._latest_tactile:
                        tactile = self._decode_tactile_payload(payload)
                        self._latest_tactile[topic] = tactile
                        self._latest_tactile_ts[topic] = ts
            except Exception:
                pass

    def get_latest_camera(self, cam):
        with self._lock:
            return self._latest_camera.get(cam), self._latest_camera_ts.get(cam)

    def get_latest_tactile(self, topic):
        with self._lock:
            data = self._latest_tactile.get(topic)
            ts = self._latest_tactile_ts.get(topic)
            if isinstance(data, np.ndarray):
                data = data.copy()
            return data, ts


# ============================================================
# Teleoperation Episode Recorder
# ============================================================
class TeleopRecorder:
    """
    一个 episode -> 一个 pkl

    obs.state.arm          : Franka joint_positions (14)
    obs.state.arm_torques  : Franka joint_torques (14)
    obs.state.hand         : Robotiq gripper width (left, right)
    obs.tactile.sensor_A   : distributed tactile array, typically (77, 3)
    obs.tactile.sensor_B   : distributed tactile array, typically (77, 3)
    action.arm             : teleop arm command
    action.hand            : teleop gripper command
    """

    def __init__(
        self,
        out_dir: str,
        hz: int,
        camera_addr: str,
        gripper_left_addr: str = "127.0.0.1:60001",
        gripper_right_addr: str = "127.0.0.1:60002",
        tactile_topics=(
            "tactile/distributed/sensor_0",
            "tactile/distributed/sensor_1",
            "tactile/distributed/sensor_2",
            "tactile/distributed/sensor_3",
        ),
    ):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.root = Path(out_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

        self.pkl_path = self.root / f"episode_{ts}.pkl"
        self.hz = hz
        self.start_time = time.time()
        self.tactile_topics = tuple(tactile_topics)

        # camera + tactile are from the same ZMQ sender
        self.streams = MultiModalZMQSubscriber(
            addr=camera_addr,
            camera_topics=("global", "left", "right"),
            tactile_topics=self.tactile_topics,
        )

        # real gripper state
        self.gripper_left = polymetis_pb2_grpc.GripperServerStub(
            grpc.insecure_channel(gripper_left_addr)
        )
        self.gripper_right = polymetis_pb2_grpc.GripperServerStub(
            grpc.insecure_channel(gripper_right_addr)
        )

        self.episode = {
            "meta": {
                "start_time": self.start_time,
                "hz": hz,
                "stream_addr": camera_addr,
                "gripper_left": gripper_left_addr,
                "gripper_right": gripper_right_addr,
                "action": {
                    "arm_joints": 14,
                    "hand_joints": 2,
                },
                "obs": {
                    "state": {
                        "arm_joints": 14,
                        "arm_torques": 14,
                        "hand_joints": 2,
                    },
                    "camera_topics": ["global", "left", "right"],
                    "tactile_topics": list(self.tactile_topics),
                },
            },
            "steps": [],
        }

        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

        print(f"[REC] Recording episode -> {self.pkl_path}")
        print("[REC] obs.hand source = Robotiq gripper gRPC")
        print(f"[REC] tactile topics = {self.tactile_topics}")

    def _get_gripper_widths(self):
        try:
            gl = self.gripper_left.GetState(polymetis_pb2.Empty()).width
        except Exception:
            gl = np.nan

        try:
            gr = self.gripper_right.GetState(polymetis_pb2.Empty()).width
        except Exception:
            gr = np.nan

        return np.array([gl, gr], dtype=np.float32)

    def teleop_hand_to_gripper_width(self, u, max_width: float = 0.085):
        u = np.asarray(u, dtype=np.float32)
        u = np.clip(a=u, a_min=0.0, a_max=1.0)
        return (1.0 - u) * max_width

    def _get_tactile_snapshot(self):
        tactile_obs = {}
        for topic in self.tactile_topics:
            data, ts = self.streams.get_latest_tactile(topic)
            key = topic.split("/")[-1]
            tactile_obs[key] = {
                "topic": topic,
                "value": data,
                "timestamp": ts,
            }
        return tactile_obs

    def record(self, *, action, obs):
        action = np.asarray(action, dtype=np.float32)

        q = np.asarray(obs["joint_positions"], dtype=np.float32)
        q_t = np.asarray(obs["joint_torques"], dtype=np.float32)
        if q.shape[0] != 16:
            raise ValueError(f"Expected obs['joint_positions'] to be 16D, got {q.shape}")

        # joint_positions: 16D = 7 + 1 + 7 + 1
        q_arm = np.concatenate([q[0:7], q[8:15]]).astype(np.float32)

        # joint_torques: 有些环境直接给 14D arm-only torque，
        # 有些环境给 16D（含 gripper 维）
        if q_t.shape[0] == 14:
            q_t_arm = q_t.astype(np.float32)
        elif q_t.shape[0] == 16:
            q_t_arm = np.concatenate([q_t[0:7], q_t[8:15]]).astype(np.float32)
        else:
            raise ValueError(
                f"Expected obs['joint_torques'] to be 14D or 16D, got {q_t.shape}"
            )

        q_hand = self._get_gripper_widths()
        hand_gripper = self.teleop_hand_to_gripper_width(action[[7, 15]], max_width=0.085)

        step = {
            "t": time.time(),
            "action": {
                "arm": np.concatenate([action[0:7], action[8:15]]).copy(),
                "hand": action[[7, 15]].copy(),
                "hand_gripper": hand_gripper.copy(),
            },
            "obs": {
                "state": {
                    "arm": q_arm.copy(),
                    "arm_torques": q_t_arm.copy(),
                    "hand": q_hand.copy(),
                },
                "camera": {},
                "tactile": self._get_tactile_snapshot(),
            },
        }

        for cam in ("global", "left", "right"):
            jpeg, ts = self.streams.get_latest_camera(cam)
            step["obs"]["camera"][cam] = {
                "jpeg": jpeg,
                "timestamp": ts,
            }

        self.episode["steps"].append(step)

    def save(self):
        with open(self.pkl_path, "wb") as f:
            pickle.dump(self.episode, f)

        print(f"\n[REC] Episode saved: {self.pkl_path}")
        print(f"[REC] Total steps: {len(self.episode['steps'])}")

    def _handle_exit(self, signum, frame):
        self.save()
        sys.exit(0)