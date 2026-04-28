#!/usr/bin/env python3
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import serial
import serial.tools.list_ports
import zmq

# ===================== 相机配置（与 camera_zmq_sender_3cams.py 一致）=====================
CAMERAS = {
    "global": "211622062326",
    "left": "018322071546",
    "right": "212622065077",
}

WIDTH = 640
HEIGHT = 480
FPS = 15
JPEG_QUALITY = 80

EXPOSURE = 120  # 固定曝光
GAIN = 32  # 固定增益

ZMQ_BIND = "tcp://0.0.0.0:5555"
ZMQ_SNDHWM = 3

# ===================== 触觉（GEN3 + Serial converter board + M3025）=====================
ENABLE_TACTILE = True
TACTILE_BAUDRATE = 921600

# 多串口（无串联）：每个传感器独占一个 /dev 串口端口。
# 这种情况下，每个 port 只轮询一个设备地址（通常为 01）。
# 传感器之间通过 name 区分，name 将作为 topic 后缀：
#   - tactile/resultant/<name>
#   - tactile/distributed/<name>
# sudo chmod 666 /dev/ttyACM0
# sudo chmod 666 /dev/ttyACM1
# sudo chmod 666 /dev/ttyACM2
# sudo chmod 666 /dev/ttyACM3
TACTILE_BUSES = [
    {"port": "/dev/ttyACM0", "sensors": [{"name": "sensor_0", "device_addr": "02"}]},
    {"port": "/dev/ttyACM1", "sensors": [{"name": "sensor_1", "device_addr": "01"}]},
    {"port": "/dev/ttyACM2", "sensors": [{"name": "sensor_2", "device_addr": "02"}]},
    {"port": "/dev/ttyACM3", "sensors": [{"name": "sensor_3", "device_addr": "01"}]},
]

TACTILE_DIST_LEN = 231  # M3025：77点*3字节
TACTILE_RESULT_LEN = 3
TACTILE_CMD_TIMEOUT_S = 0.2
TACTILE_LOOP_INTERVAL_S = 0.0

TACTILE_TOPIC_RESULTANT_PREFIX = "tactile/resultant/"
TACTILE_TOPIC_DISTRIBUTED_PREFIX = "tactile/distributed/"


def start_camera(serial_number: str):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial_number)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)

    sensor = profile.get_device().query_sensors()[0]
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, EXPOSURE)
    sensor.set_option(rs.option.gain, GAIN)
    return pipe


def init_cameras():
    ctx = rs.context()
    found = {dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()}

    pipes = {}
    for name, serial_number in CAMERAS.items():
        if serial_number not in found:
            print(f"[WARN] {name} not found ({serial_number})")
            continue
        try:
            pipes[name] = start_camera(serial_number)
            print(f"[OK] {name} camera started ({serial_number})")
        except Exception as e:
            print(f"[ERR] Failed to start {name}: {e}")
    return pipes


def _calculate_lrc(data: bytes) -> int:
    lrc = 0
    for b in data:
        lrc = (lrc + b) & 0xFF
    lrc = ((~lrc) + 1) & 0xFF
    return lrc


def _build_frames(device_addr: str, dist_len: int) -> Dict[str, str]:
    len_low = dist_len & 0xFF
    len_high = (dist_len >> 8) & 0xFF
    len_hex = f"{len_low:02X}{len_high:02X}"  # 小端：低位在前

    calibration = f"55 AA 0A 00 {device_addr} 00 79 03 00 00 00 01 00 01"
    resultant = f"55 AA 09 00 {device_addr} 00 FB F0 03 00 00 03 00"
    distributed = (
        f"55 AA 09 00 {device_addr} 00 FB 0E 04 00 00 {len_hex[0:2]} {len_hex[2:4]}"
    )
    return {
        "calibration": calibration,
        "resultant_force": resultant,
        "distributed_force": distributed,
    }


def _send_command(
    ser: serial.Serial,
    frame_hex: str,
    timeout_s: float,
    expected_min_len: Optional[int] = None,
) -> Optional[bytes]:
    frame = frame_hex.replace(" ", "")
    base = bytes.fromhex(frame)
    lrc = _calculate_lrc(base)
    payload = base + bytes([lrc])

    ser.reset_input_buffer()
    ser.write(payload)

    resp = b""
    start = time.time()
    while time.time() - start < timeout_s:
        if ser.in_waiting > 0:
            resp += ser.read(ser.in_waiting)
            if expected_min_len is not None and len(resp) >= expected_min_len:
                break
        time.sleep(0.0001)
    return resp if resp else None


def _parse_resultant_force_3b(data: bytes) -> Optional[Tuple[float, float, float]]:
    if len(data) != 3:
        return None
    b1, b2, b3 = data[0], data[1], data[2]

    def signed(x: int) -> int:
        return x if x <= 127 else x - 256

    v1 = signed(b1)
    v2 = signed(b2)
    v3 = b3
    return (v1 * 0.1, v2 * 0.1, v3 * 0.1)


def _parse_distributed_force(data: bytes, dist_len: int) -> np.ndarray:
    if len(data) < dist_len:
        dist_len = len(data)
    group = dist_len // 3
    out = np.zeros((group, 3), dtype=np.float32)

    def signed(x: int) -> int:
        return x if x <= 127 else x - 256

    for i in range(group):
        b1 = data[i * 3]
        b2 = data[i * 3 + 1]
        b3 = data[i * 3 + 2]
        out[i, 0] = signed(b1) * 0.1
        out[i, 1] = signed(b2) * 0.1
        out[i, 2] = b3 * 0.1
    return out


def _pick_default_port() -> Optional[str]:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    return ports[0].device


@dataclass
class TactileSample:
    ts: float
    resultant: Optional[Tuple[float, float, float]] = None
    distributed: Optional[np.ndarray] = None
    seq: int = 0


def _tactile_worker(
    latest: Dict[str, TactileSample],
    lock: threading.Lock,
    stop: threading.Event,
    *,
    port: str,
    sensors,
):
    try:
        ser = serial.Serial(
            port=port,
            baudrate=TACTILE_BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            write_timeout=0.1,
            inter_byte_timeout=0.0005,
            xonxoff=False,
            rtscts=False,
        )
    except Exception as e:
        print(f"[TACTILE] Failed to open serial {port}: {e}")
        return

    frames = {
        s["name"]: _build_frames(s["device_addr"], TACTILE_DIST_LEN) for s in sensors
    }
    seq = 0

    for _, cmds in frames.items():
        try:
            _send_command(ser, cmds["calibration"], timeout_s=1.5, expected_min_len=8)
        except Exception:
            pass

    print(f"[TACTILE] Serial ready: {port} @ {TACTILE_BAUDRATE}")
    while not stop.is_set():
        loop_start = time.time()
        for sensor in sensors:
            name = sensor["name"]
            cmds = frames[name]
            ts = time.time()

            try:
                resp_r = _send_command(
                    ser,
                    cmds["resultant_force"],
                    timeout_s=TACTILE_CMD_TIMEOUT_S,
                    expected_min_len=18,
                )
                res = None
                if resp_r and len(resp_r) > 14:
                    res = _parse_resultant_force_3b(
                        resp_r[14 : 14 + TACTILE_RESULT_LEN]
                    )

                resp_d = _send_command(
                    ser,
                    cmds["distributed_force"],
                    timeout_s=TACTILE_CMD_TIMEOUT_S,
                    expected_min_len=14 + TACTILE_DIST_LEN + 1,
                )
                dist = None
                if resp_d and len(resp_d) > 14:
                    dist = _parse_distributed_force(
                        resp_d[14 : 14 + TACTILE_DIST_LEN], TACTILE_DIST_LEN
                    )
                # 就加在这里
                if dist is not None:
                    print(f"[TACTILE_RX] {name}: dist shape={dist.shape}, ts={ts:.3f}")
                else:
                    print(
                        f"[TACTILE_RX] {name}: dist=None, resp_len={0 if resp_d is None else len(resp_d)}"
                    )

                with lock:
                    seq += 1
                    latest[name] = TactileSample(
                        ts=ts, resultant=res, distributed=dist, seq=seq
                    )
            except Exception:
                continue

        if TACTILE_LOOP_INTERVAL_S > 0:
            dt = time.time() - loop_start
            if dt < TACTILE_LOOP_INTERVAL_S:
                time.sleep(TACTILE_LOOP_INTERVAL_S - dt)

    try:
        ser.close()
    except Exception:
        pass


def main():
    # ========== ZMQ PUB ==========
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(ZMQ_BIND)
    sock.setsockopt(zmq.SNDHWM, ZMQ_SNDHWM)

    print("=== Starting Cameras ===")
    pipes = init_cameras()
    print(f"=== ZMQ Sender Ready ({ZMQ_BIND}) ===")
    time.sleep(1.0)

    tactile_latest: Dict[str, TactileSample] = {}
    tactile_lock = threading.Lock()
    tactile_stop = threading.Event()
    tactile_last_sent_seq: Dict[str, int] = {}
    tactile_threads: list[threading.Thread] = []
    if ENABLE_TACTILE:
        for bus in TACTILE_BUSES:
            t = threading.Thread(
                target=_tactile_worker,
                args=(tactile_latest, tactile_lock, tactile_stop),
                kwargs={"port": bus["port"], "sensors": bus["sensors"]},
                daemon=True,
            )
            t.start()
            tactile_threads.append(t)

    send_stats = {
        name: {"last_print": time.time(), "count": 0} for name in pipes.keys()
    }

    try:
        while True:
            loop_start = time.time()

            # 1) 发相机
            for name, pipe in pipes.items():
                try:
                    frames = pipe.wait_for_frames(timeout_ms=200)
                    color = frames.get_color_frame()
                    if not color:
                        continue

                    frame = np.asanyarray(color.get_data())
                    ok, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )
                    if not ok:
                        continue

                    ts = time.time()
                    sock.send_multipart(
                        [name.encode(), np.float64(ts).tobytes(), jpeg.tobytes()]
                    )

                    st = send_stats[name]
                    st["count"] += 1
                    elapsed = ts - st["last_print"]
                    if elapsed >= 1.0:
                        fps = st["count"] / elapsed
                        print(f"[SEND_FPS] {name}: {fps:.2f} Hz | size={len(jpeg)} B")
                        st["count"] = 0
                        st["last_print"] = ts
                except Exception as e:
                    print(f"[ERR] Camera {name} error: {e}")
                    print("[RECOVER] Restarting camera...")
                    try:
                        pipes[name] = start_camera(CAMERAS[name])
                    except Exception as e2:
                        print(f"[FATAL] Restart failed: {e2}")

            # 2) 发触觉（最新值，有更新才发）
            if ENABLE_TACTILE:
                with tactile_lock:
                    latest_copy = dict(tactile_latest)
                for sname, sample in latest_copy.items():
                    last_seq = tactile_last_sent_seq.get(sname, 0)
                    if sample.seq <= last_seq:
                        continue
                    tactile_last_sent_seq[sname] = sample.seq

                    ts = sample.ts
                    if sample.resultant is not None:
                        fx, fy, fz = sample.resultant
                        payload = np.asarray([fx, fy, fz], dtype=np.float32).tobytes()
                        topic = (TACTILE_TOPIC_RESULTANT_PREFIX + sname).encode()
                        sock.send_multipart([topic, np.float64(ts).tobytes(), payload])

                    if sample.distributed is not None:
                        payload = (
                            sample.distributed.astype(np.float32, copy=False)
                            .ravel()
                            .tobytes()
                        )
                        topic = (TACTILE_TOPIC_DISTRIBUTED_PREFIX + sname).encode()
                        sock.send_multipart([topic, np.float64(ts).tobytes(), payload])

            # 节奏控制（约等于相机 FPS）
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / FPS) - elapsed)
            time.sleep(sleep_time)
    finally:
        tactile_stop.set()
        for t in tactile_threads:
            t.join(timeout=1.0)


if __name__ == "__main__":
    main()
