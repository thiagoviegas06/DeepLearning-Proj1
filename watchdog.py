"""Memory watchdog: kills training gracefully before the Mac swap-thrashes.

Runs detached (nohup). Every 20s: if system free memory < THRESHOLD, send
SIGTERM to any train.py/simclr.py process — our scripts finish the current
epoch, save checkpoints, and exit cleanly. Logs to logs/watchdog.log.
"""
import os
import subprocess
import sys
import time

THRESHOLD = 10  # percent free memory; below this = SIGTERM training
LOG = os.path.join("logs", "watchdog.log")


def free_pct():
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.strip().split()[-1].rstrip("%"))
    return 100


def training_pids():
    out = subprocess.run(["pgrep", "-f", "train.py|simclr.py"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


log(f"watchdog started (pid {os.getpid()}, threshold {THRESHOLD}%)")
while True:
    try:
        pids = training_pids()
        if pids:
            pct = free_pct()
            if pct < THRESHOLD:
                log(f"free={pct}% < {THRESHOLD}% -> SIGTERM {pids}")
                for pid in pids:
                    os.kill(pid, 15)
                time.sleep(10)
                for pid in training_pids():   # escalate if alive after grace period
                    try:
                        os.kill(pid, 9)
                        log(f"escalated to SIGKILL for {pid}")
                    except ProcessLookupError:
                        pass
                log("training stopped by watchdog")
        time.sleep(20)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        log(f"error: {e}")
        time.sleep(30)
