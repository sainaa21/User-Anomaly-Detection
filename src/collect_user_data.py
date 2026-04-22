import time
import pandas as pd
from pynput import keyboard, mouse
import os

key_timestamps = []
click_count = 0
start_time = time.time()

USER_ID = 1   # change if needed

def on_key_press(key):
    key_timestamps.append(time.time())

def on_click(x, y, button, pressed):
    global click_count
    if pressed:
        click_count += 1

keyboard_listener = keyboard.Listener(on_press=on_key_press)
mouse_listener = mouse.Listener(on_click=on_click)

keyboard_listener.start()
mouse_listener.start()

print("🔴 Recording... Use system normally for 20 seconds")

RECORD_TIME = 20
time.sleep(RECORD_TIME)

keyboard_listener.stop()
mouse_listener.stop()

end_time = time.time()
session_duration = end_time - start_time

# typing speed (keys per minute)
typing_speed = len(key_timestamps) * (60 / session_duration)

# average delay between keystrokes
if len(key_timestamps) > 1:
    delays = [
        key_timestamps[i+1] - key_timestamps[i]
        for i in range(len(key_timestamps) - 1)
    ]
    avg_key_delay = sum(delays) / len(delays)
else:
    avg_key_delay = 0

# clicks per second
click_rate = click_count / session_duration

row = {
    "user_id": USER_ID,
    "typing_speed": typing_speed,
    "avg_key_delay": avg_key_delay,
    "click_rate": click_rate,
    "session_duration": session_duration,
    "label": 0   # YOU = normal
}

df = pd.DataFrame([row])

file_path = "data/real_user_sessions.csv"

if os.path.exists(file_path):
    df.to_csv(file_path, mode='a', header=False, index=False)
else:
    df.to_csv(file_path, index=False)

print("\n✅ Session saved!")
print(df)