import pyautogui
import time
import random

# ---------------------------
# SETTINGS
# ---------------------------

TOTAL_RUNTIME_HOURS = 4          # total active duration
total_runtime_seconds = TOTAL_RUNTIME_HOURS * 60 * 60
end_time = time.time() + total_runtime_seconds

# Screen size
screen_width, screen_height = pyautogui.size()

# Tiny mouse jitter settings
JITTER_MIN, JITTER_MAX = 1, 3
JITTER_SLEEP_MIN, JITTER_SLEEP_MAX = 0.5, 1.5  # seconds

# Occasional bigger moves
BIG_MOVE_MIN, BIG_MOVE_MAX = 50, 120
BIG_MOVE_STEPS_MIN, BIG_MOVE_STEPS_MAX = 20, 50
BIG_MOVE_CHANCE = 0.02  # ~2% chance each second

# Keyboard keys to press (harmless)
KEYS = ['shift', 'ctrl', 'alt']
KEY_PRESS_CHANCE = 0.05  # 5% chance each second

print(f"Ultra-reliable activity simulator started for {TOTAL_RUNTIME_HOURS} hours...")
print("Press Ctrl+C to stop manually at any time.")

# ---------------------------
# Helper function: human-like move
# ---------------------------
def human_move(start_x, start_y, end_x, end_y, steps):
    for i in range(steps):
        t = (i + 1) / steps
        # Ease-in-out curve
        t = 3*t**2 - 2*t**3
        new_x = start_x + (end_x - start_x) * t + random.uniform(-2, 2)
        new_y = start_y + (end_y - start_y) * t + random.uniform(-2, 2)
        new_x = max(0, min(screen_width - 1, new_x))
        new_y = max(0, min(screen_height - 1, new_y))
        pyautogui.moveTo(new_x, new_y, duration=random.uniform(0.01, 0.08))

# ---------------------------
# MAIN LOOP
# ---------------------------
while time.time() < end_time:
    x, y = pyautogui.position()

    # --- Tiny jitter ---
    jitter_x = random.randint(JITTER_MIN, JITTER_MAX) * random.choice([-1, 1])
    jitter_y = random.randint(JITTER_MIN, JITTER_MAX) * random.choice([-1, 1])
    new_x = max(0, min(screen_width - 1, x + jitter_x))
    new_y = max(0, min(screen_height - 1, y + jitter_y))
    pyautogui.moveTo(new_x, new_y, duration=random.uniform(0.01, 0.1))

    # --- Occasionally do a bigger human-like move ---
    if random.random() < BIG_MOVE_CHANCE:
        dx = random.randint(BIG_MOVE_MIN, BIG_MOVE_MAX) * random.choice([-1, 1])
        dy = random.randint(BIG_MOVE_MIN, BIG_MOVE_MAX) * random.choice([-1, 1])
        end_x = max(0, min(screen_width - 1, x + dx))
        end_y = max(0, min(screen_height - 1, y + dy))
        steps = random.randint(BIG_MOVE_STEPS_MIN, BIG_MOVE_STEPS_MAX)
        human_move(x, y, end_x, end_y, steps)
        x, y = pyautogui.position()  # update after big move

    # --- Occasionally press a harmless key ---
    if random.random() < KEY_PRESS_CHANCE:
        key = random.choice(KEYS)
        pyautogui.press(key)
        print(f"Pressed {key} at {time.strftime('%H:%M:%S')}")

    # Random short sleep to vary timing
    time.sleep(random.uniform(JITTER_SLEEP_MIN, JITTER_SLEEP_MAX))

print(f"{TOTAL_RUNTIME_HOURS} hours completed. Script exited.")
