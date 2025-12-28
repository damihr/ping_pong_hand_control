import cv2
import mediapipe as mp
import numpy as np
import random
import time
import pyautogui

# --- Setup Hand Tracking ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Get screen dimensions for fullscreen ---
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# --- Game Variables ---
# Use screen dimensions for fullscreen
width, height = screen_width, screen_height
ball_radius = 25
ball_x, ball_y = width // 2, height // 2
# Even faster ball speed
ball_dx, ball_dy = 28, 22

paddle_height = 180
paddle_width = 30
left_y, right_y = height // 2, height // 2
left_score, right_score = 0, 0

# Smoothing for paddle movement (higher = more responsive, less smoothing)
paddle_smoothing_1p = 0.7  # 1 player mode
paddle_smoothing_2p = 0.85  # 2 player mode (more responsive)
left_y_smooth = height // 2
right_y_smooth = height // 2

# Game mode: 1 = 1 player (bot), 2 = 2 player
game_mode = 1  # Default to 1 player
bot_difficulty = 1.0  # Bot reaction speed (0.0 = slow, 1.0 = perfect) - set to perfect

# --- Open Webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Create fullscreen window ---
cv2.namedWindow("🏓 Air Pong", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("🏓 Air Pong", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Mode Selection Screen ---
def show_mode_selection():
    """Display mode selection screen"""
    selection_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Title
    cv2.putText(selection_frame, "AIR PONG", 
                (width // 2 - 200, height // 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    
    # Mode options
    cv2.putText(selection_frame, "Press 1: 1 Player (vs Bot)", 
                (width // 2 - 250, height // 2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(selection_frame, "Press 2: 2 Player", 
                (width // 2 - 200, height // 2 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    
    cv2.putText(selection_frame, "Press Q to quit", 
                (width // 2 - 150, height - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    cv2.imshow("🏓 Air Pong", selection_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            return 1
        elif key == ord('2'):
            return 2
        elif key == ord('q'):
            return 0

# --- Bot AI Function ---
def update_bot_paddle(ball_x, ball_y, ball_dx, ball_dy, paddle_x, paddle_center_y, paddle_height):
    """AI bot that follows the ball intelligently - returns new center y position"""
    # Predict where ball will be when it reaches the paddle
    if ball_dx > 0:  # Ball moving towards right paddle
        # Calculate time until ball reaches paddle
        distance_to_paddle = paddle_x - ball_x
        if distance_to_paddle > 0 and ball_dx > 0:
            time_to_reach = distance_to_paddle / abs(ball_dx)
            
            # Predict ball's y position when it reaches paddle
            predicted_y = ball_y + (ball_dy * time_to_reach)
            
            # Account for bounces off top/bottom
            while predicted_y < 0 or predicted_y > height:
                if predicted_y < 0:
                    predicted_y = -predicted_y
                    ball_dy = -ball_dy  # Reverse direction
                elif predicted_y > height:
                    predicted_y = 2 * height - predicted_y
                    ball_dy = -ball_dy  # Reverse direction
            
            # Target is center of paddle aligned with predicted ball position
            target_y = predicted_y
        else:
            # Ball moving away, return to center
            target_y = height // 2
    else:
        # Ball moving away from bot, return to center
        target_y = height // 2
    
    # Perfect movement - instantly move to target position
    current_center = float(paddle_center_y)
    move_speed = 50  # Very fast movement (almost instant)
    diff = target_y - current_center
    
    # Move towards target with high speed
    if abs(diff) > 1:  # Move even for small differences
        sign = 1 if diff > 0 else -1
        new_center = current_center + sign * min(abs(diff), move_speed)
    else:
        new_center = target_y  # Snap to exact position if close
    
    # Keep paddle within bounds
    new_center = max(paddle_height // 2, min(new_center, height - paddle_height // 2))
    
    return float(new_center)  # Return center position as regular float

print("🏓 Air Pong - Select game mode...")

# Show mode selection
selected_mode = show_mode_selection()
if selected_mode == 0:
    cap.release()
    cv2.destroyAllWindows()
    exit()

game_mode = selected_mode
mode_text = "1 Player (vs Bot)" if game_mode == 1 else "2 Player"
print(f"🏓 Air Pong started! Mode: {mode_text}")
print("Move your hands up/down to hit the ball. Press M to change mode, Q to quit.")

# Frame rate control for smoothness
fps_target = 60
frame_time = 1.0 / fps_target
last_time = time.time()

while True:
    current_time = time.time()
    delta_time = current_time - last_time
    
    # Frame rate limiting
    if delta_time < frame_time:
        time.sleep(frame_time - delta_time)
        delta_time = frame_time
    
    last_time = time.time()
    
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Create game frame with webcam as background (resized to fullscreen)
    frame_resized = cv2.resize(frame, (width, height))
    # Brighter background - more visible webcam
    game_frame = cv2.addWeighted(frame_resized, 0.75, 
                                  np.zeros((height, width, 3), dtype=np.uint8), 0.25, 0)
    
    # Process hand tracking on resized frame for proper landmark drawing
    rgb_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_resized)

    # --- Track hands with smoothing and draw landmarks ---
    if result.multi_hand_landmarks:
        hands_data = []
        
        # Collect all hand positions
        for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
            lm = hand_landmarks.landmark
            cx = int(lm[9].x * width)  # x position of palm center
            cy = int(lm[9].y * height)  # y position of palm center
            label = hand_label.classification[0].label
            hands_data.append((cx, cy, hand_landmarks, label))
            # Draw hand landmarks
            mp_draw.draw_landmarks(game_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # In 1 player mode, only track left hand (player)
        if game_mode == 1:
            # Use the hand labeled as "Left" by MediaPipe
            for cx, cy, hand_landmarks, label in hands_data:
                if label == "Left":
                    smoothing = paddle_smoothing_1p
                    left_y_smooth = left_y_smooth * (1 - smoothing) + cy * smoothing
                    break
        else:
            # 2 player mode: assign hands to paddles based on x-position
            # Left side of screen (left paddle) vs right side (right paddle)
            if len(hands_data) == 1:
                # Only one hand detected - use x position to determine which paddle
                cx, cy, _, _ = hands_data[0]
                if cx < width // 2:
                    # Left side -> left paddle
                    smoothing = paddle_smoothing_2p
                    left_y_smooth = left_y_smooth * (1 - smoothing) + cy * smoothing
                else:
                    # Right side -> right paddle
                    smoothing = paddle_smoothing_2p
                    right_y_smooth = right_y_smooth * (1 - smoothing) + cy * smoothing
            elif len(hands_data) == 2:
                # Two hands detected - assign based on x position
                # Sort by x position: leftmost -> left paddle, rightmost -> right paddle
                hands_data.sort(key=lambda h: h[0])
                # Leftmost hand -> left paddle
                _, cy_left, _, _ = hands_data[0]
                smoothing = paddle_smoothing_2p
                left_y_smooth = left_y_smooth * (1 - smoothing) + cy_left * smoothing
                # Rightmost hand -> right paddle
                _, cy_right, _, _ = hands_data[1]
                right_y_smooth = right_y_smooth * (1 - smoothing) + cy_right * smoothing
    
    # Update paddle positions from smoothed values (even if hand not detected this frame)
    if game_mode == 1:
        left_y = int(left_y_smooth)
    else:
        # 2 player mode: update both paddles
        left_y = int(left_y_smooth)
        right_y = int(right_y_smooth)

    # --- Bot AI for 1 player mode ---
    if game_mode == 1:
        right_paddle_x = width - 50 - paddle_width
        right_center = right_y + paddle_height // 2
        right_center = update_bot_paddle(ball_x, ball_y, ball_dx, ball_dy, 
                                         right_paddle_x, right_center, paddle_height)
        right_y = int(right_center - paddle_height // 2)

    # --- Draw paddles (centered on screen) ---
    left_paddle_x = 50
    right_paddle_x = width - 50 - paddle_width
    
    left_top = int(max(0, min(left_y - paddle_height // 2, height - paddle_height)))
    right_top = int(max(0, min(right_y - paddle_height // 2, height - paddle_height)))
    
    # Draw paddles with gradient effect
    cv2.rectangle(game_frame, 
                  (left_paddle_x, left_top), 
                  (left_paddle_x + paddle_width, left_top + paddle_height), 
                  (0, 255, 255), -1)
    cv2.rectangle(game_frame, 
                  (right_paddle_x, right_top),
                  (right_paddle_x + paddle_width, right_top + paddle_height), 
                  (255, 0, 255), -1)
    
    # Paddle borders
    cv2.rectangle(game_frame, 
                  (left_paddle_x, left_top), 
                  (left_paddle_x + paddle_width, left_top + paddle_height), 
                  (255, 255, 255), 2)
    cv2.rectangle(game_frame, 
                  (right_paddle_x, right_top),
                  (right_paddle_x + paddle_width, right_top + paddle_height), 
                  (255, 255, 255), 2)
    
    # Show "BOT" label on right paddle in 1 player mode
    if game_mode == 1:
        cv2.putText(game_frame, "BOT", 
                    (right_paddle_x - 40, right_top + paddle_height // 2 + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Move ball ---
    ball_x += ball_dx
    ball_y += ball_dy
    
    # Safety check: prevent ball from getting stuck (if ball somehow gets inside paddle, force it out)
    if ball_x < left_paddle_x + paddle_width + ball_radius and ball_x > left_paddle_x - ball_radius:
        if left_top <= ball_y <= left_top + paddle_height:
            if ball_dx < 0:  # Moving left, push it right
                ball_x = left_paddle_x + paddle_width + ball_radius + 5
                ball_dx = abs(ball_dx)
    
    if ball_x > right_paddle_x - ball_radius and ball_x < right_paddle_x + paddle_width + ball_radius:
        if right_top <= ball_y <= right_top + paddle_height:
            if ball_dx > 0:  # Moving right, push it left
                ball_x = right_paddle_x - ball_radius - 5
                ball_dx = -abs(ball_dx)

    # --- Collision with top/bottom (with proper boundary check) ---
    if ball_y - ball_radius <= 0:
        ball_y = ball_radius
        ball_dy = abs(ball_dy)
    elif ball_y + ball_radius >= height:
        ball_y = height - ball_radius
        ball_dy = -abs(ball_dy)

    # --- Collision with paddles (improved detection with proper bounds checking) ---
    # Left paddle - check if ball is within paddle bounds
    ball_left = ball_x - ball_radius
    ball_right = ball_x + ball_radius
    ball_top = ball_y - ball_radius
    ball_bottom = ball_y + ball_radius
    
    paddle_left_left = left_paddle_x
    paddle_left_right = left_paddle_x + paddle_width
    paddle_left_top = left_top
    paddle_left_bottom = left_top + paddle_height
    
    # Left paddle collision - check if ball overlaps with paddle
    if (ball_right >= paddle_left_left and ball_left <= paddle_left_right and
            ball_bottom >= paddle_left_top and ball_top <= paddle_left_bottom):
        # Only bounce if ball is moving towards the paddle
        if ball_dx < 0:
            # Calculate hit position on paddle (affects angle)
            hit_pos = (ball_y - left_top) / paddle_height  # 0 to 1
            # Ensure ball is outside paddle to prevent getting stuck
            ball_x = left_paddle_x + paddle_width + ball_radius + 1
            ball_dx = abs(ball_dx) + 2  # Speed up more
            # Angle based on where ball hits paddle
            ball_dy = int((hit_pos - 0.5) * 10) + random.choice([-3, -2, -1, 1, 2, 3])
            ball_dy = np.clip(ball_dy, -28, 28)  # Limit max speed

    # Right paddle - check if ball is within paddle bounds
    paddle_right_left = right_paddle_x
    paddle_right_right = right_paddle_x + paddle_width
    paddle_right_top = right_top
    paddle_right_bottom = right_top + paddle_height
    
    # Right paddle collision - check if ball overlaps with paddle
    if (ball_right >= paddle_right_left and ball_left <= paddle_right_right and
            ball_bottom >= paddle_right_top and ball_top <= paddle_right_bottom):
        # Only bounce if ball is moving towards the paddle
        if ball_dx > 0:
            # Calculate hit position on paddle
            hit_pos = (ball_y - right_top) / paddle_height
            # Ensure ball is outside paddle to prevent getting stuck
            ball_x = right_paddle_x - ball_radius - 1
            ball_dx = -(abs(ball_dx) + 2)  # Speed up more
            # Angle based on where ball hits paddle
            ball_dy = int((hit_pos - 0.5) * 10) + random.choice([-3, -2, -1, 1, 2, 3])
            ball_dy = np.clip(ball_dy, -28, 28)  # Limit max speed

    # --- Scoring ---
    if ball_x < 0:
        right_score += 1
        ball_x, ball_y = width // 2, height // 2
        ball_dx = abs(ball_dx) if ball_dx < 0 else ball_dx
        ball_dy = random.choice([-18, -16, 16, 18])
        # Reset paddle smoothing
        left_y_smooth = height // 2
        right_y_smooth = height // 2
    elif ball_x > width:
        left_score += 1
        ball_x, ball_y = width // 2, height // 2
        ball_dx = -abs(ball_dx) if ball_dx > 0 else ball_dx
        ball_dy = random.choice([-18, -16, 16, 18])
        # Reset paddle smoothing
        left_y_smooth = height // 2
        right_y_smooth = height // 2

    # --- Draw ball with glow effect ---
    cv2.circle(game_frame, (int(ball_x), int(ball_y)), ball_radius + 3, (0, 200, 0), -1)
    cv2.circle(game_frame, (int(ball_x), int(ball_y)), ball_radius, (0, 255, 0), -1)
    cv2.circle(game_frame, (int(ball_x), int(ball_y)), ball_radius - 5, (150, 255, 150), -1)

    # --- Display scores (centered) ---
    score_font_scale = 4
    score_thickness = 8
    left_score_text = str(left_score)
    right_score_text = str(right_score)
    
    # Get text size for centering
    (left_w, left_h), _ = cv2.getTextSize(left_score_text, cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, score_thickness)
    (right_w, right_h), _ = cv2.getTextSize(right_score_text, cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, score_thickness)
    
    cv2.putText(game_frame, left_score_text, 
                (width // 4 - left_w // 2, height // 4), 
                cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, (0, 255, 255), score_thickness)
    cv2.putText(game_frame, right_score_text, 
                (3 * width // 4 - right_w // 2, height // 4), 
                cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, (255, 0, 255), score_thickness)

    # --- Middle divider (centered perfectly) ---
    center_x = width // 2
    dash_height = 30
    dash_gap = 20
    for y in range(0, height, dash_height + dash_gap):
        cv2.rectangle(game_frame, 
                      (center_x - 3, y), 
                      (center_x + 3, y + dash_height), 
                      (255, 255, 255), -1)

    # --- Draw center line ---
    cv2.line(game_frame, (center_x, 0), (center_x, height), (100, 100, 100), 1)

    # --- Instructions ---
    mode_display = "1 Player (vs Bot)" if game_mode == 1 else "2 Player"
    cv2.putText(game_frame, f"Mode: {mode_display} | M=change mode | Q=quit", 
                (width // 2 - 300, height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("🏓 Air Pong", game_frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('m') or key == ord('M'):
        # Switch mode
        game_mode = 2 if game_mode == 1 else 1
        mode_text = "1 Player (vs Bot)" if game_mode == 1 else "2 Player"
        print(f"Mode changed to: {mode_text}")
        # Reset scores and ball
        left_score = 0
        right_score = 0
        ball_x, ball_y = width // 2, height // 2
        ball_dx = 28 if ball_dx > 0 else -28
        ball_dy = random.choice([-18, -16, 16, 18])
        left_y_smooth = height // 2
        right_y_smooth = height // 2

cap.release()
cv2.destroyAllWindows()
