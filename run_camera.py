import cv2
from ultralytics import YOLO
import math

# ==============================================================================
# 1. CONFIGURATION & THRESHOLDS
# ==============================================================================
MODEL_PATH = 'best.pt'
CAMERA_INDEX = 0       # 0 for Laptop, 1 for USB Webcam

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.5 
AR_MIN = 0.4           # Oval shape lower limit (Vertical oval)
AR_MAX = 2.5           # Oval shape upper limit (Horizontal oval)

# Tracking Thresholds
LOCK_DISTANCE_THRESHOLD = 150 # Max pixels the lemon can move between frames
ALIGNMENT_TOLERANCE = 20      # Pixels. If error is less than this, we are "Aligned"

# Harvest "Distance" Threshold (Step 5)
# If the lemon's area is bigger than this, we are close enough to cut.
# You must tune this value! (e.g., measure the area when the cutter is touching a lemon)
HARVEST_AREA_LIMIT = 45000    

# ==============================================================================
# 2. ROBOT SETUP (SERVO CONSTRAINTS & STATE)
# ==============================================================================
# Define the physical limits of your 5 DOF arm
ROBOT_CONSTRAINTS = {
    'motor_1': {'min': 0, 'max': 180, 'home': 90, 'axis': 'X (Base)'},
    'motor_2': {'min': 20, 'max': 160, 'home': 45, 'axis': 'Y (Shoulder)'},
    'motor_3': {'min': 20, 'max': 160, 'home': 90, 'axis': 'Z (Forward/Elbow)'} 
}

# Current Motor Angles (Start at Home)
current_angles = {
    'motor_1': ROBOT_CONSTRAINTS['motor_1']['home'],
    'motor_2': ROBOT_CONSTRAINTS['motor_2']['home'],
    'motor_3': ROBOT_CONSTRAINTS['motor_3']['home']
}

# Control Gains (How fast the robot moves per pixel of error)
K_PAN = 0.05    # X-Axis Speed
K_TILT = 0.05   # Y-Axis Speed
K_FORWARD = 0.2 # Forward Speed (Approach)

# State Variables for Locking
is_target_locked = False
locked_target_center = None # (x, y)
locked_target_area = 0

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# ==============================================================================
# 4. MAIN EXECUTION LOOP
# ==============================================================================
# Load Model & Camera
model = YOLO(MODEL_PATH).to('cpu')
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

# Get Dimensions
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
CAMERA_CENTER_X = W // 2
CAMERA_CENTER_Y = H // 2

print("--- HarBot System Active ---")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --------------------------------------------------------------------------
    # STEP 1: DETECT & FILTER (Oval + Shape)
    # --------------------------------------------------------------------------
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    current_candidates = []

    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter for Ovals (Lemons)
            if AR_MIN < aspect_ratio < AR_MAX:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                current_candidates.append({
                    'center': (cx, cy),
                    'box': (x1, y1, x2, y2),
                    'area': area
                })

    # --------------------------------------------------------------------------
    # STEP 2: TARGET LOCKING (Logic to stick to one lemon)
    # --------------------------------------------------------------------------
    final_target = None
    system_state = "SEARCHING"
    box_color = (0, 255, 255) # Yellow

    if is_target_locked and locked_target_center is not None:
        # --- TRACKING MODE ---
        # Find the lemon closest to where our target was last frame.
        # This prevents switching targets even if a bigger one appears.
        best_match = None
        min_dist = float('inf')
        
        for candidate in current_candidates:
            dist = calculate_distance(locked_target_center, candidate['center'])
            if dist < min_dist:
                min_dist = dist
                best_match = candidate
        
        if best_match and min_dist < LOCK_DISTANCE_THRESHOLD:
            # Update the lock
            final_target = best_match
            locked_target_center = best_match['center']
            locked_target_area = best_match['area']
            system_state = "LOCKED"
            box_color = (0, 0, 255) # Red
        else:
            print("⚠️ Target Lost! Re-scanning...")
            is_target_locked = False
            
    else:
        # --- SELECTION MODE ---
        # Find the Largest Lemon (Most Harvestable)
        if current_candidates:
            current_candidates.sort(key=lambda x: x['area'], reverse=True)
            final_target = current_candidates[0]
            
            # Lock it
            is_target_locked = True
            locked_target_center = final_target['center']
            locked_target_area = final_target['area']
            system_state = "ACQUIRED"
            box_color = (0, 255, 0) # Green

    # --------------------------------------------------------------------------
    # STEP 3 & 4: MEASURE ERROR & CALCULATE ANGLES
    # --------------------------------------------------------------------------
    if final_target:
        # Calculate Pixel Error
        error_x = final_target['center'][0] - CAMERA_CENTER_X
        error_y = final_target['center'][1] - CAMERA_CENTER_Y
        
        # Check if Aligned (Within Tolerance)
        is_aligned_x = abs(error_x) < ALIGNMENT_TOLERANCE
        is_aligned_y = abs(error_y) < ALIGNMENT_TOLERANCE

        # --- MOTOR CONTROL LOGIC ---
        
        if not (is_aligned_x and is_aligned_y):
            # PHASE A: ALIGNMENT (Move X and Y Motors)
            system_state = "ALIGNING"
            
            # Convert X Error -> Motor 1 (Base) Angle
            delta_pan = error_x * K_PAN
            # Note: If moving wrong way, change -= to +=
            current_angles['motor_1'] -= delta_pan 
            
            # Convert Y Error -> Motor 2 (Shoulder) Angle
            delta_tilt = error_y * K_TILT
            # Note: If moving wrong way, change += to -=
            current_angles['motor_2'] += delta_tilt 
            
        else:
            # PHASE B: APPROACH (Step 5 - Move Forward)
            # Only move forward if we are ALIGNED.
            
            if locked_target_area < HARVEST_AREA_LIMIT:
                system_state = "APPROACHING"
                box_color = (255, 0, 255) # Purple
                
                # Move Motor 3 (Elbow/Forward) to extend reach
                # This depends on your specific robot kinematics!
                current_angles['motor_3'] += K_FORWARD 
            
            else:
                # STOP! We are close enough.
                system_state = "HARVEST READY"
                box_color = (255, 255, 255) # White
                # HERE YOU WOULD TRIGGER THE CUTTER
                # cutter_activate()

        # --- APPLY SAFETY CLAMPS (Constraints) ---
        current_angles['motor_1'] = clamp(current_angles['motor_1'], ROBOT_CONSTRAINTS['motor_1']['min'], ROBOT_CONSTRAINTS['motor_1']['max'])
        current_angles['motor_2'] = clamp(current_angles['motor_2'], ROBOT_CONSTRAINTS['motor_2']['min'], ROBOT_CONSTRAINTS['motor_2']['max'])
        current_angles['motor_3'] = clamp(current_angles['motor_3'], ROBOT_CONSTRAINTS['motor_3']['min'], ROBOT_CONSTRAINTS['motor_3']['max'])

        # --- SIMULATE SENDING TO ROBOT ---
        # In real life, send these via Serial: e.g., arduino.write(f"{m1},{m2}...")
        m1 = int(current_angles['motor_1'])
        m2 = int(current_angles['motor_2'])
        m3 = int(current_angles['motor_3'])
        
        print(f"[{system_state}] Err:({error_x}, {error_y}) | Area:{locked_target_area} | Motors: M1={m1}, M2={m2}, M3={m3}")

        # Draw UI
        cx, cy = final_target['center']
        cv2.rectangle(frame, (final_target['box'][0], final_target['box'][1]), (final_target['box'][2], final_target['box'][3]), box_color, 2)
        cv2.line(frame, (CAMERA_CENTER_X, CAMERA_CENTER_Y), (cx, cy), box_color, 2)
        cv2.putText(frame, f"{system_state}", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # Draw Crosshair
    cv2.line(frame, (CAMERA_CENTER_X - 20, CAMERA_CENTER_Y), (CAMERA_CENTER_X + 20, CAMERA_CENTER_Y), (0, 0, 255), 2)
    cv2.line(frame, (CAMERA_CENTER_X, CAMERA_CENTER_Y - 20), (CAMERA_CENTER_X, CAMERA_CENTER_Y + 20), (0, 0, 255), 2)

    cv2.imshow("HarBot Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()