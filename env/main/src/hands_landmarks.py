import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press SPACE to print 63 landmark values (x, y, z). Press Q to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow('Hand Landmarks', frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break
    elif key == 32:  # SPACE pressed
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            print(f"\nLandmark coordinates ({len(coords)} values):")
            print(coords)
        else:
            print("No hand detected. Try again.")

cap.release()
cv2.destroyAllWindows()