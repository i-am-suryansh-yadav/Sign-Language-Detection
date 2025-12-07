# import cv2
# import mediapipe as mp

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)
#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
#     cv2.imshow('Hands', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows() 
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

print("Press SPACE to print 63 landmark values (x, y, z). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for natural interaction
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Check for spacebar to print landmarks
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                coords = []
                for lm in handLms.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                print(f"\nLandmark coordinates ({len(coords)} values):")
                print(coords)

    else:
        key = cv2.waitKey(1) & 0xFF  # even if no hand is detected
        if key == ord(' '):
            print("\nNo hand detected. Try again.")

    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()