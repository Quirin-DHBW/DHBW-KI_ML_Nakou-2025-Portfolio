# pipinstall opencv-python first for windows cv driver shenanigans
import cv2

# Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # (0 = default camera)

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect all faces in the frame
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Find largest detected face
    largest_face = None
    max_area = 0
    for (x, y, w, h) in faces:  # x,y = start coordinates; w,h = rectangle
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)

    # If face biggest, make even biggerer
    if largest_face:
        x, y, w, h = largest_face
        margin = int(0.2 * w)  # Add margin for good measure - should be enough for emotion detection like this
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(grey.shape[1], x + w + margin), min(grey.shape[0], y + h + margin)
        
        # Standalone face and pixel-inator
        zoomed_face = grey[y1:y2, x1:x2]
        zoomed_face = cv2.resize(zoomed_face, (48, 48))  # Resize for consistency

        # Save the frame
        cv2.imwrite("zoomed_face.jpg", zoomed_face)

    # Show the original frame with rectangles around faces - for testing
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow("Webcam Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
