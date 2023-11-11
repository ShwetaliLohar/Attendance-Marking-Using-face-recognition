import cv2
import face_recognition
import os
import csv
import datetime

known_faces_path = 'Training_images'
output_dir = 'output_csv'

os.makedirs(output_dir, exist_ok=True)

known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = face_recognition.load_image_file(os.path.join(known_faces_path, filename))
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No face found in {filename}. Skipping.")

recognized_faces_today = {}

video_capture = cv2.VideoCapture('Classroom_Clip.mp4')

while True:
    ret, frame = video_capture.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    current_date = datetime.date.today().strftime('%Y-%m-%d')

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

            if name not in recognized_faces_today:
                recognized_faces_today[name] = current_date

                csv_filename = os.path.join(output_dir, f'{current_date}.csv')
                with open(csv_filename, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([current_date, name])

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

