
# ==========================================
# SMART EMPLOYEE MONITORING SYSTEM
# MAIN PROGRAM
# ==========================================

import cv2
import csv
import os
import time

from datetime import datetime
from ultralytics import YOLO

from config import (
    CAMERA_INDEX,
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    PERSON_CLASS_ID,
    WINDOW_NAME,
    LOG_FILE
)


def create_log():

    if not os.path.exists(LOG_FILE):

        with open(LOG_FILE, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "Person ID",
                "First Seen",
                "Last Seen",
                "Observed Seconds"
            ])


def save_log(person_id, person):

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            person_id,
            person["first_seen"],
            person["last_seen"],
            round(person["observed_time"], 2)
        ])


def format_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def main():

    print("Loading AI model...")

    model = YOLO(MODEL_NAME)

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():

        print("ERROR: Camera could not be opened.")
        return

    create_log()

    people = {}

    previous_frame_time = time.time()

    print("Smart Employee Monitoring System Started")
    print("Press Q to exit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame could not be read.")
            break

        current_time = time.time()

        frame_delta = max(
            0,
            min(current_time - previous_frame_time, 1.0)
        )

        previous_frame_time = current_time

        results = model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        if results and results[0].boxes.id is not None:

            boxes = results[0].boxes

            coordinates = boxes.xyxy.cpu().numpy()

            track_ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(coordinates, track_ids):

                x1, y1, x2, y2 = map(int, box)

                if track_id not in people:

                    people[track_id] = {

                        "first_seen": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                        "last_seen": "",

                        "observed_time": 0.0
                    }

                person = people[track_id]

                person["observed_time"] += frame_delta

                person["last_seen"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Display information

                info = [
                    f"Person ID: {track_id}",
                    "Name: Not enrolled",
                    "Job: Not assigned",
                    "Location: Unknown",
                    f"Observed: {format_time(person['observed_time'])}",
                    "Status: Visible"
                ]

                # Draw person rectangle

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw information panel

                panel_top = max(0, y1 - 135)

                cv2.rectangle(
                    frame,
                    (x1, panel_top),
                    (x2, y1),
                    (35, 35, 35),
                    -1
                )

                for index, text in enumerate(info):

                    cv2.putText(
                        frame,
                        text,
                        (x1 + 5, panel_top + 20 + index * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )

        # Main title

        cv2.putText(
            frame,
            "SMART EMPLOYEE MONITORING",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # Save observed time records

    for person_id, person in people.items():

        save_log(person_id, person)

    camera.release()

    cv2.destroyAllWindows()

    print("System stopped successfully.")
    print(f"Records saved in {LOG_FILE}")


if __name__ == "__main__":

    main()
