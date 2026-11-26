
# ==========================================
# SMART EMPLOYEE MONITORING SYSTEM
# CONFIGURATION FILE
# ==========================================

CAMERA_INDEX = 0

MODEL_NAME = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.50

PERSON_CLASS_ID = 0

WINDOW_NAME = "Smart Employee Monitoring System"

LOG_FILE = "attendance_log.csv"

# Demo employee records
# These are sample labels, not verified identities.

EMPLOYEES = {
    1: {
        "name": "Employee 01",
        "job": "Accountant",
        "location": "Office 01"
    },
    2: {
        "name": "Employee 02",
        "job": "Manager",
        "location": "Office 02"
    },
    3: {
        "name": "Employee 03",
        "job": "Developer",
        "location": "Office 03"
    }
}
