import os 

EVENT_SCREENSHOT_FOLDER_USER = os.path.join(
                    os.path.expanduser("~"),
                    "Library", "Application Support", "Sundial", "EventScreenshots", '{user_id}')
EVENT_SCREENSHOT_FOLDER = os.path.join(
                    os.path.expanduser("~"),
                    "Library", "Application Support", "Sundial", "EventScreenshots")
