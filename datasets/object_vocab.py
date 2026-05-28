import re


OBJECT_CLASSES = {
    0: "book",
    1: "laptop",
    2: "phone",
    3: "tv_monitor",
    4: "remote",
    5: "keyboard_mouse",
    6: "cup",
    7: "bottle",
    8: "glass",
    9: "utensil",
    10: "bowl",
    11: "food_snack",
    12: "dining_table",
    13: "chair",
    14: "couch",
    15: "bed",
    16: "sink",
    17: "cooking_appliance",
    18: "refrigerator",
}

OBJECT_TO_ID = {name: idx for idx, name in OBJECT_CLASSES.items()}
NUM_OBJECT_CLASSES = len(OBJECT_CLASSES)
NONE_OBJECT_ID = NUM_OBJECT_CLASSES
OBJECT_SUMMARY_FEATURES = (
    "present",
    "max_conf",
    "mean_conf",
    "frame_frac",
    "log_count",
    "max_area",
    "mean_area",
    "cx",
    "cy",
)
OBJECT_SUMMARY_FEATURE_DIM = len(OBJECT_SUMMARY_FEATURES)

COCO_TO_OBJECT = {
    "book": "book",
    "laptop": "laptop",
    "cell phone": "phone",
    "tv": "tv_monitor",
    "remote": "remote",
    "keyboard": "keyboard_mouse",
    "mouse": "keyboard_mouse",
    "cup": "cup",
    "bottle": "bottle",
    "wine glass": "glass",
    "fork": "utensil",
    "knife": "utensil",
    "spoon": "utensil",
    "bowl": "bowl",
    "banana": "food_snack",
    "apple": "food_snack",
    "sandwich": "food_snack",
    "orange": "food_snack",
    "broccoli": "food_snack",
    "carrot": "food_snack",
    "hot dog": "food_snack",
    "pizza": "food_snack",
    "donut": "food_snack",
    "cake": "food_snack",
    "dining table": "dining_table",
    "chair": "chair",
    "couch": "couch",
    "bed": "bed",
    "sink": "sink",
    "microwave": "cooking_appliance",
    "oven": "cooking_appliance",
    "toaster": "cooking_appliance",
    "refrigerator": "refrigerator",
}

# Compatibility aliases for scripts that consume generic detector names.
DETECTOR_TO_OBJECT = COCO_TO_OBJECT
YOLO_TO_OBJECT = COCO_TO_OBJECT

DEFAULT_OBJECT_CLASS_THRESHOLDS = {
    "book": 0.50,
    "laptop": 0.50,
    "phone": 0.65,
    "tv_monitor": 0.70,
    "remote": 0.60,
    "keyboard_mouse": 0.55,
    "cup": 0.50,
    "bottle": 0.50,
    "glass": 0.50,
    "utensil": 0.50,
    "bowl": 0.50,
    "food_snack": 0.50,
    "dining_table": 0.60,
    "chair": 0.55,
    "couch": 0.55,
    "bed": 0.55,
    "sink": 0.55,
    "cooking_appliance": 0.55,
    "refrigerator": 0.55,
}

DEFAULT_OBJECT_CAMERA_ALLOWLIST = {
    "tv_monitor": {"c05", "c06"},
}

DEFAULT_OBJECT_IGNORE_REGIONS = {
    # Static recording/control hardware in the top-left of Toyota kitchen camera c03.
    "c03": [(0.0, 0.0, 0.26, 0.42)],
}


def normalize_camera_id(camera_id):
    text = str(camera_id).strip().lower()
    if not text:
        return None
    if text.startswith("c"):
        text = text[1:]
    if not text.isdigit():
        return None
    return f"c{int(text):02d}"


def camera_id_from_file_id(file_id):
    match = re.search(r"_c(\d+)$", str(file_id))
    if match is None:
        return None
    return normalize_camera_id(match.group(1))


def parse_object_camera_allowlist(text=None):
    if text is None:
        return {
            name: set(cameras)
            for name, cameras in DEFAULT_OBJECT_CAMERA_ALLOWLIST.items()
        }
    if isinstance(text, dict):
        return {
            str(name): {
                camera
                for camera in (normalize_camera_id(value) for value in cameras)
                if camera is not None
            }
            for name, cameras in text.items()
        }

    text = str(text).strip()
    if text.lower() in {"", "none", "off", "0"}:
        return {}

    allowlist = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid object camera allowlist '{item}'. Expected class=c05,c06."
            )
        object_name, cameras_text = [part.strip() for part in item.split("=", 1)]
        if object_name not in OBJECT_TO_ID:
            raise ValueError(f"Unknown object class in camera allowlist: {object_name}")
        cameras = {
            camera
            for camera in (
                normalize_camera_id(value)
                for value in re.split(r"[,| ]+", cameras_text)
                if value
            )
            if camera is not None
        }
        allowlist[object_name] = cameras
    return allowlist


def object_allowed_for_file_id(object_name, file_id, camera_allowlist):
    cameras = camera_allowlist.get(object_name)
    if not cameras:
        return True
    camera_id = camera_id_from_file_id(file_id)
    return camera_id in cameras


def _parse_normalized_region(text):
    values = [float(value.strip()) for value in str(text).split(",")]
    if len(values) != 4:
        raise ValueError(f"Invalid ignore region '{text}'. Expected x1,y1,x2,y2.")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(
            f"Ignore region must be normalized xyxy within [0, 1], got {text}."
        )
    return (x1, y1, x2, y2)


def parse_object_ignore_regions(text=None):
    if text is None:
        return {
            camera: list(regions)
            for camera, regions in DEFAULT_OBJECT_IGNORE_REGIONS.items()
        }
    if isinstance(text, dict):
        parsed = {}
        for camera, regions in text.items():
            camera_id = normalize_camera_id(camera)
            if camera_id is None:
                raise ValueError(f"Invalid camera id in object ignore regions: {camera}")
            parsed[camera_id] = [tuple(region) for region in regions]
        return parsed

    text = str(text).strip()
    if text.lower() in {"", "none", "off", "0"}:
        return {}

    regions = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid object ignore region '{item}'. Expected c03=x1,y1,x2,y2."
            )
        camera_text, region_text = [part.strip() for part in item.split("=", 1)]
        camera_id = normalize_camera_id(camera_text)
        if camera_id is None:
            raise ValueError(f"Invalid camera id in object ignore region: {camera_text}")
        regions.setdefault(camera_id, []).append(_parse_normalized_region(region_text))
    return regions


def object_box_ignored_for_file_id(box_xyxy, file_id, width, height, ignore_regions):
    camera_id = camera_id_from_file_id(file_id)
    regions = ignore_regions.get(camera_id)
    if not regions:
        return False
    if width is None or height is None or float(width) <= 0 or float(height) <= 0:
        return False
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    cx = ((x1 + x2) * 0.5) / float(width)
    cy = ((y1 + y2) * 0.5) / float(height)
    for rx1, ry1, rx2, ry2 in regions:
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return True
    return False

ACTION_TO_OBJECT_GROUPS = {
    "Cook.Cleandishes": ["sink", "bowl", "utensil", "cup"],
    "Cook.Cleanup": ["sink", "dining_table", "bowl", "utensil", "cup"],
    "Cook.Cut": ["utensil", "food_snack", "bowl"],
    "Cook.Stir": ["utensil", "bowl", "cup"],
    "Cook.Usestove": ["cooking_appliance"],
    "Cutbread": ["utensil", "food_snack"],
    "Drink.Frombottle": ["bottle"],
    "Drink.Fromcan": [],
    "Drink.Fromcup": ["cup"],
    "Drink.Fromglass": ["glass"],
    "Eat.Attable": ["food_snack", "bowl", "utensil", "dining_table"],
    "Eat.Snack": ["food_snack", "bowl"],
    "Enter": [],
    "Getup": [],
    "Laydown": [],
    "Leave": [],
    "Makecoffee.Pourgrains": ["cup"],
    "Makecoffee.Pourwater": ["cup"],
    "Maketea.Boilwater": ["cup"],
    "Maketea.Insertteabag": ["cup"],
    "Pour.Frombottle": ["bottle", "cup"],
    "Pour.Fromcan": ["cup"],
    "Pour.Fromkettle": ["cup"],
    "Readbook": ["book"],
    "Sitdown": ["chair", "couch"],
    "Takepills": ["cup"],
    "Uselaptop": ["laptop", "keyboard_mouse"],
    "Usetablet": [],
    "Usetelephone": ["phone"],
    "Walk": [],
    "WatchTV": ["tv_monitor", "remote", "couch"],
}

ACTION_TO_OBJECT = ACTION_TO_OBJECT_GROUPS

STRONG_ACTION_OBJECTS = {
    ("Uselaptop", "laptop"),
    ("Readbook", "book"),
    ("Usetelephone", "phone"),
    ("Drink.Frombottle", "bottle"),
    ("Drink.Fromcup", "cup"),
    ("Drink.Fromglass", "glass"),
}

OBJECTLESS_ACTIONS = {
    "Enter",
    "Getup",
    "Laydown",
    "Leave",
    "Walk",
}

GROUPS = {
    "laptop_book_tv": ["Uselaptop", "Readbook", "WatchTV"],
    "phone_tablet": ["Usetelephone", "Usetablet"],
    "drink": [
        "Drink.Frombottle",
        "Drink.Fromcan",
        "Drink.Fromcup",
        "Drink.Fromglass",
    ],
    "eat_pills": ["Eat.Attable", "Eat.Snack", "Takepills"],
    "cook_eat_kitchen": [
        "Cook.Cleandishes",
        "Cook.Cleanup",
        "Cook.Cut",
        "Cook.Stir",
        "Cook.Usestove",
        "Eat.Attable",
        "Eat.Snack",
        "Takepills",
    ],
}
