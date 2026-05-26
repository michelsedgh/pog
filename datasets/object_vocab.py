OBJECT_CLASSES = {
    0: "book",
    1: "laptop",
    2: "phone",
    3: "tablet",
    4: "cup_mug",
    5: "bottle",
    6: "can",
    7: "glass",
    8: "pill_bottle_medicine",
    9: "kettle",
    10: "tv_monitor",
    11: "remote",
    12: "stove_cooktop",
    13: "sink_dishes",
    14: "plate_bowl_food",
}

OBJECT_TO_ID = {name: idx for idx, name in OBJECT_CLASSES.items()}
NUM_OBJECT_CLASSES = len(OBJECT_CLASSES)
NONE_OBJECT_ID = NUM_OBJECT_CLASSES

YOLO_TO_OBJECT = {
    "book": "book",
    "laptop": "laptop",
    "cell phone": "phone",
    "mobile phone": "phone",
    "tablet": "tablet",
    "cup": "cup_mug",
    "wine glass": "glass",
    "bottle": "bottle",
    "can": "can",
    "bowl": "plate_bowl_food",
    "plate": "plate_bowl_food",
    "tv": "tv_monitor",
    "monitor": "tv_monitor",
    "remote": "remote",
    "keyboard": "laptop",
}

ACTION_TO_OBJECT = {
    "Cook.Cleandishes": ["sink_dishes"],
    "Cook.Cleanup": ["sink_dishes"],
    "Cook.Cut": ["stove_cooktop", "plate_bowl_food"],
    "Cook.Stir": ["stove_cooktop", "cup_mug", "plate_bowl_food"],
    "Cook.Usestove": ["stove_cooktop"],
    "Cutbread": ["plate_bowl_food"],
    "Drink.Frombottle": ["bottle"],
    "Drink.Fromcan": ["can"],
    "Drink.Fromcup": ["cup_mug"],
    "Drink.Fromglass": ["glass"],
    "Eat.Attable": ["plate_bowl_food"],
    "Eat.Snack": ["plate_bowl_food"],
    "Enter": [],
    "Getup": [],
    "Laydown": [],
    "Leave": [],
    "Makecoffee.Pourgrains": ["cup_mug"],
    "Makecoffee.Pourwater": ["cup_mug"],
    "Maketea.Boilwater": ["kettle"],
    "Maketea.Insertteabag": ["kettle", "cup_mug"],
    "Pour.Frombottle": ["bottle"],
    "Pour.Fromcan": ["can"],
    "Pour.Fromkettle": ["kettle"],
    "Readbook": ["book"],
    "Sitdown": [],
    # Option B for now: do not depend on generic detectors finding medicine.
    "Takepills": ["cup_mug"],
    "Uselaptop": ["laptop"],
    "Usetablet": ["tablet"],
    "Usetelephone": ["phone"],
    "Walk": [],
    "WatchTV": ["tv_monitor"],
}

OBJECTLESS_ACTIONS = {
    action_name
    for action_name, object_names in ACTION_TO_OBJECT.items()
    if len(object_names) == 0
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
}
