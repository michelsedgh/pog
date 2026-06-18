from collections import OrderedDict

from datasets.object_vocab import (
    GROUPS,
    OBJECTLESS_ACTIONS,
    STRONG_ACTION_OBJECTS,
)


TOYOTA_ACTION_TAXONOMIES = ("toyota_31", "product_v1")

TOYOTA_CS_ACTIONS = (
    "Cook.Cleandishes",
    "Cook.Cleanup",
    "Cook.Cut",
    "Cook.Stir",
    "Cook.Usestove",
    "Cutbread",
    "Drink.Frombottle",
    "Drink.Fromcan",
    "Drink.Fromcup",
    "Drink.Fromglass",
    "Eat.Attable",
    "Eat.Snack",
    "Enter",
    "Getup",
    "Laydown",
    "Leave",
    "Makecoffee.Pourgrains",
    "Makecoffee.Pourwater",
    "Maketea.Boilwater",
    "Maketea.Insertteabag",
    "Pour.Frombottle",
    "Pour.Fromcan",
    "Pour.Fromkettle",
    "Readbook",
    "Sitdown",
    "Takepills",
    "Uselaptop",
    "Usetablet",
    "Usetelephone",
    "Walk",
    "WatchTV",
)

TOYOTA_CV_ACTIONS = (
    "Cutbread",
    "Drink.Frombottle",
    "Drink.Fromcan",
    "Drink.Fromcup",
    "Drink.Fromglass",
    "Eat.Attable",
    "Eat.Snack",
    "Enter",
    "Getup",
    "Leave",
    "Pour.Frombottle",
    "Pour.Fromcan",
    "Readbook",
    "Sitdown",
    "Takepills",
    "Uselaptop",
    "Usetablet",
    "Usetelephone",
    "Walk",
)

CS_DICT = {action_name: index + 1 for index, action_name in enumerate(TOYOTA_CS_ACTIONS)}
CV_DICT = {action_name: index + 1 for index, action_name in enumerate(TOYOTA_CV_ACTIONS)}

PRODUCT_V1_CS_ACTIONS = (
    "Cook.Cleandishes",
    "Cook.Cleanup",
    "Cut",
    "Cook.Stir",
    "Cook.Usestove",
    "Drink",
    "Eat.Attable",
    "Eat.Snack",
    "Enter",
    "Getup",
    "Laydown",
    "Leave",
    "Makecoffee.Pourgrains",
    "Makecoffee.Pourwater",
    "Maketea.Boilwater",
    "Maketea.Insertteabag",
    "Pour.Frombottle",
    "Pour.Fromcan",
    "Pour.Fromkettle",
    "Readbook",
    "Sitdown",
    "Uselaptop",
    "Usetablet",
    "Usetelephone",
    "Walk",
    "WatchTV",
)

PRODUCT_V1_RAW_TO_ACTION = {
    "Cook.Cleandishes": "Cook.Cleandishes",
    "Cook.Cleanup": "Cook.Cleanup",
    "Cook.Cut": "Cut",
    "Cook.Stir": "Cook.Stir",
    "Cook.Usestove": "Cook.Usestove",
    "Cutbread": "Cut",
    "Drink.Frombottle": "Drink",
    "Drink.Fromcan": "Drink",
    "Drink.Fromcup": "Drink",
    "Drink.Fromglass": "Drink",
    "Eat.Attable": "Eat.Attable",
    "Eat.Snack": "Eat.Snack",
    "Enter": "Enter",
    "Getup": "Getup",
    "Laydown": "Laydown",
    "Leave": "Leave",
    "Makecoffee.Pourgrains": "Makecoffee.Pourgrains",
    "Makecoffee.Pourwater": "Makecoffee.Pourwater",
    "Maketea.Boilwater": "Maketea.Boilwater",
    "Maketea.Insertteabag": "Maketea.Insertteabag",
    "Pour.Frombottle": "Pour.Frombottle",
    "Pour.Fromcan": "Pour.Fromcan",
    "Pour.Fromkettle": "Pour.Fromkettle",
    "Readbook": "Readbook",
    "Sitdown": "Sitdown",
    "Uselaptop": "Uselaptop",
    "Usetablet": "Usetablet",
    "Usetelephone": "Usetelephone",
    "Walk": "Walk",
    "WatchTV": "WatchTV",
}


def normalize_toyota_action_taxonomy(action_taxonomy):
    name = str(action_taxonomy or "toyota_31").strip().lower().replace("-", "_")
    aliases = {
        "31": "toyota_31",
        "toyota": "toyota_31",
        "legacy": "toyota_31",
        "product": "product_v1",
        "product_1": "product_v1",
    }
    name = aliases.get(name, name)
    if name not in TOYOTA_ACTION_TAXONOMIES:
        raise ValueError(
            "Unknown Toyota action taxonomy "
            f"{action_taxonomy!r}. Expected one of {TOYOTA_ACTION_TAXONOMIES}."
        )
    return name


def _normalize_task_type(task_type):
    task_type = str(task_type or "CS").upper()
    if task_type not in {"CS", "CV"}:
        raise ValueError(f"Unsupported Toyota task_type: {task_type}")
    return task_type


def toyota_action_names(task_type="CS", action_taxonomy="toyota_31"):
    task_type = _normalize_task_type(task_type)
    action_taxonomy = normalize_toyota_action_taxonomy(action_taxonomy)
    if action_taxonomy == "toyota_31":
        return list(TOYOTA_CS_ACTIONS if task_type == "CS" else TOYOTA_CV_ACTIONS)
    if task_type != "CS":
        raise ValueError("product_v1 is defined for Toyota CS labels only")
    return list(PRODUCT_V1_CS_ACTIONS)


def toyota_action_to_index(task_type="CS", action_taxonomy="toyota_31"):
    return {
        action_name: index
        for index, action_name in enumerate(
            toyota_action_names(task_type, action_taxonomy)
        )
    }


def toyota_num_classes(task_type="CS", action_taxonomy="toyota_31"):
    return len(toyota_action_names(task_type, action_taxonomy))


def toyota_canonical_action(raw_action_name, task_type="CS", action_taxonomy="toyota_31"):
    raw_action_name = str(raw_action_name)
    task_type = _normalize_task_type(task_type)
    action_taxonomy = normalize_toyota_action_taxonomy(action_taxonomy)
    if action_taxonomy == "toyota_31":
        return raw_action_name if raw_action_name in toyota_label_dict(task_type) else None
    return PRODUCT_V1_RAW_TO_ACTION.get(raw_action_name)


def toyota_label_dict(task_type="CS", action_taxonomy="toyota_31"):
    task_type = _normalize_task_type(task_type)
    action_taxonomy = normalize_toyota_action_taxonomy(action_taxonomy)
    if action_taxonomy == "toyota_31":
        return CS_DICT if task_type == "CS" else CV_DICT

    action_to_label = {
        action_name: index + 1
        for index, action_name in enumerate(toyota_action_names(task_type, action_taxonomy))
    }
    return {
        raw_action: action_to_label[canonical_action]
        for raw_action, canonical_action in PRODUCT_V1_RAW_TO_ACTION.items()
    }


def _unique_in_action_order(action_names, task_type, action_taxonomy):
    action_order = toyota_action_to_index(task_type, action_taxonomy)
    values = OrderedDict()
    for action_name in action_names:
        if action_name in action_order:
            values[action_name] = None
    return tuple(sorted(values.keys(), key=lambda name: action_order[name]))


def toyota_group_action_names(task_type="CS", action_taxonomy="toyota_31"):
    task_type = _normalize_task_type(task_type)
    action_taxonomy = normalize_toyota_action_taxonomy(action_taxonomy)
    groups = {}
    for group_name, raw_action_names in GROUPS.items():
        if action_taxonomy == "product_v1" and group_name in {
            "eat_pills",
            "drink_cup_bottle_glass",
        }:
            continue
        canonical_names = [
            toyota_canonical_action(raw_action, task_type, action_taxonomy)
            for raw_action in raw_action_names
        ]
        action_names = _unique_in_action_order(
            (name for name in canonical_names if name is not None),
            task_type,
            action_taxonomy,
        )
        if action_names:
            groups[group_name] = action_names

    if action_taxonomy == "product_v1":
        groups["eat"] = ("Eat.Attable", "Eat.Snack")
        groups["drink"] = ("Drink",)
    return groups


def toyota_action_object_map(task_type="CS", action_taxonomy="toyota_31"):
    task_type = _normalize_task_type(task_type)
    action_taxonomy = normalize_toyota_action_taxonomy(action_taxonomy)
    action_names = set(toyota_action_names(task_type, action_taxonomy))
    output = OrderedDict()
    for raw_action, object_names in STRONG_ACTION_OBJECTS.items():
        action_name = toyota_canonical_action(raw_action, task_type, action_taxonomy)
        if action_name is None or action_name not in action_names:
            continue
        existing = output.setdefault(action_name, OrderedDict())
        for object_name in object_names:
            existing[object_name] = None
    return {action_name: tuple(objects.keys()) for action_name, objects in output.items()}


def toyota_objectless_action_names(task_type="CS", action_taxonomy="toyota_31"):
    task_type = _normalize_task_type(task_type)
    action_taxonomy = normalize_toyota_action_taxonomy(action_taxonomy)
    return _unique_in_action_order(
        (
            toyota_canonical_action(raw_action, task_type, action_taxonomy)
            for raw_action in OBJECTLESS_ACTIONS
        ),
        task_type,
        action_taxonomy,
    )

