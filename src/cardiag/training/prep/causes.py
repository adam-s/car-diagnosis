"""Canonicalize the free-text fused_cause vocabulary (~359 raw values) into
~22 part-family groups that are (a) acoustically meaningful and (b) have
enough support to train/eval on.

Precision of the tail matters less than consistency of the head: the top ~60
raw values cover >90% of fault clips. Unmatched tail values fall to 'other'.
"""

import re

# Explicit raw -> group map for everything with meaningful support.
# Keys are lowercase, stripped. Lookup also tries a naive de-pluralized form.
CANONICAL = {
    # wheel bearing family
    "wheel bearing": "wheel_bearing",
    "axle bearing": "wheel_bearing",
    "rear axle bearing": "wheel_bearing",
    "wheel hub": "wheel_bearing",
    # brakes
    "brake pad": "brakes",
    "brake pad wear indicator": "brakes",
    "brake wear indicator": "brakes",
    "brake rotor": "brakes",
    "brake disc": "brakes",
    "brake caliper": "brakes",
    "brake drum": "brakes",
    "brake shoe": "brakes",
    "brake dust shield": "brakes",
    "brake booster": "brakes",
    "brake wheel cylinder": "brakes",
    "wheel cylinder": "brakes",
    "caliper pin": "brakes",
    "handbrake cable": "brakes",
    # accessory belt drive
    "serpentine belt": "belt",
    "serpentine belt tensioner": "belt",
    "belt tensioner": "belt",
    "idler pulley": "belt",
    "idler pulley bearing": "belt",
    "alternator pulley": "belt",
    "pulley bearing": "belt",
    "harmonic balancer": "belt",
    # valvetrain (top-end tick)
    "valve train": "valvetrain",
    "lifter": "valvetrain",
    "valve lifter": "valvetrain",
    "hydraulic lifter": "valvetrain",
    "engine lifter": "valvetrain",
    "rocker arm": "valvetrain",
    "tappet": "valvetrain",
    "valve tappet": "valvetrain",
    "camshaft position sensor": "valvetrain",
    # bottom-end knock
    "engine knock": "rod_knock",
    "rod knock": "rod_knock",
    "connecting rod bearing": "rod_knock",
    "rod bearing": "rod_knock",
    "engine bearing": "rod_knock",
    "connecting rod": "rod_knock",
    "connecting rod bushing": "rod_knock",
    "main shaft bearing": "rod_knock",
    "piston": "rod_knock",
    "piston ring": "rod_knock",
    "piston slap": "rod_knock",
    "piston rod": "rod_knock",
    # CV / driveline joints
    "cv joint": "cv_axle",
    "cv axle": "cv_axle",
    "u-joint": "cv_axle",
    "universal joint": "cv_axle",
    "driveshaft": "cv_axle",
    "carrier bearing": "cv_axle",
    "center support bearing": "cv_axle",
    "driveshaft center support bearing": "cv_axle",
    # power steering
    "power steering pump": "power_steering",
    "power steering hose": "power_steering",
    "power steering suction hose": "power_steering",
    "rack and pinion": "power_steering",
    "steering rack": "power_steering",
    # cooling
    "water pump": "water_pump",
    "water pump bearing": "water_pump",
    "coolant hose": "cooling_other",
    "radiator": "cooling_other",
    "thermostat": "cooling_other",
    "heater core": "cooling_other",
    "coolant temperature sensor": "cooling_other",
    "temperature gauge sender": "cooling_other",
    "oil cooler": "cooling_other",
    # forced induction
    "turbocharger": "turbo",
    "turbo": "turbo",
    # suspension & steering linkage
    "ball joint": "suspension",
    "lower ball joint": "suspension",
    "shock absorber": "suspension",
    "rear shock absorber": "suspension",
    "shock absorber mount": "suspension",
    "strut": "suspension",
    "strut mount": "suspension",
    "sway bar link": "suspension",
    "sway bar end link": "suspension",
    "sway bar bushing": "suspension",
    "control arm": "suspension",
    "lower control arm": "suspension",
    "control arm bushing": "suspension",
    "rear control arm bushing": "suspension",
    "coil spring": "suspension",
    "suspension bushing": "suspension",
    "suspension bush": "suspension",
    "suspension": "suspension",
    "suspension component": "suspension",
    "rubber bushing": "suspension",
    "bushing": "suspension",
    "link bar bushing": "suspension",
    "spindle bush": "suspension",
    "tie rod": "suspension",
    "tie rod end": "suspension",
    # mounts
    "engine mount": "mounts",
    "motor mount": "mounts",
    "transmission mount": "mounts",
    # exhaust
    "exhaust leak": "exhaust",
    "exhaust manifold": "exhaust",
    "exhaust manifold gasket": "exhaust",
    "exhaust gasket": "exhaust",
    "exhaust pipe": "exhaust",
    "exhaust heat shield": "exhaust",
    "heat shield": "exhaust",
    "muffler": "exhaust",
    "catalytic converter": "exhaust",
    # vacuum / air
    "vacuum leak": "vacuum_leak",
    "vacuum line": "vacuum_leak",
    "pcv valve": "vacuum_leak",
    "purge valve": "vacuum_leak",
    "intake manifold gasket": "vacuum_leak",
    "map sensor": "vacuum_leak",
    "idle air control valve": "vacuum_leak",
    "air filter": "vacuum_leak",
    # belt-driven / electrical accessories
    "ac compressor": "accessories",
    "ac compressor clutch": "accessories",
    "ac compressor pulley bearing": "accessories",
    "alternator": "accessories",
    "alternator bearing": "accessories",
    "starter motor": "accessories",
    "battery": "accessories",
    "oil pump": "accessories",
    "carburetor": "accessories",
    # differential / final drive
    "differential": "differential",
    "rear differential": "differential",
    "differential bearing": "differential",
    "differential pinion bearing": "differential",
    "differential carrier": "differential",
    "differential spider gears": "differential",
    "pinion bearing": "differential",
    "ring and pinion gear": "differential",
    "power transfer unit": "differential",
    "transaxle": "differential",
    # transmission / clutch
    "transmission": "transmission",
    "transmission pump": "transmission",
    "transmission front pump": "transmission",
    "transmission oil seal": "transmission",
    "torque converter": "transmission",
    "clutch": "transmission",
    "clutch plate": "transmission",
    "clutch release bearing": "transmission",
    "release bearing": "transmission",
    "synchronizer ring": "transmission",
    "cvt slide piece": "transmission",
    "cvt belt": "transmission",
    "cvt bearing": "transmission",
    # timing drive
    "timing chain": "timing",
    "timing belt": "timing",
    # wheels & tires
    "tire": "wheel_tire",
    "tyre": "wheel_tire",
    "tire cupping": "wheel_tire",
    "tire puncture": "wheel_tire",
    "wheel stud": "wheel_tire",
    "wheel bolt": "wheel_tire",
    # fuel & ignition
    "fuel pump": "fuel_ignition",
    "fuel injector": "fuel_ignition",
    "spark plug": "fuel_ignition",
    "ignition coil": "fuel_ignition",
    "crankshaft position sensor": "fuel_ignition",
    # oil sealing / leaks (sound: usually tick/knock context)
    "valve cover gasket": "seals_gaskets",
    "oil seal": "seals_gaskets",
    "engine oil seal": "seals_gaskets",
    "oil filter": "seals_gaskets",
    "oil filter housing": "seals_gaskets",
    "oil pan": "seals_gaskets",
    "oxygen sensor": "other",
    # body / interior (not powertrain faults)
    "door latch": "body",
    "door hinge": "body",
    "door wiring harness": "body",
    "car speaker": "body",
}

# Ordered keyword fallbacks for the long tail not in CANONICAL.
RULES = [
    (r"\bbrake", "brakes"),
    (r"wheel bearing|hub bearing", "wheel_bearing"),
    (r"\bcv\b|axle", "cv_axle"),
    (r"power steering|steering rack", "power_steering"),
    (r"serpentine|drive belt|belt tensioner|pulley", "belt"),
    (r"lifter|tappet|valve train|rocker", "valvetrain"),
    (r"rod (knock|bearing)|piston|crank", "rod_knock"),
    (r"shock|strut|sway bar|control arm|ball joint|bushing|spring|tie rod",
     "suspension"),
    (r"mount\b", "mounts"),
    (r"exhaust|muffler|catalytic|heat shield", "exhaust"),
    (r"vacuum|pcv|intake leak", "vacuum_leak"),
    (r"differential|pinion|transfer case", "differential"),
    (r"transmission|clutch|torque converter|cvt|gearbox", "transmission"),
    (r"timing (chain|belt)", "timing"),
    (r"\btires?\b|\btyres?\b|wheel (stud|bolt|nut)", "wheel_tire"),
    (r"fuel|injector|spark|ignition", "fuel_ignition"),
    (r"alternator|compressor|starter|battery", "accessories"),
    (r"turbo", "turbo"),
    (r"water pump", "water_pump"),
    (r"coolant|radiator|thermostat", "cooling_other"),
    (r"gasket|seal\b", "seals_gaskets"),
    (r"door|window|trim|speaker|seat", "body"),
    (r"bearing", "bearing_other"),
]


def canonical_cause(raw):
    """Map a raw fused_cause string to a canonical group (or 'other')."""
    if not raw:
        return None
    s = raw.strip().lower()
    if s in CANONICAL:
        return CANONICAL[s]
    # naive de-pluralization: "brake pads" -> "brake pad"
    if s.endswith("s") and s[:-1] in CANONICAL:
        return CANONICAL[s[:-1]]
    for pat, group in RULES:
        if re.search(pat, s):
            return group
    return "other"


# CLAP l1 sound-type strings -> short class keys.
L1_MAP = {
    "normal smooth engine idle": "normal_idle",
    "squealing or squeaking noise": "squeal",
    "ticking or clicking noise": "tick",
    "humming or droning roar": "hum",
    "grinding noise": "grind",
    "hissing noise": "hiss",
    "knocking or clunking noise": "knock",
    "high-pitched whining noise": "whine",
    "rattling noise": "rattle",
    "shop_tool": "shop_tool",
}


def canonical_l1(raw):
    if not raw:
        return None
    return L1_MAP.get(raw, "other")
