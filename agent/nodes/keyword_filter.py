"""
Compliance keyword filter — extracted from KARR-AI's worker.agent.nodes.routing.

Single function `has_compliance_keywords(message)` returning True if the
message looks compliance-flavored. Used by the Code Charlie classifier as a
fast deterministic check before falling back to the LLM.
"""

COMPLIANCE_KEYWORDS = [
    # Document & Standard References
    "dbc", "dubai building code", "part d",
    "uae flsc", "flsc", "civil defence",
    "asme", "bs en", "cibse", "bco",
    "section", "table", "figure",

    # Compliance/Requirement Terms
    "requirement", "requirements", "compliance", "compliant",
    "regulation", "regulations", "regulatory",
    "code", "codes", "building code",
    "standard", "standards", "specification", "specifications",
    "minimum", "maximum", "shall", "must", "required",
    "permitted", "permissible", "allowed", "acceptable",
    "not exceed", "at least", "no more than", "not less than",

    # Equipment Types
    "elevator", "elevators", "lift", "lifts",
    "escalator", "escalators",
    "moving walk", "moving walks", "travelator",
    "passenger elevator", "service elevator", "goods lift",
    "firefighting elevator", "firefighter elevator", "fire elevator",
    "bed elevator", "stretcher elevator", "panoramic elevator",
    "evacuation elevator",
    "swing mode", "multipurpose",

    # Building Types & Classifications
    "residential", "apartment", "apartments",
    "staff accommodation", "labour accommodation", "labor accommodation",
    "student accommodation", "accommodation",
    "studio", "bedroom", "bedrooms",
    "hotel", "hotels", "hotel apartment", "hotel apartments",
    "1-star", "2-star", "3-star", "4-star", "5-star",
    "guest room", "guest rooms", "keys",
    "office", "offices", "open plan",
    "retail", "shopping centre", "shopping center", "mall", "malls",
    "car park", "car parking", "parking", "basement parking", "podium parking",
    "educational", "school", "schools", "university", "universities", "classroom",
    "healthcare", "clinic", "clinics", "hospital", "hospitals",
    "assembly", "arena", "arenas", "amusement park", "ballroom", "meeting room",

    # Building Height Categories
    "high-rise", "high rise", "super high-rise", "super high rise",
    "high depth", "low-rise", "low rise",
    "23m", "90m", "35m",

    # Elevator Components & Spatial Terms
    "car", "cabin", "car size", "cabin size",
    "rated capacity", "rated load", "rated speed",
    "door", "doors", "door opening", "door closing", "door dwell",
    "two-panel", "four-panel", "centre opening", "center opening", "side opening",
    "door width", "door height", "door size", "door time", "door timing",
    "shaft", "hoistway", "hoist-way", "pit", "headroom", "overhead",
    "machine room", "controller",
    "rcc", "reinforced concrete", "fire rated", "fire-rated",
    "landing", "landing area", "landing call",
    "lobby", "lobbies", "elevator lobby", "entrance lobby",
    "waiting area", "queue",

    # Dimensions & Measurements
    "width", "depth", "height", "clearance",
    "dimension", "dimensions", "size",
    "distance", "travel distance", "horizontal distance",
    "2.4m", "4.5m", "1.8m", "60m", "150m", "2200", "2400",

    # Units of Measurement
    "mm", "millimeter", "millimeters", "millimetre", "millimetres",
    "cm", "centimeter", "centimeters", "centimetre", "centimetres",
    "m", "meter", "meters", "metre", "metres",
    "m²", "m2", "sqm", "square meter", "square meters", "square metre",
    "kg", "kilogram", "kilograms",
    "m/s", "meters per second", "metres per second",
    "m/s²", "m/s2",
    "second", "seconds", "sec", "min", "minute", "minutes",
    "person", "persons", "people", "pax",
    "%", "percent", "percentage",
    "watt", "watts", "lumen", "lumens",
    "hour", "hours", "hr", "1h", "2h",

    # Performance & Traffic Terms
    "speed", "velocity", "acceleration", "jerk",
    "travel time", "flight time",
    "capacity", "handling capacity", "hc5", "hc5%",
    "load", "rated load", "occupant load",
    "capacity factor", "filling rate",
    "waiting time", "average waiting time",
    "destination time", "time to destination",
    "interval", "dwell time",
    "population", "occupancy", "occupancy rate",
    "occupant", "occupants", "persons",

    # Design & Grouping Terms
    "grouping", "group", "elevator group",
    "zoning", "zone", "floor zone",
    "core", "elevator core",
    "boarding floor", "boarding floors",
    "occupied floor", "occupiable floor", "typical floor",
    "magnet floor",
    "ground floor", "basement", "basements", "podium",
    "design method", "method 1", "method 2",
    "prescriptive", "performance-based", "performance based",
    "traffic analysis", "traffic pattern", "traffic study",
    "vt consultant", "vt design",

    # Control Systems
    "control system", "conventional control",
    "destination control", "dcs",
    "destination dispatch", "dd",
    "hall call", "hcdc",
    "hybrid system", "up-peak",
    "vvvf", "regenerative",

    # Safety & Fire Terms
    "fire", "firefighting", "firefighter", "fire service",
    "safety", "safe",
    "emergency", "evacuation",
    "smoke", "fire resistance",
    "dcd", "dubai civil defence",

    # Accessibility & People
    "accessible", "accessibility",
    "people of determination", "disability", "disabilities",
    "wheelchair", "impaired mobility",
    "passenger", "passengers",
    "visitor", "visitors", "guest", "guests",

    # Energy & Efficiency
    "energy", "energy conservation", "energy efficient",
    "standby", "lighting", "lumens",
    "reduced speed", "on demand",
    "photocell", "detector",

    # Escalator/Moving Walk Specific
    "step", "steps", "flat steps",
    "step width", "pallet width",
    "angle", "inclination", "incline",
    "handrail", "balustrade",

    # Question Patterns
    "how many", "how much", "what is the",
    "is it allowed", "can i", "do i need",
    "number of", "quantity",
]


def has_compliance_keywords(message: str) -> bool:
    """True if `message` contains any compliance-related keyword."""
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in COMPLIANCE_KEYWORDS)
