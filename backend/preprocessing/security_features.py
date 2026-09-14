import re


PATTERNS = {
    "instruction_override": [
        r"\bignore (all )?(previous|prior|above) instructions\b",
        r"\bdisregard (all )?(previous|prior|above) instructions\b",
        r"\bforget (all )?(previous|prior|above) instructions\b",
        r"\boverride (your|the) instructions\b",
        r"\bdo not follow (your|the) instructions\b",
    ],

    "system_prompt_reference": [
        r"\bsystem prompt\b",
        r"\bsystem instructions\b",
        r"\bhidden instructions\b",
        r"\bhidden prompt\b",
        r"\breveal (your|the) prompt\b",
        r"\bshow (your|the) prompt\b",
    ],

    "role_change_attempt": [
        r"\byou are now\b",
        r"\bact as\b",
        r"\bpretend to be\b",
        r"\broleplay as\b",
        r"\bdeveloper mode\b",
        r"\bunrestricted mode\b",
    ],

    "data_extraction_attempt": [
        r"\breveal\b",
        r"\bexpose\b",
        r"\bextract\b",
        r"\bprint\b",
        r"\bshow me\b",
        r"\bgive me\b",
    ],

    "tool_manipulation": [
        r"\bcall (the )?(tool|function|api)\b",
        r"\bexecute (this )?(tool|command|function)\b",
        r"\bsend (an )?email\b",
        r"\bdelete (the )?(file|database|record)\b",
        r"\btransfer money\b",
    ],

    "policy_bypass": [
        r"\bbypass\b",
        r"\bcircumvent\b",
        r"\bevade\b",
        r"\bdisable (the )?(filter|safety|security)\b",
        r"\bwithout restrictions\b",
    ],

    "prompt_leakage": [
        r"\breveal your prompt\b",
        r"\bshow your instructions\b",
        r"\bprint the system message\b",
        r"\bwhat are your hidden instructions\b",
    ],
}

SENSITIVE_PATTERNS = [
    r"\bapi[_ -]?key\b",
    r"\bpassword\b",
    r"\bsecret\b",
    r"\bauthentication token\b",
    r"\baccess token\b",
    r"\bprivate key\b",
    r"\bcredentials?\b",
]


def count_hits(text: str, patterns: list[str]) -> int:
    return sum(
        len(re.findall(pattern, text, re.IGNORECASE))
        for pattern in patterns
    )


def extract_security_features(text: str) -> dict:
    features = {}

    for name, patterns in PATTERNS.items():
        hits = count_hits(text, patterns)

        features[name] = int(hits > 0)
        features[f"{name}_hits"] = hits

    features["sensitive_data_reference"] = int(
        count_hits(text, SENSITIVE_PATTERNS) > 0
    )

    features["imperative_signal"] = int(
        bool(
            re.search(
                r"\b(ignore|disregard|reveal|show|print|execute|"
                r"follow|forget|override|disable|bypass)\b",
                text,
                re.IGNORECASE
            )
        )
    )

    instruction_hits = (
        features["instruction_override_hits"]
        + features["role_change_attempt_hits"]
        + features["tool_manipulation_hits"]
        + features["policy_bypass_hits"]
    )

    word_count = max(len(text.split()), 1)

    features["instruction_density"] = round(
        instruction_hits / word_count,
        4
    )

    return features
