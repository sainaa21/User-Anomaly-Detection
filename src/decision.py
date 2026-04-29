# src/decision.py

LOW_THRESHOLD = 0.2
MEDIUM_THRESHOLD = 0.5
HIGH_THRESHOLD = 0.7


def decide_action(risk: float) -> str:
    if risk < LOW_THRESHOLD:
        return "ALLOW"
    elif risk < MEDIUM_THRESHOLD:
        return "MONITOR"
    elif risk < HIGH_THRESHOLD:
        return "WARN"
    else:
        return "REAUTHENTICATE"


def generate_explanation(row, risk):
    reasons = []

    if row.get("typing_z", 0) > 2:
        reasons.append("typing deviation")

    if row.get("click_z", 0) > 2:
        reasons.append("click anomaly")

    if row.get("activity_ratio", 1) < 0.3:
        reasons.append("low activity")

    if not reasons and risk > 0.5:
        return "behavior slightly deviates from user pattern"

    return " + ".join(reasons) if reasons else "behavior normal"


def execute_action(action: str):
    if action == "ALLOW":
        pass
    elif action == "MONITOR":
        print("Monitoring...")
    elif action == "WARN":
        print("Warning user...")
    elif action == "REAUTHENTICATE":
        print("Re-authentication triggered")


def log_decision(risk, action):
    print(f"[DECISION] Risk: {risk:.3f} → {action}")