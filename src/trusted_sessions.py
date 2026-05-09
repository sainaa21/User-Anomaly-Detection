def is_trusted_session(risk_score, threshold=0.3):
    return risk_score < threshold