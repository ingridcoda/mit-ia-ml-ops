from pathlib import Path

import joblib


def load_credit_model():
    # Sobe 3 níveis: utils -> production_app -> root -> outputs
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "outputs" / "modeling" / "credit_model_v1.joblib"

    if not model_path.exists():
        return None
    return joblib.load(model_path)
