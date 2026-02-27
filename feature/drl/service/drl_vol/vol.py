from config import EXECUTION

class VolService:
    def predict(self, state, signal, position, profile=None) -> float:
        if signal == "hold":
            return EXECUTION.HOLD_VOL
        return EXECUTION.DEFAULT_VOL

