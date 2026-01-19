class RegistryService:
    def __init__(self):
        self.store = []

    def save(self, agent, metrics, profile):
        self.store.append({
            "agent": agent,
            "metrics": metrics,
            "profile": profile
        })

    def select(self, profile):
        candidates = [m for m in self.store if m["profile"]["name"] == profile["name"]]
        key = profile["eval"]

        if key == "stability":
            return max(candidates, key=lambda x: x["metrics"]["stability"])
        if key == "profit":
            return max(candidates, key=lambda x: x["metrics"]["profit"])
        if key == "balanced":
            return max(
                candidates,
                key=lambda x: x["metrics"]["profit"] - x["metrics"]["volatility"]
            )
