from feature.data.output.read import ProfileOutput
from feature.data.entities.profile import PROFILE

class ProfileService:
    _ACTIVE_PROFILE: PROFILE | None = None

    @staticmethod
    def set_profile(profile_name: str = "balanced"):
        df = ProfileOutput().run(name=profile_name)
        if df is None or df.empty:
            raise ValueError("Profile not found")
        row = df.iloc[0]
        ProfileService._ACTIVE_PROFILE = PROFILE(
            name=row["name"],
            Stability=row["Stability"],
            Volatility=row["Volatility"],
            Aggression=row["Aggression"],
            Confidence=row["Confidence"],
            Horizon=row["Horizon"],
            Min_Trend_length=row["Min_Trend_length"],
            eval=row["eval"],
        )

    @staticmethod
    def run() -> PROFILE:
        if ProfileService._ACTIVE_PROFILE is None:
            raise RuntimeError("Profile not initialized")
        return ProfileService._ACTIVE_PROFILE
