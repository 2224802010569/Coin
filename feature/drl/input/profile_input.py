import pandas as pd
from feature.data.entities.profile import PROFILE
from feature.middleware.profile import ProfileService


class ProfileInput:
    def run(self) -> PROFILE:
        return ProfileService().run()