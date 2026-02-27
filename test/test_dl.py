import sys
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from feature.drl.input.profile_input import ProfileInput
from feature.drl.usecase.sign_uc import SignUC
from feature.middleware.profile import ProfileService
from test_csv import csv_pd


ProfileService().set_profile()
df = SignUC().run()
csv_pd(name="rl_output", df=df)