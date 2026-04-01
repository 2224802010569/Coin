import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from feature.processing.input.data_input import DataInput
from feature.processing.service.processing import ProcessingService
from feature.processing.usecase.processing_uc import ProcessingUC
from test_csv import csv_pd


# df = DataInput().run(tf = "1d")
# df = ProcessingService().run(df = df)
df = ProcessingUC().run(tf = "1d")
csv_pd(name = "processing", df = df)