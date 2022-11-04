import pandas as pd


class DataPath:
    """Get the data path and read it"""
    def __init__(self, config):
        self.data_path = config["data_path"]

    def get_data(self):
        return pd.read_csv(self.data_path)

    def get_large_data(self):
        for chunk in pd.read_csv(self.data_path):
            yield chunk


class ReadData(DataPath):
    """"Read data and eliminate some columns"""
    def read_data(self) -> pd.DataFrame:
        pass
