import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset.affect.get_data import Affect


class DatasetConfigurationTest(unittest.TestCase):
    def test_mosi_download_creates_missing_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "nested" / "mosi_data.pkl"
            dataset = object.__new__(Affect)
            dataset.data_path = str(data_path)
            dataset.dataset = "mosi"

            with patch("dataset.affect.get_data.gdown.download") as download:
                dataset._download_file()

            self.assertTrue(data_path.parent.is_dir())
            download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
