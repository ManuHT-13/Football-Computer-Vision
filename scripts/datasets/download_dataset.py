import os
from dotenv import load_dotenv
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

load_dotenv()

soccernet_password = os.getenv("SOCCERNET_PASSWORD")
if not soccernet_password:
    raise ValueError("SOCCERNET_PASSWORD not defined in .env")

downloader = SoccerNetDownloader(LocalDirectory="/datasets/soccernet")
downloader.password = soccernet_password
downloader.downloadDataTask(task="tracking-2023", split=["train", "valid", "test"])