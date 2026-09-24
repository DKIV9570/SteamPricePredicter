from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
ITAD_RAW = RAW / "itad"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"  # trained model artifacts; versioned in git, unlike data/

load_dotenv(ROOT / ".env")
ITAD_API_KEY = (os.getenv("ITAD_API_KEY") or "").strip()
# Per-account quota shown on https://isthereanydeal.com/apps/my/ (requests per 5 minutes)
ITAD_RATE_PER_5MIN = int(os.getenv("ITAD_RATE_PER_5MIN") or 100)

USER_AGENT = "PricePredicter/0.1 (+https://github.com/)"
STEAM_SHOP_ID = 61
