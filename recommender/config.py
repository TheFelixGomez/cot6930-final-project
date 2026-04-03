import os

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data", "ml-1m"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
EVAL_K = int(os.getenv("EVAL_K", "20"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.2"))
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "1000"))
KNN_NEIGHBORS = int(os.getenv("KNN_NEIGHBORS", "21"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))