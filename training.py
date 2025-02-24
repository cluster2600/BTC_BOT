# training.py
import logging
from _0_dl_trainval_data import main as trainval_main
from _1_optimize_cpcv import optimize as cpcv_optimize
from _1_optimize_kcv import optimize as kcv_optimize
from random_forest import main as rf_main

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

if __name__ == "__main__":
    logging.debug("Starting training pipeline...")
    logging.debug("Running trainval_main...")
    trainval_main()
    logging.debug("Running cpcv_optimize...")
    cpcv_optimize(name_test='model', model_name='ppo', gpu_id='0')
    logging.debug("Running kcv_optimize...")
    kcv_optimize(name_test='model', model_name='ppo', gpu_id='0')
    logging.debug("Running random forest ensemble training...")
    rf_main()
    logging.debug("All training processes completed.")