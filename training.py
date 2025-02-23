"""
This script launches the training of all models at once.
"""

from _0_dl_trainval_data import main as trainval_main
from _1_optimize_cpcv import optimize as cpcv_optimize
from _1_optimize_kcv import optimize as kcv_optimize

if __name__ == "__main__":
    trainval_main()
    cpcv_optimize(name_test='model', model_name='ppo', gpu_id='0')
<<<<<<< HEAD
    kcv_optimize(name_test='model', model_name='ppo', gpu_id='0')
=======
    kcv_optimize(name_test='model', model_name='ppo', gpu_id='0')
>>>>>>> e579a3a (Updated processor_Binance.py, function_CPCV.py, requirements.txt, and added .gitignore to exclude venv310)
