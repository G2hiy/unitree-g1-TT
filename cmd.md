python scripts/rsl_rl/train.py --task=Bydmimic-Flat-G1-v0 --registry_name 874374490-xiamen-university/csv_to_npz/fore_hand --logger wandb --log_project_name HITTER --run_name fore_hand_mimic_3_4_test_final  --max_iterations 2000 --headless

python scripts/rsl_rl/train.py --task=Bydmimic-Flat-G1-v0 --registry_name 874374490-xiamen-university/csv_to_npz/dance1_subject1 --logger wandb --log_project_name Bydmimic --run_name dance  --max_iterations 3000 --headless

python scripts/rsl_rl/train_tt.py --task G1-TableTennis-v0 --predictor --num_envs 4096 --max_iterations 10000 --logger wandb --log_project_name TT --run_name TT_3_10_final --headless


python scripts/rsl_rl/train_tt.py --task G1-TableTennis-v0 --predictor --num_envs 4096 --max_iterations 10000 --logger wandb --log_project_name TT --run_name TT_3_12 --headless


python scripts/rsl_rl/train_tt.py --task G1-TableTennis-v0 --predictor --num_envs 4096 --max_iterations 3000  --headless --resume True --load_run 2026-03-17_21-18-58_TT_3_17 --logger wandb --log_project_name TT --run_name TT_3_18

python scripts/rsl_rl/train_tt.py --task G1-Tracking-v0 --predictor --num_envs 4096 --max_iterations 10000  --headless --logger wandb --log_project_name TT --run_name TT_4_3_forehand_task --resume True --load_run 2026-03-30_19-51-28_TT_3_30_forehand

python scripts/rsl_rl/train.py --task=Bydmimic-Flat-G1-v0 --registry_name 874374490-xiamen-university/csv_to_npz/xmu --max_iterations 5000 --headless --logger wandb --log_project_name Bydmimic --run_name xmu


