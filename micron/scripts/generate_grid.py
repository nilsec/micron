from micron.prepare_grid import prepare_grid

grid_solve_parameters={"evidence_factor": [12, 14, 16],
                       "comb_angle_factor": [12, 14, 16],
                       "start_edge_prior": [180, 200, 220],
                       "selection_cost": [-70, -80, -90]}

prepare_grid(base_dir="",
             experiment="",
             train_number=0,
             predict_number=0,
             graph_number=0,
             grid_solve_parameters=grid_solve_parameters,
             mount_dirs="",
             singularity="",
             num_cpus=5,
             num_block_workers=5,
             queue="normal",
             num_cache_workers=5,
             min_solve_number=0)
