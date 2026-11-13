SEED_FRAC = 0.05       # initial labeled share of train files
QUERY_FRAC = 0.05      # files added per round
N_ROUNDS = 5           # rounds after the seed -> 6 evaluation points (5%..30%)
AL_EPOCHS = 12         # minimum epochs per round
MIN_STEPS = 500        # minimum optimizer steps per round
RARE_QUANTILE = 0.25   # "rare" = bottom quarter of species by train chunks
