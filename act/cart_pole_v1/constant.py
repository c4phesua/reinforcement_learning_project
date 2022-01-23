import os

DATABASE_NAME = 'cart_pole_v1.db'
TABLE_NAME = 'cart_pole_v1_save'
EXPERIENCE_REPLAY_SAVE = os.path.join(os.path.dirname(__file__),
                                      '..', '..', 'training_results', 'cart_pole_v1_memory.pkl')
