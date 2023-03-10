import os
PATH = os.getcwd()+'/'

MODEL = 'path'

DATA  = PATH + 'DATA/'
PROCESSED_DATA = PATH + 'PROCESSED_DATA/'
PROCESSED_FILE = PROCESSED_DATA+'/output_cleaned.txt'
LOGS = PATH + '/LOGS'
TRAINED_MODEL_PATH = PATH + 'MODEL/'
TRAINED_MODEL = PATH +'MODEL/modelv1'

EPOCHS = 1
BATCH_SIZE = 8
STEPS = 2
LOGGING_STEPS = 1

EXTRACT_DATA = False
CLEAN_DATA = False
TRAIN = False
EVAL = True