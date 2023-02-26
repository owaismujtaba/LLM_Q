import config


if __name__ == '__main__':
    
    if config.EXTRACT_DATA:
        from data_utils import extract_write_text
        extract_write_text()
        
    if config.CLEAN_DATA:
        from data_utils import clean_data
        clean_data()
        
    
    if config.TRAIN:
        from train import trainer
        trainer()
        
        
    if config.EVAL:
        from evaluations import evaluate_model
        evaluate_model()