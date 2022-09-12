import pandas as pd
import logging
import sys
from seq2seq_model import Seq2SeqModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = pd.read_csv("./data/idner2k/train_spc_tok.csv", sep=',').values.tolist()
train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = pd.read_csv("./data/idner2k/dev_spc_tok.csv", sep=',').values.tolist()
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 50,
    "train_batch_size": 10,
    "num_train_epochs": 20,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 25,
    "manual_seed": int(sys.argv[3]), # 4,
    "save_steps": 11898,
    "gradient_accumulation_steps": 10,
    "output_dir": sys.argv[1], # "./exp/template_indobart_200_5"
    "best_model_dir": sys.argv[2], # "./outputs/best_model_lalala"
}

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="mbart", # model too large CUDA Out of Memory
    encoder_decoder_name="facebook/mbart-large-50", # model too large CUDA Out of Memory
    # encoder_decoder_type="indobart",
    # encoder_decoder_name="indobenchmark/indobart",
    args=model_args,
    # use_cuda=False,
)
print('initialize model done')


# Train the model
model.train_model(train_df, eval_data=eval_df)
print('model training done')

# Evaluate the model
results = model.eval_model(eval_df)
print('model evaluation done')

# Use the model for prediction
print(model.predict(["Presiden terpilih Joko Widodo mengungkapkan pihaknya tidak akan membedakan spesifikasi kandidat menteri."]))
print(model.predict(["jelasnya di Kantor Transisi Joko Widodo Jusuf Kalla di Jalan Situbondo Nomor 10 , Menteng , Jakarta Pusat , Kamis (25/9/2014) ."]))
print(model.predict(["Badan Pengawas Pemilu (Bawaslu) menyampaikan temuannya ada sekitar 4.1 juta pemilih ."]))
print(model.predict(["Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a group c championship match on friday."]))
