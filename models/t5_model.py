import torch 
from transformers import T5ForConditionalGeneration

class T5FineTuner(torch.nn.Module):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path).to(self.device)
        # self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        # self.optimizer, self.lr_scheduler = self.configure_optimizers()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
