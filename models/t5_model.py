import torch 
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,

)

class T5FineTuner(torch.nn.Module):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        #self.optimizer, self.lr_scheduler = self.configure_optimizers()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
    
    def generate(self, input_ids, attention_mask=None,max_length=50,**kwargs):

        self.model.to(self.device)
      
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=50,
                    num_beams=5,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
              
            )
        return generated_ids
    


def infer_single_sentence(model, tokenizer, sentence, max_length=50):
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    output_ids = model.generate(inputs,attention_mask=None, max_length=max_length)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

def load_model_and_tokenizer(checkpoint_path,hparams ):
    model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    model.load_state_dict(torch.load(checkpoint_path))
    tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
    return model, tokenizer

