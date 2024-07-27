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
    
    def generate(self, input_text, max_length=50, num_beams=5, early_stopping=True, device='cpu'):
        """
        Generate text using the T5 model.

        Args:
            input_text (str): The input text to generate from.
            max_length (int): The maximum length of the generated sequence.
            num_beams (int): The number of beams for beam search.
            early_stopping (bool): Whether to stop the beam search when at least `num_beams` sentences are finished per batch.
            device (str): The device to run the model on ('cpu' or 'cuda').

        Returns:
            str: The generated text.
        """
        self.model.to(device)
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping
            )
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return generated_text
    




def load_model_and_tokenizer(checkpoint_path,hparams ):
    model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    model.load_state_dict(torch.load(checkpoint_path))
    tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
    return model, tokenizer

