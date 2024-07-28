from datasets import load_dataset
import pandas as pd
import os



from torch.utils.data import Dataset
class QuoraDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=64):
        self.path = os.path.join(data_dir, type_path + '.csv')
        self.question = "question"
        self.target_column = "target"
        self.data = pd.read_csv(self.path)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
    


    def _build(self):
        for idx in range(len(self.data)):
            target,question= self.data.loc[idx, self.target_column], self.data.loc[idx, self.question]
    
            input_ = "Question: %s </s>" % (question)
            target = "Mimik_human: %s </s>" %(target)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

def get_dataset(tokenizer, type_path, args):
    return QuoraDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


