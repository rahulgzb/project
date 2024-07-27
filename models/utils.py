from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import os 
import torch 

class CheckpointManager:
    def __init__(self, dir_path, monitor='val_loss', mode='min', save_top_k=1):
        self.dir_path = dir_path
        os.makedirs(dir_path, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_value = None
        self.saved_checkpoints = []

    def save_checkpoint(self, model, epoch, metrics):
        current_value = metrics[self.monitor]
        if self.best_value is None or (self.mode == 'min' and current_value < self.best_value) or (self.mode == 'max' and current_value > self.best_value):
            self.best_value = current_value
            checkpoint_path = os.path.join(self.dir_path, f'checkpoint_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)
            if len(self.saved_checkpoints) > self.save_top_k:
                os.remove(self.saved_checkpoints.pop(0))

class trainner_helper():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _step(self, batch):
        # Move batch data to device
        input_ids = batch["source_ids"].to(self.device)
        attention_mask = batch["source_mask"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        target_mask = batch["target_mask"].to(self.device)
        
        lm_labels = target_ids.clone()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lm_labels=lm_labels,
            decoder_attention_mask=target_mask
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch):
        self.optimizer.zero_grad()
        loss = self._step(batch)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

    def validate(self, dataloader):
        self.eval()
        val_losses = []
        with torch.no_grad():
            for batch in dataloader:
                loss = self._step(batch)
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        self.train()
        return avg_val_loss




def train_model(hparams,model,train_dataloader,val_dataloader):
    
    checkpoint_manager = CheckpointManager(hparams.output_dir, monitor="avg_val_loss", mode="min", save_top_k=1)

    optimizer, lr_scheduler = model.configure_optimizers()
    scaler = torch.cuda.amp.GradScaler() if hparams.fp_16 else None
    trainer=trainner_helper()
    for epoch in range(hparams.num_train_epochs):
        model.train()
        train_losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=hparams.fp_16):
                loss = trainer._step(batch)
            if hparams.fp_16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)

        avg_val_loss = trainer.validate(val_dataloader)
        metrics = {"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss}
       
        checkpoint_manager.save_checkpoint(model, epoch, metrics)

        print(f"Epoch {epoch + 1}/{hparams.num_train_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

    # # Simulate test end logging (if needed)
    # test_dataloader = model.val_dataloader()  # Assuming you have a separate test dataset
    # avg_test_loss = model.validate(test_dataloader)
    # test_metrics = {"avg_test_loss": avg_test_loss}
   