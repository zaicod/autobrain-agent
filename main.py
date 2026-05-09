import torch
import torch.optim as optim
from models.multimodal_fusion import AutoBrainFusionNet
from script.pretrain_tage_multisr_masked_reconstruction import TrainingMonitorAgent

def mock_data_loader(batch_size=16):
    eeg = torch.randn(batch_size, 128)
    fnirs = torch.randn(batch_size, 72, 100)
    labels = torch.randint(0, 2, (batch_size,))
    return eeg, fnirs, labels

def run_agent_pipeline():
    print("Initializing AutoBrain-Agent Workflow...")
    
    model = AutoBrainFusionNet(num_classes=2)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    agent = TrainingMonitorAgent(model, optimizer, target_val_loss=0.6)
    

    epochs = 5
    for epoch in range(epochs):
        model.train()
        eeg_batch, fnirs_batch, targets = mock_data_loader()

        outputs = model(eeg_batch, fnirs_batch)

        loss = agent.optimize_step(outputs, targets, outputs[1], epoch)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

        val_loss_sim = 0.85 - (epoch * 0.08) 
        agent.validate_epoch(val_loss_sim)

if __name__ == "__main__":
    run_agent_pipeline()