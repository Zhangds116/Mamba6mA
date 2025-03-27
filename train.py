from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from dna_map import read_dna_sequences, map_to_indices
from net.mamba import Mamba
from net.model_args import ModelArgs
import numpy as np
import torch




batch_size = 64
learning_rate = 0.00005
num_epochs = 200
def train_main(neg_path, pos_path, val_neg_path, val_pos_path):
    negative_file_path = neg_path
    negative_sequences = read_dna_sequences(negative_file_path)

    positive_file_path = pos_path
    positive_sequences = read_dna_sequences(positive_file_path)


    negative_encoded = map_to_indices(negative_sequences)
    positive_encoded = map_to_indices(positive_sequences)

    X = np.concatenate([negative_encoded, positive_encoded], axis=0).astype(np.int64)
    y = np.concatenate([np.zeros(len(negative_encoded)), np.ones(len(positive_encoded))]).astype(np.int64)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    val_negative_file_path = val_neg_path
    val_negative_sequences = read_dna_sequences(val_negative_file_path)
    val_positive_file_path = val_pos_path
    val_positive_sequences = read_dna_sequences(val_positive_file_path)


    val_negative_encoded = map_to_indices(val_negative_sequences)
    val_positive_encoded = map_to_indices(val_positive_sequences)

    X_val = np.concatenate([val_negative_encoded, val_positive_encoded], axis=0)
    y_val = np.concatenate([np.zeros(len(val_negative_encoded)), np.ones(len(val_positive_encoded))])

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model_args = ModelArgs()
    model_args.__post_init__()
    mamba_model = Mamba(model_args)

    optimizer = optim.Adam(mamba_model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    mamba_model.to(device)
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 30

    for epoch in range(num_epochs):
        mamba_model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            logits = mamba_model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.round(logits)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        mamba_model.eval()
        with torch.no_grad():
            val_preds = []
            val_logits = []
            for inputs, labels in tqdm(val_dataloader, desc='Validating'):
                inputs = inputs.to(device)
                batch_logits = mamba_model(inputs)
                batch_preds = torch.round(batch_logits).detach().cpu().numpy()
                val_preds.extend(batch_preds)
                val_logits.extend(batch_logits.cpu().numpy())

            val_accuracy = accuracy_score(y_val_tensor.cpu().numpy(), val_preds)
            val_auc = roc_auc_score(y_val_tensor.cpu().numpy(), val_logits)
            val_mcc = matthews_corrcoef(y_val_tensor.cpu().numpy(), val_preds)
            tn, fp, fn, tp = confusion_matrix(y_val_tensor.cpu().numpy(), val_preds).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)

            print(f'Validation Metrics: Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, MCC: {val_mcc:.4f}, '
                  f'Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(mamba_model.state_dict(), 'best_mamba_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                break

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)

        print(f'Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')


    print(f'best epoch {best_epoch} ')


if __name__ == "__main__":
     train_main('./data/6mA_C.elegans/train_neg.txt',
                './data/6mA_C.elegans/train_pos.txt',
                './data/6mA_C.elegans/test_neg.txt',
                './data/6mA_C.elegans/test_pos.txt')


