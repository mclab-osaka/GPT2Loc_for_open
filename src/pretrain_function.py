import torch
from utils import visualize_tsne
import torch.nn as nn
import math


def pre_train(
    model,
    epochs,
    train_loader,
    optimizer_for_pretrain,
    criterion,
    model_name,
    device,
    pretrain_method="rnc",
    is_mask=True,
):

    print(len(train_loader))

    for epoch in range(epochs + 1):
        model.train()
        total_loss = 0
        encoded_features_array = []
        labels_array = []

        for (
            id_input,
            rssi_input,
            labels,
            noised_rssi_input_0,
            noised_rssi_input_1,
            at_mask_0,
            at_mask_1,
            label_0,
            label_1,
        ) in train_loader:

            B, L = id_input.size(0), id_input.size(1)
            id_input = id_input.to(device)
            rssi_input = rssi_input.to(device)

            rssi_input = rssi_input.view(B, 1, L).to(device)
            noised_rssi_input_0 = noised_rssi_input_0.view(B, 1, L).to(device)
            noised_rssi_input_1 = noised_rssi_input_1.view(B, 1, L).to(device)

            labels = labels.to(device)
            if pretrain_method == "label_aug_rnc":
                labels = label_1.to(device)

            _at_mask_0 = None
            _at_mask_1 = None
            if is_mask:
                _at_mask_0 = at_mask_0.to(device)
                _at_mask_1 = at_mask_1.to(device)

            optimizer_for_pretrain.zero_grad()

            if pretrain_method == "rnc":
                # Forward pass
                loc, noised_encoded_features_0, recovered_rssi_0 = model(
                    id_input, noised_rssi_input_0, attention_mask=_at_mask_0
                )
                loc , noised_encoded_features_1, recovered_rssi_1 = model(
                    id_input, noised_rssi_input_1, attention_mask=_at_mask_1
                )
                features = torch.cat(
                    [
                        noised_encoded_features_0.unsqueeze(1),
                        noised_encoded_features_1.unsqueeze(1),
                    ],
                    dim=1,
                )
                loss = criterion(features, labels)
            elif pretrain_method == "bert":
                _, noised_encoded_features_0, recovered_rssi_0 = model(
                    id_input, noised_rssi_input_0, attention_mask=_at_mask_0
                )
                criterion = nn.MSELoss()
                rssi_input = rssi_input.squeeze()
                # print(recovered_rssi_0.size(), rssi_input.size())
                loss = criterion(recovered_rssi_0, rssi_input)
            elif pretrain_method == "end-to-end":
                loc, noised_encoded_features_0, recovered_rssi_0 = model(
                    id_input, rssi_input, attention_mask=None
                )
                criterion = nn.MSELoss()
                # print(recovered_rssi_0.size(), rssi_input.size())
                loss = criterion(loc, labels)

            # loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer_for_pretrain.step()

            total_loss += loss.item()

            if epoch % epochs == 0 and epoch > 0:
                encoded_features_array.append(noised_encoded_features_0)
                labels_array.append(labels)

        if epoch % epochs == 0 and epoch > 0:
            _encoded_features = torch.cat(encoded_features_array, dim=0)
            _labels = torch.cat(labels_array, dim=0)

            visualize_tsne(_encoded_features, _labels, pos=0, prefix="pretrain")
            visualize_tsne(_encoded_features, _labels, pos=1, prefix="pretrain")
            # visualize_tsne(rssi_input.squeeze(1), labels, pos=1, method="radio_map")
            visualize_tsne(_encoded_features, _labels, pos=0, method="radio_map")

        if epoch % 100 == 0:
            model_path = "./model/{}_{}.pth".format(model_name, epoch)
            torch.save(model.state_dict(), model_path)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}"
        )

    return model
