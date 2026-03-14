import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ================= 处理字典数据的工具类 =================
class DictReplayDataset(Dataset):
    def __init__(self, data_list, bin_labels, mul_labels):
        self.data_list = data_list
        self.bin_labels = bin_labels
        self.mul_labels = mul_labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.bin_labels[idx], self.mul_labels[idx]

def dict_collate_fn(batch):
    dicts, bins, muls = zip(*batch)
    collated_dict = {}
    for k in dicts[0].keys():
        if isinstance(dicts[0][k], torch.Tensor):
            collated_dict[k] = torch.stack([d[k] for d in dicts])
        else:
            collated_dict[k] = [d[k] for d in dicts]
    return collated_dict, torch.stack(bins), torch.stack(muls)

# ================= 增量适配器 =================
class IncrementalAdapter:
    def __init__(self, model, config: dict, **kwargs):
        self.model = model
        self.config = config

        self.buffer_size = config.get('buffer_size', 2000)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('lr', 1e-4)
        self.epochs = config.get('epochs', 3)

        self.lambda_topo = config.get('lambda_topo', 0.1)
        self.lambda_cons = config.get('lambda_cons', 0.1)
        self.lambda_proto = config.get('lambda_proto', 0.1)

        self.replay_buffer = []

        self.classifiers = self._get_target_classifiers()
        logger.info(f"🔍 Joint Adapter targeted classifiers: {self.classifiers}")

    def _get_target_classifiers(self):
        clss = []
        if hasattr(self.model, 'is_malicious_classifier') and self.model.is_malicious_classifier is not None:
            clss.append('is_malicious_classifier')
        if hasattr(self.model, 'attack_family_classifier') and self.model.attack_family_classifier is not None:
            clss.append('attack_family_classifier')
        return clss

    def update_buffer(self, batch_dicts, bin_labels, mul_labels):
        batch_size = len(bin_labels)
        for i in range(batch_size):
            single_dict = {}
            for k, v in batch_dicts.items():
                if isinstance(v, torch.Tensor):
                    single_dict[k] = v[i].detach().cpu().clone()
                else:
                    single_dict[k] = v[i] if isinstance(v, list) else v

            b_lbl = bin_labels[i].detach().cpu().clone()
            m_lbl = mul_labels[i].detach().cpu().clone()

            if len(self.replay_buffer) < self.buffer_size:
                self.replay_buffer.append((single_dict, b_lbl, m_lbl))
            else:
                idx = torch.randint(0, len(self.replay_buffer), (1,)).item()
                self.replay_buffer[idx] = (single_dict, b_lbl, m_lbl)

    def _get_features_and_logits(self, classifier_or_name, x):
        if isinstance(classifier_or_name, str):
            classifier = getattr(self.model, classifier_or_name)
        else:
            classifier = classifier_or_name

        if isinstance(classifier, nn.Sequential):
            h = x
            for layer in classifier[:-1]:
                h = layer(h)
            logits = classifier[-1](h)
            return h, logits
        else:
            return x, classifier(x)

    def adapt(self, drift_data_batch, label_arg1, label_arg2=None, **kwargs):
        device = self.model.device

        # 解析双标签输入
        if label_arg2 is not None and isinstance(label_arg2, torch.Tensor):
            drift_bin_labels = label_arg1
            drift_mul_labels = label_arg2
        else:
            labels = label_arg1
            if labels.dim() > 1 and labels.shape[1] > 1:
                drift_mul_labels = torch.argmax(labels, dim=1)
                drift_bin_labels = (labels.sum(dim=-1) > 0).float()
            else:
                drift_bin_labels = labels.float()
                drift_mul_labels = torch.zeros_like(labels)

        self.update_buffer(drift_data_batch, drift_bin_labels, drift_mul_labels)

        if not self.replay_buffer:
            return

        # 拼装 DataLoader
        train_data_list = []
        batch_size = len(drift_bin_labels)
        for i in range(batch_size):
            single_dict = {}
            for k, v in drift_data_batch.items():
                if isinstance(v, torch.Tensor):
                    single_dict[k] = v[i].cpu()
                else:
                    single_dict[k] = v[i] if isinstance(v, list) else v
            train_data_list.append(single_dict)

        buf_dicts = [x[0] for x in self.replay_buffer]
        train_data_list.extend(buf_dicts)

        train_bins = torch.cat([drift_bin_labels.cpu(), torch.stack([x[1] for x in self.replay_buffer])], dim=0)
        train_muls = torch.cat([drift_mul_labels.cpu(), torch.stack([x[2] for x in self.replay_buffer])], dim=0)

        dataset = DictReplayDataset(train_data_list, train_bins, train_muls)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dict_collate_fn)

        self._freeze_backbone()

        # 动态收集所有解冻的参数 (包含了分类器和后6层BERT)
        opt_params = [p for p in self.model.parameters() if p.requires_grad]

        old_classifiers = {}
        for cls_name in self.classifiers:
            classifier = getattr(self.model, cls_name)
            old_c = copy.deepcopy(classifier)
            old_c.eval()
            for p in old_c.parameters(): p.requires_grad = False
            old_classifiers[cls_name] = old_c

        optimizer = optim.Adam(opt_params, lr=self.learning_rate)
        criterion_bin = nn.BCEWithLogitsLoss()
        criterion_mul = nn.CrossEntropyLoss()

        self.model.train()

        # 打破 Lightning Test 阶段的 no_grad() 枷锁
        with torch.enable_grad():
            for epoch in range(self.epochs):
                for batch_dict, batch_bin, batch_mul in loader:
                    batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
                    batch_bin, batch_mul = batch_bin.to(device), batch_mul.to(device)
                    optimizer.zero_grad()

                    outputs = self.model(batch_dict)
                    backbone_feats = outputs['multiview_embeddings']

                    loss_cls = torch.tensor(0.0, device=device)
                    total_loss_topo = torch.tensor(0.0, device=device)
                    total_loss_cons = torch.tensor(0.0, device=device)
                    total_loss_proto = torch.tensor(0.0, device=device)

                    for cls_name in self.classifiers:
                        h_new, logits_new = self._get_features_and_logits(cls_name, backbone_feats)

                        with torch.no_grad():
                            old_c = old_classifiers[cls_name]
                            h_old, _ = self._get_features_and_logits(old_c, backbone_feats.detach())

                        if cls_name == 'is_malicious_classifier':
                            target_y = batch_bin.unsqueeze(1).float() if batch_bin.dim() == 1 else batch_bin.float()
                            loss_cls += criterion_bin(logits_new, target_y)
                        elif cls_name == 'attack_family_classifier':
                            mask = (batch_bin.view(-1) == 1)
                            if mask.sum() > 0:
                                loss_cls += criterion_mul(logits_new[mask], batch_mul[mask].long())

                        # Topo 损失
                        dist_new = torch.cdist(h_new, h_new)
                        dist_old = torch.cdist(h_old, h_old)
                        total_loss_topo += nn.functional.mse_loss(dist_new, dist_old)

                        # Cons 损失
                        noise = torch.randn_like(backbone_feats) * 0.05
                        _, logits_noisy = self._get_features_and_logits(cls_name, backbone_feats + noise)
                        total_loss_cons += nn.functional.mse_loss(logits_noisy, logits_new.detach())

                        # Proto 损失
                        y_idx = batch_bin if cls_name == 'is_malicious_classifier' else batch_mul
                        h_proto = h_new
                        if cls_name == 'attack_family_classifier':
                            mask = (batch_bin.view(-1) == 1)
                            if mask.sum() > 1:
                                h_proto = h_new[mask]
                                y_idx = y_idx[mask]
                            else:
                                continue

                        unique_c = torch.unique(y_idx.long())
                        valid_p = 0
                        lp = torch.tensor(0.0, device=device)

                        for c in unique_c:
                            c_mask = (y_idx.long() == c)
                            if c_mask.sum() > 1:
                                feats_c = h_proto[c_mask]
                                proto = feats_c.mean(dim=0, keepdim=True)
                                lp += nn.functional.mse_loss(feats_c, proto.expand_as(feats_c))
                                valid_p += 1

                        if valid_p > 0:
                            total_loss_proto += lp / valid_p

                    loss = loss_cls + (self.lambda_topo * total_loss_topo) + (self.lambda_cons * total_loss_cons) + (self.lambda_proto * total_loss_proto)

                    loss.backward()
                    optimizer.step()

        self._unfreeze_backbone()
        self.model.eval()

        del old_classifiers, optimizer, dataset, loader
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _freeze_backbone(self):
        for param in self.model.parameters(): param.requires_grad = False
        for cls_name in self.classifiers:
            if hasattr(self.model, cls_name) and getattr(self.model, cls_name) is not None:
                for param in getattr(self.model, cls_name).parameters(): param.requires_grad = True
        for name, param in self.model.named_parameters():
            if "layer." in name:
                try:
                    layer_num = int(name.split("layer.")[1].split(".")[0])
                    if layer_num >= 6: param.requires_grad = True
                except ValueError:
                    pass

    def _unfreeze_backbone(self):
        for param in self.model.parameters(): param.requires_grad = True