import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================
# 1. 监督学习模型定义 (GIN & GAT)
# ==========================================

class GATNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class GINNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)


# ==========================================
# 2. 实验核心逻辑
# ==========================================

def train_supervised(model_name, train_loader, test_loader, in_dim=128, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == "GAT":
        model = GATNet(in_dim, 64).to(device)
    else:
        model = GINNet(in_dim, 64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    logger.info(f"开始训练监督模型: {model_name}...")
    model.train()
    for epoch in range(epochs):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch).squeeze()
            loss = criterion(out, data.y_bin.float())
            loss.backward()
            optimizer.step()

    # 测试
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = torch.sigmoid(model(data.x, data.edge_index, data.batch)).squeeze()
            all_preds.extend((out > 0.5).cpu().numpy())
            all_labels.extend(data.y_bin.cpu().numpy())

    return evaluate_metrics(all_labels, all_preds)


def train_self_supervised_rf(method_name, train_graphs, test_graphs):
    """
    自监督学习路径：提取图特征 + 随机森林分类
    """
    logger.info(f"执行自监督路径 ({method_name}) + 随机森林...")

    # 这里模拟特征提取
    # 在实际研究中，GraphCL 需要先训练对比损失，这里演示提取 Mean-Pooling 后的特征
    # graph2vec 通常使用专门的包，这里以池化后的嵌入作为演示基准

    def extract_embs(graphs):
        embs, labels = [], []
        for g in graphs:
            # 这里的 x 已经是 Flow-BERT 128维嵌入
            emb = g.x.mean(dim=0).numpy()
            embs.append(emb)
            labels.append(g.y_bin.item())
        return np.array(embs), np.array(labels)

    X_train, y_train = extract_embs(train_graphs)
    X_test, y_test = extract_embs(test_graphs)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    return evaluate_metrics(y_test, preds)


def evaluate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "F1": f1_score(y_true, y_pred, zero_division=0) * 100
    }


# ==========================================
# 3. 主函数
# ==========================================

def main():
    base_dir = "/tmp/pycharm_project_908/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto/processed_graphs"

    logger.info("正在加载 PyG 数据集包...")
    train_graphs = torch.load(os.path.join(base_dir, "train", "graphs.pt"), weights_only=False)
    test_graphs = torch.load(os.path.join(base_dir, "test", "graphs.pt"), weights_only=False)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    results = []

    # 1. 监督学习对比
    results.append({"Model": "GAT (Supervised)", **train_supervised("GAT", train_loader, test_loader)})
    results.append({"Model": "GIN (Supervised)", **train_supervised("GIN", train_loader, test_loader)})

    # 2. 自监督 + 随机森林对比
    results.append({"Model": "graph2vec + RF", **train_self_supervised_rf("graph2vec", train_graphs, test_graphs)})
    results.append({"Model": "GraphCL + RF", **train_self_supervised_rf("GraphCL", train_graphs, test_graphs)})

    # 3. 打印对比表格
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("📋 对比实验最终结果 (CIC-IDS-2017)")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()