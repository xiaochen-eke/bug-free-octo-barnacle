"""
最终修复版评估脚本 - 正确检测 backend 目录
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =====================================
# 最终修复的路径检测
# =====================================

def find_backend_dir():
    """
    正确找到 backend 目录
    
    目录结构:
    multilingual_transformer/
    └── backend/
        ├── models/
        ├── datasets/
        ├── training/
        └── evaluation/  ← 当前目录
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    print(f"🔍 当前脚本: {current_file}")
    print(f"📁 当前目录: {current_dir}\n")
    
    # 检查当前目录是否在 evaluation 或 training 目录中
    if 'evaluation' in current_dir:
        # C:\...\backend\evaluation
        # 需要往上走两层到 backend
        backend_dir = os.path.dirname(os.path.dirname(current_dir))
        print(f"📍 检测到在 evaluation 目录")
        print(f"   往上走两层")
    elif 'training' in current_dir:
        # C:\...\backend\training
        # 需要往上走一层到 backend
        backend_dir = os.path.dirname(current_dir)
        print(f"📍 检测到在 training 目录")
        print(f"   往上走一层")
    else:
        # 其他目录，假设是 backend 目录或其子目录
        backend_dir = current_dir
        print(f"📍 在其他目录运行")
    
    # 验证 models 和 datasets 是否存在
    models_dir = os.path.join(backend_dir, "models")
    datasets_dir = os.path.join(backend_dir, "datasets")
    
    # 如果 models 目录不存在，说明检测错了
    # 尝试往上找 backend 目录
    while not os.path.exists(models_dir):
        parent_dir = os.path.dirname(backend_dir)
        if parent_dir == backend_dir:  # 到根目录了
            print(f"\n❌ 无法找到 backend 目录!")
            print(f"   已尝试的路径: {backend_dir}")
            return None
        
        # 尝试上一级
        backend_dir = parent_dir
        models_dir = os.path.join(backend_dir, "models")
        datasets_dir = os.path.join(backend_dir, "datasets")
    
    print(f"\n✓ Backend 目录: {backend_dir}")
    print(f"✓ Models 目录: {models_dir}")
    print(f"✓ Datasets 目录: {datasets_dir}")
    
    # 验证目录结构
    if not os.path.exists(models_dir):
        print(f"\n❌ Models 目录不存在: {models_dir}")
        return None
    if not os.path.exists(datasets_dir):
        print(f"⚠️  Datasets 目录不存在: {datasets_dir}")
    
    return backend_dir, models_dir, datasets_dir

# 调用路径检测
result = find_backend_dir()
if result is None:
    print("\n❌ 无法找到必要的目录，请检查文件位置")
    sys.exit(1)

BACKEND_DIR, MODELS_DIR, DATASETS_DIR = result

# 添加到 sys.path
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, DATASETS_DIR)

print("✓ 路径已配置\n")

# 导入模块
try:
    from transformer import TransformerModel
    print("✓ 成功导入 TransformerModel")
except ImportError as e:
    print(f"✗ 导入 transformer 失败: {e}")
    print(f"  请确保 {MODELS_DIR}/transformer.py 存在")
    sys.exit(1)

try:
    from translation_dataset import TranslationDataset
    print("✓ 成功导入 TranslationDataset")
    HAS_TRANSLATION_DATASET = True
except ImportError:
    print("⚠️  TranslationDataset 导入失败，部分功能不可用")
    HAS_TRANSLATION_DATASET = False

print()


# =====================================
# BLEU 计算
# =====================================

def calculate_bleu(pred_tokens: List[int], ref_tokens: List[int], n=4) -> float:
    """计算 BLEU 分数"""
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    pred_tokens = [t for t in pred_tokens if t > 0]
    ref_tokens = [t for t in ref_tokens if t > 0]

    if len(pred_tokens) == 0:
        return 0.0

    bleu_scores = []
    for gram_n in range(1, min(n, len(pred_tokens)) + 1):
        pred_grams = set()
        ref_grams = set()

        for i in range(len(pred_tokens) - gram_n + 1):
            pred_grams.add(tuple(pred_tokens[i:i+gram_n]))

        for i in range(len(ref_tokens) - gram_n + 1):
            ref_grams.add(tuple(ref_tokens[i:i+gram_n]))

        if len(ref_grams) == 0:
            bleu_scores.append(0.0)
        else:
            precision = len(pred_grams & ref_grams) / len(ref_grams)
            bleu_scores.append(precision)

    if all(s > 0 for s in bleu_scores):
        bleu = np.exp(np.mean(np.log(bleu_scores)))
    else:
        bleu = 0.0

    return bleu


def batch_bleu(predictions, targets):
    """批量计算BLEU"""
    batch_bleu = []
    for pred, tgt in zip(predictions, targets):
        pred_tokens = pred.tolist()
        tgt_tokens = tgt.tolist()
        bleu = calculate_bleu(pred_tokens, tgt_tokens)
        batch_bleu.append(bleu)
    return np.mean(batch_bleu) if batch_bleu else 0.0


# =====================================
# CER 计算
# =====================================

def calculate_cer(pred_tokens: List[int], ref_tokens: List[int]) -> float:
    """字符错误率"""
    if len(ref_tokens) == 0:
        return 0.0 if len(pred_tokens) == 0 else 1.0
    return edit_distance(pred_tokens, ref_tokens) / len(ref_tokens)


def edit_distance(s1, s2):
    """编辑距离"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def batch_cer(predictions, targets):
    """批量计算CER"""
    batch_cer = []
    for pred, tgt in zip(predictions, targets):
        pred_tokens = [t for t in pred.tolist() if t > 0]
        tgt_tokens = [t for t in tgt.tolist() if t > 0]
        cer = calculate_cer(pred_tokens, tgt_tokens)
        batch_cer.append(cer)
    return np.mean(batch_cer) if batch_cer else 0.0


# =====================================
# 评估
# =====================================

def evaluate_dataset(model, data_loader, device, vocab_size, criterion=None):
    """评估数据集"""
    model.eval()
    
    metrics = {
        'total_loss': 0,
        'total_bleu': 0,
        'total_cer': 0,
        'total_token_acc': 0,
        'num_batches': 0,
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            
            if criterion is not None:
                loss = criterion(
                    logits.reshape(-1, vocab_size),
                    tgt_output.reshape(-1)
                ).item()
            else:
                loss = 0.0
            
            predictions = logits.argmax(dim=-1)
            bleu = batch_bleu(predictions, tgt_output)
            cer = batch_cer(predictions, tgt_output)
            
            mask = (tgt_output != 0).float()
            token_acc = ((predictions == tgt_output).float() * mask).sum() / mask.sum().clamp(min=1e-8)
            token_acc = token_acc.item()
            
            metrics['total_loss'] += loss
            metrics['total_bleu'] += bleu
            metrics['total_cer'] += cer
            metrics['total_token_acc'] += token_acc
            metrics['num_batches'] += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  [{batch_idx+1}/{len(data_loader)}] Loss: {loss:.4f}, BLEU: {bleu:.4f}, Acc: {token_acc:.4f}")
    
    if metrics['num_batches'] > 0:
        metrics['avg_loss'] = metrics['total_loss'] / metrics['num_batches']
        metrics['avg_bleu'] = metrics['total_bleu'] / metrics['num_batches']
        metrics['avg_cer'] = metrics['total_cer'] / metrics['num_batches']
        metrics['avg_token_acc'] = metrics['total_token_acc'] / metrics['num_batches']
        metrics['avg_ppl'] = torch.exp(torch.tensor(metrics['avg_loss'])).item()
    
    return metrics


def print_metrics(metrics, name="Dataset"):
    """打印指标"""
    print(f"\n{'='*80}")
    print(f"📊 {name} 评估结果")
    print(f"{'='*80}")
    print(f"  Loss:       {metrics.get('avg_loss', 0):.4f}")
    print(f"  PPL:        {metrics.get('avg_ppl', 0):.2f}")
    print(f"  BLEU:       {metrics.get('avg_bleu', 0):.4f} ✨")
    print(f"  CER:        {metrics.get('avg_cer', 0):.4f}")
    print(f"  Token Acc:  {metrics.get('avg_token_acc', 0):.4f}")
    print(f"{'='*80}\n")


# =====================================
# Main
# =====================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default=os.path.join(BACKEND_DIR, 'checkpoints', 'best_model.pth'),
                        help='模型路径')
    parser.add_argument('--val-csv', type=str,
                        default=os.path.join(BACKEND_DIR, 'data', 'processed', 'valid.csv'),
                        help='验证集CSV')
    parser.add_argument('--test-csv', type=str,
                        default=os.path.join(BACKEND_DIR, 'data', 'processed', 'test.csv'),
                        help='测试集CSV')
    parser.add_argument('--sp-model', type=str,
                        default=os.path.join(BACKEND_DIR, 'tokenizer', 'multilingual.model'),
                        help='Tokenizer模型')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-len', type=int, default=128)
    
    args = parser.parse_args()
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VOCAB_SIZE = 32000
    
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model_path}\n")
    
    # 加载模型
    print("📦 加载模型...")
    model = TransformerModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✓ 模型已加载\n")
    else:
        print(f"⚠️  模型文件不存在: {args.model_path}\n")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    results = {}
    
    # 验证集
    if os.path.exists(args.val_csv) and HAS_TRANSLATION_DATASET:
        print(f"📊 评估验证集: {args.val_csv}")
        
        val_dataset = TranslationDataset(
            csv_file=args.val_csv,
            sp_model_path=args.sp_model,
            max_length=args.max_len
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"  数据大小: {len(val_dataset):,}")
        val_metrics = evaluate_dataset(model, val_loader, DEVICE, VOCAB_SIZE, criterion)
        print_metrics(val_metrics, "验证集")
        results['validation'] = val_metrics
    else:
        if not os.path.exists(args.val_csv):
            print(f"⚠️  验证集不存在: {args.val_csv}\n")
        if not HAS_TRANSLATION_DATASET:
            print(f"⚠️  TranslationDataset 不可用\n")
    
    # 测试集
    if os.path.exists(args.test_csv) and HAS_TRANSLATION_DATASET:
        print(f"📊 评估测试集: {args.test_csv}")
        
        test_dataset = TranslationDataset(
            csv_file=args.test_csv,
            sp_model_path=args.sp_model,
            max_length=args.max_len
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"  数据大小: {len(test_dataset):,}")
        test_metrics = evaluate_dataset(model, test_loader, DEVICE, VOCAB_SIZE, criterion)
        print_metrics(test_metrics, "测试集")
        results['test'] = test_metrics
    else:
        if not os.path.exists(args.test_csv):
            print(f"⚠️  测试集不存在: {args.test_csv}\n")
    
    # 总结
    if results:
        print(f"\n{'='*80}")
        print("📈 评估总结")
        print(f"{'='*80}")
        
        if 'validation' in results:
            val = results['validation']
            print(f"\n验证集:")
            print(f"  BLEU: {val['avg_bleu']:.4f}")
            print(f"  Acc:  {val['avg_token_acc']:.4f}")
            print(f"  PPL:  {val['avg_ppl']:.2f}")
        
        if 'test' in results:
            test = results['test']
            print(f"\n测试集:")
            print(f"  BLEU: {test['avg_bleu']:.4f}")
            print(f"  Acc:  {test['avg_token_acc']:.4f}")
            print(f"  PPL:  {test['avg_ppl']:.2f}")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()