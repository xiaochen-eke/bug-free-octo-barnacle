"""
终极版评估脚本 - 自动修复路径，在任何目录运行都可以
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
# 自动路径修复（核心）
# =====================================

def setup_paths():
    """自动检测并设置正确的路径"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    print(f"🔍 当前脚本: {current_file}")
    print(f"📁 当前目录: {current_dir}")
    
    # 找到 backend 目录
    if 'evaluation' in current_dir:
        # 在 evaluation 目录运行
        backend_dir = os.path.dirname(os.path.dirname(current_dir))
        print("📍 检测到在 evaluation 目录运行")
    elif 'training' in current_dir:
        # 在 training 目录运行
        backend_dir = os.path.dirname(current_dir)
        print("📍 检测到在 training 目录运行")
    else:
        # 其他目录，假设是 backend 或其子目录
        backend_dir = current_dir
        # 往上找 backend 目录
        while backend_dir and not backend_dir.endswith('backend'):
            parent = os.path.dirname(backend_dir)
            if parent == backend_dir:  # 到根目录了
                break
            backend_dir = parent
        print("📍 在其他目录运行")
    
    print(f"✓ Backend 目录: {backend_dir}")
    
    # 设置路径
    models_dir = os.path.join(backend_dir, "models")
    datasets_dir = os.path.join(backend_dir, "datasets")
    
    print(f"✓ Models 目录: {models_dir}")
    print(f"✓ Datasets 目录: {datasets_dir}")
    
    # 添加到 sys.path
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    if datasets_dir not in sys.path:
        sys.path.insert(0, datasets_dir)
    
    print("✓ 路径已配置\n")
    
    return backend_dir, models_dir, datasets_dir

# 调用路径修复
BACKEND_DIR, MODELS_DIR, DATASETS_DIR = setup_paths()

# 现在导入模块应该可以工作了
try:
    from transformer import TransformerModel
    print("✓ 成功导入 TransformerModel")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print(f"  请确保 {MODELS_DIR}/transformer.py 存在")
    sys.exit(1)

try:
    from translation_dataset import TranslationDataset
    print("✓ 成功导入 TranslationDataset\n")
    HAS_TRANSLATION_DATASET = True
except ImportError:
    print("⚠️  TranslationDataset 不存在，部分功能不可用\n")
    HAS_TRANSLATION_DATASET = False


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
            
            # 前向传播
            logits = model(src, tgt_input)
            
            # Loss
            if criterion is not None:
                loss = criterion(
                    logits.reshape(-1, vocab_size),
                    tgt_output.reshape(-1)
                ).item()
            else:
                loss = 0.0
            
            # 预测
            predictions = logits.argmax(dim=-1)
            
            # 指标
            bleu = batch_bleu(predictions, tgt_output)
            cer = batch_cer(predictions, tgt_output)
            
            # Token 准确率
            mask = (tgt_output != 0).float()
            token_acc = ((predictions == tgt_output).float() * mask).sum() / mask.sum().clamp(min=1e-8)
            token_acc = token_acc.item()
            
            # 累加
            metrics['total_loss'] += loss
            metrics['total_bleu'] += bleu
            metrics['total_cer'] += cer
            metrics['total_token_acc'] += token_acc
            metrics['num_batches'] += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  [{batch_idx+1}/{len(data_loader)}] Loss: {loss:.4f}, BLEU: {bleu:.4f}, Acc: {token_acc:.4f}")
    
    # 平均值
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
    
    # =====================================
    # 加载模型
    # =====================================
    
    print("📦 加载模型...")
    model = TransformerModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✓ 模型已加载: {args.model_path}\n")
    else:
        print(f"⚠️  模型文件不存在: {args.model_path}")
        print(f"    会使用随机初始化的模型\n")
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    results = {}
    
    # =====================================
    # 验证集
    # =====================================
    
    if os.path.exists(args.val_csv):
        print(f"📊 评估验证集: {args.val_csv}")
        
        if not HAS_TRANSLATION_DATASET:
            print("⚠️  TranslationDataset 不可用")
        else:
            val_dataset = TranslationDataset(
                csv_file=args.val_csv,
                sp_model_path=args.sp_model,
                max_length=args.max_len
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0  # Windows 不支持多进程，改为0
            )
            
            print(f"  数据大小: {len(val_dataset):,}")
            val_metrics = evaluate_dataset(model, val_loader, DEVICE, VOCAB_SIZE, criterion)
            print_metrics(val_metrics, "验证集")
            results['validation'] = val_metrics
    else:
        print(f"⚠️  验证集不存在: {args.val_csv}\n")
    
    # =====================================
    # 测试集
    # =====================================
    
    if os.path.exists(args.test_csv):
        print(f"📊 评估测试集: {args.test_csv}")
        
        if not HAS_TRANSLATION_DATASET:
            print("⚠️  TranslationDataset 不可用")
        else:
            test_dataset = TranslationDataset(
                csv_file=args.test_csv,
                sp_model_path=args.sp_model,
                max_length=args.max_len
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0  # Windows 不支持多进程，改为0
            )
            
            print(f"  数据大小: {len(test_dataset):,}")
            test_metrics = evaluate_dataset(model, test_loader, DEVICE, VOCAB_SIZE, criterion)
            print_metrics(test_metrics, "测试集")
            results['test'] = test_metrics
    else:
        print(f"⚠️  测试集不存在: {args.test_csv}\n")
    
    # =====================================
    # 总结
    # =====================================
    
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