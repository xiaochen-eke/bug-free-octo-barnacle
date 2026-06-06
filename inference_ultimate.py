"""
终极版推理脚本 - 自动修复路径，在任何目录运行
"""

import os
import sys
from typing import List

import torch
import sentencepiece as spm

# =====================================
# 自动路径修复（核心）
# =====================================

def setup_paths():
    """自动检测并设置正确的路径"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # 找到 backend 目录
    if 'evaluation' in current_dir:
        backend_dir = os.path.dirname(os.path.dirname(current_dir))
    elif 'training' in current_dir:
        backend_dir = os.path.dirname(current_dir)
    else:
        backend_dir = current_dir
        while backend_dir and not backend_dir.endswith('backend'):
            parent = os.path.dirname(backend_dir)
            if parent == backend_dir:
                break
            backend_dir = parent
    
    # 设置路径
    models_dir = os.path.join(backend_dir, "models")
    datasets_dir = os.path.join(backend_dir, "datasets")
    
    # 添加到 sys.path
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    if datasets_dir not in sys.path:
        sys.path.insert(0, datasets_dir)
    
    return backend_dir, models_dir, datasets_dir

BACKEND_DIR, MODELS_DIR, DATASETS_DIR = setup_paths()

# 导入模块
try:
    from transformer import TransformerModel
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print(f"  Models 目录: {MODELS_DIR}")
    sys.exit(1)


# =====================================
# 推理器
# =====================================

class Translator:
    """多语言翻译推理器"""
    
    def __init__(self, model_path: str, sp_model_path: str, 
                 vocab_size: int = 32000, max_len: int = 128,
                 device: str = 'cuda'):
        """初始化翻译器"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        print("📦 加载模型...")
        self.model = TransformerModel(vocab_size=vocab_size).to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✓ 模型已加载")
        else:
            raise FileNotFoundError(f"模型不存在: {model_path}")
        
        self.model.eval()
        
        print("🔤 加载 Tokenizer...")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        print(f"✓ Tokenizer已加载\n")
    
    def encode(self, text: str, lang_token: str = "") -> torch.Tensor:
        """编码文本"""
        if lang_token:
            text = f"{lang_token} {text}"
        
        tokens = self.sp.EncodeAsIds(text)
        
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        
        return torch.tensor([tokens], dtype=torch.long).to(self.device)
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token"""
        token_ids = [t for t in token_ids if t > 0]
        text = self.sp.DecodeIds(token_ids)
        return text
    
    def translate(self, src_text: str, src_lang: str = "", beam_size: int = 1) -> str:
        """翻译文本"""
        src_tokens = self.encode(src_text, src_lang)
        
        with torch.no_grad():
            tgt_tokens = torch.tensor([[0]], dtype=torch.long).to(self.device)
            
            for step in range(self.max_len - 1):
                logits = self.model(src_tokens, tgt_tokens)
                next_token_logits = logits[0, -1, :]
                
                if beam_size == 1:
                    next_token = next_token_logits.argmax(dim=-1).item()
                else:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, beam_size)
                    next_token = top_k_indices[0].item()
                
                tgt_tokens = torch.cat([
                    tgt_tokens,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)
                
                if next_token == 0:
                    break
        
        tgt_ids = tgt_tokens[0].tolist()
        translation = self.decode(tgt_ids)
        return translation
    
    def translate_batch(self, texts: List[str], src_lang: str = "") -> List[str]:
        """批量翻译"""
        results = []
        for text in texts:
            result = self.translate(text, src_lang)
            results.append(result)
        return results


# =====================================
# 命令行接口
# =====================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多语言翻译推理')
    parser.add_argument('--model-path', type=str,
                        default=os.path.join(BACKEND_DIR, 'checkpoints', 'best_model.pth'),
                        help='模型路径')
    parser.add_argument('--sp-model', type=str,
                        default=os.path.join(BACKEND_DIR, 'tokenizer', 'multilingual.model'),
                        help='Tokenizer模型')
    parser.add_argument('--src-lang', type=str, default='<2en>',
                        help='源语言标记')
    parser.add_argument('--input', type=str, default='',
                        help='输入文本')
    parser.add_argument('--input-file', type=str, default='',
                        help='输入文件')
    parser.add_argument('--output-file', type=str, default='',
                        help='输出文件')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🌍 多语言翻译推理系统")
    print("="*80 + "\n")
    
    # 初始化翻译器
    try:
        translator = Translator(
            model_path=args.model_path,
            sp_model_path=args.sp_model
        )
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        sys.exit(1)
    
    # 推理
    if args.input_file:
        # 从文件读取
        print(f"📖 读取输入文件: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        translations = []
        for i, line in enumerate(lines):
            text = line.strip()
            if not text:
                continue
            
            print(f"[{i+1}/{len(lines)}] 翻译中...")
            translation = translator.translate(text, args.src_lang)
            translations.append(translation)
        
        if args.output_file:
            print(f"\n💾 保存结果: {args.output_file}")
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for trans in translations:
                    f.write(trans + '\n')
    
    elif args.input:
        # 单句翻译
        print(f"源文本: {args.input}")
        translation = translator.translate(args.input, args.src_lang)
        print(f"翻译:   {translation}")
    
    else:
        # 交互模式
        print("💬 交互翻译模式")
        print(f"源语言标记: {args.src_lang}")
        print("输入 'exit' 退出\n")
        
        while True:
            try:
                text = input("源文本> ").strip()
                if text.lower() == 'exit':
                    break
                if not text:
                    continue
                
                translation = translator.translate(text, args.src_lang)
                print(f"翻译>   {translation}\n")
            
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}\n")


if __name__ == "__main__":
    main()