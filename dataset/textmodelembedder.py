from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class TextModelEmbedder(ABC):
    """
    抽象基类：文本（或图文）模型的特征提取器
    
    子类需要实现：
    1. _load_model_and_preprocess()   # 加载模型 & 预处理函数
    2. _extract_features()             # 核心：把处理好的输入转为 embedding
    
    统一对外接口：
    .embed(texts, normalize=True) → torch.Tensor [batch, dim]
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        mrl_truncate: int,        # MRL 前缀截断长度
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        normalize: bool = True,           # 默认输出 L2 归一化的向量
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float32
        self.normalize = normalize
        self.mrl_truncate = mrl_truncate
        self.model = None
        self.preprocess = None          # 文本 → model 输入的预处理函数
        
        self._load_model_and_preprocess()
        self._move_to_device()

    @abstractmethod
    def _load_model_and_preprocess(self):
        """
        子类实现：加载模型和预处理逻辑
        必须设置：
            self.model
            self.preprocess  (callable: List[str] → Any)
        """
        raise NotImplementedError

    def _move_to_device(self):
        if self.model is not None:
            if hasattr(self.model, "model"):  # 适配 Qwen3-VL-Embedding 的包装结构
                self.model.model.to(device=self.device, dtype=self.torch_dtype)
                self.model.model.eval()
            else:
                self.model.to(device=self.device, dtype=self.torch_dtype)
                self.model.eval()

    @abstractmethod
    def _extract_features(self, processed_input: Any, tokenlevel: bool = False) -> torch.Tensor:
        """
        子类实现：从预处理后的输入中提取 embedding
        返回： [batch, dim] 的特征张量（未归一化）
        """
        raise NotImplementedError

    @torch.inference_mode()
    def embedtext(
        self,
        texts: Union[str, List[str]],
        normalize: Optional[bool] = None,
        batch_size: int = 512,
        tokenlevel: bool = False
    ) -> torch.Tensor:
        """
        统一入口：输入文本 → 输出 embedding
        
        Args:
            texts: 单条 str 或 List[str]
            normalize: 是否 L2 归一化（覆盖初始化时的设置）
            batch_size: 当输入很长列表时分批处理，避免 OOM
        
        Returns:
            torch.Tensor: [n, dim]，默认已归一化
        """
        if isinstance(texts, str):
            texts = [texts]
        
        normalize = normalize if normalize is not None else self.normalize
        
        all_embeddings = []
        
        if len(texts) <= batch_size:
            # 小批量直接处理，避免不必要的循环和 tqdm
            processed = self.preprocess(texts)
            emb, mask = self._extract_features(processed, tokenlevel=tokenlevel)
            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)
            return emb, mask
        else:
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch_texts = texts[i:i + batch_size]
                
                # 预处理（tokenize / prompt wrap / etc）
                processed = self.preprocess(batch_texts)
                
                # 提取特征
                emb, mask = self._extract_features(processed, tokenlevel=tokenlevel)
                
                # 归一化（可选）
                if normalize:
                    emb = F.normalize(emb, p=2, dim=-1)
                    
                all_embeddings.append(emb)
            
            return torch.cat(all_embeddings, dim=0), mask

    

class Qwen3VLEmbeddingTextEmbedder(TextModelEmbedder):
    """Qwen3-VL-Embedding 的纯文本提取器示例"""
    
    def _load_model_and_preprocess(self):
        from third_party.Qwen3_VL_Embedding.src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        
        self.model = Qwen3VLEmbedder(
            model_name_or_path=self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device=self.device,
            attn_implementation="flash_attention_2",  # 可选
        )
        # 预处理：这里简单返回文本列表，实际可能需要加 instruction prefix
        self.preprocess = lambda texts: [
            {"text": t} for t in texts
        ]

    def _extract_features(self, processed_input: List[dict], tokenlevel: bool = False) -> torch.Tensor:
        full_emb, mask = self.model.process(processed_input, tokenlevel=tokenlevel)          # [batch, 2048]
        
        #  MRL 前缀截断
        if self.mrl_truncate > 0 and full_emb.shape[-1] > self.mrl_truncate:
            full_emb = full_emb[..., :self.mrl_truncate]

        return full_emb, mask

class CLIPTextEmbedder(TextModelEmbedder):
    def _load_model_and_preprocess(self):
        import clip
        self.model = clip.load(self.model_name_or_path, device=self.device)[0]
        #CLIP 不支持 MRL 前缀截断，但为了兼容接口，设置 mrl_truncate 为 embed_dim（即不截断）
        self.mrl_truncate = self.model.text_projection.shape[1]
        # 设置 preprocess：文本列表 → token 张量
        self.preprocess = lambda texts: clip.tokenize(texts, truncate=True)

    def _extract_features(self, processed_input: torch.Tensor) -> torch.Tensor:
        # processed_input 已经是 tokenize 后的结果 [batch, context_length]
        tokenlist = processed_input.to(self.device)
        with torch.no_grad():
            # CLIP 的 encode_text 在 bfloat16 下可能有问题，使用 float32 进行编码
            if self.model.dtype != torch.float32:
                txt_feats = self.model.float().encode_text(tokenlist)
            else:
                txt_feats = self.model.encode_text(tokenlist)
            txt_feats = txt_feats.to(self.torch_dtype)
        
        # MRL 前缀截断
        if self.mrl_truncate > 0 and txt_feats.shape[1] > self.mrl_truncate:
            txt_feats = txt_feats[:, :self.mrl_truncate]
            
        return txt_feats
    
class CLIPTextMTEmbedder(nn.Module):
    """
    CLIP 文本特征提取器 - 支持嵌套列表输入 (List[List[str]])
    
    支持两种模式：
    1. EOT 模式：返回 [B, P, D] 的句子级别特征（每个phrase取EOT token）
    2. Token 模式：返回 [B, P, L, D] 的 token 级别特征（SOT & PAD 置零）
    
    其中:
        B = batch size (样本数)
        P = num_phrases (每个样本的caption/phrase数量)
        L = sequence length (token数量，统一padding)
        D = embed_dim
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float32,
        normalize: bool = True,
    ):
        super().__init__()
        import clip
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.normalize = normalize
        
        # 加载 CLIP 模型
        self.clip_model, _ = clip.load(model_name_or_path, device=self.device)
        self.clip_model.eval().to(self.device, dtype=torch_dtype)
        
        # 获取模型参数
        self.embed_dim = self.clip_model.text_projection.shape[1]
        self.context_length = self.clip_model.context_length
    
    def _flatten_texts(self, texts_list: List[List[str]]) -> tuple:
        """
        将嵌套列表展平
        
        Args:
            texts_list: List[List[str]], 每个元素是一个样本的caption列表
        
        Returns:
            flat_texts: List[str], 展平后的所有文本
            group_sizes: List[int], 每个样本的caption数量
        """
        group_sizes = [len(tlist) for tlist in texts_list]
        flat_texts = [s for sub in texts_list for s in sub]
        return flat_texts, group_sizes
    
    @torch.no_grad()
    def embedtext(
        self,
        texts_list: List[List[str]],
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        EOT 模式：返回句子级别特征 [B, P, D]
        
        Args:
            texts_list: List[List[str]]
        
        Returns:
            features: [B, P, D] 每个phrase的EOT token特征
        """
        if not texts_list:
            return torch.empty(0, 0, self.embed_dim, device=self.device)
        
        B = len(texts_list)
        P = len(texts_list[0]) if B > 0 else 0
        
        # 展平处理
        flat_texts, group_sizes = self._flatten_texts(texts_list)
        if not flat_texts:
            return torch.empty(B, 0, self.embed_dim, device=self.device)
        
        # Tokenize
        import clip
        tokens = clip.tokenize(flat_texts, truncate=True).to(self.device)
        
        # 提取特征
        x = self.clip_model.token_embedding(tokens).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        x = x @ self.clip_model.text_projection  # [N, L, D]
        
        # 取 EOT token (每个序列的最后一个有效token)
        eot_indices = tokens.argmax(dim=-1)  # EOT位置
        N = x.shape[0]
        eot_features = x[torch.arange(N), eot_indices]  # [N, D]
        
        # 归一化
        normalize = normalize if normalize is not None else self.normalize
        if normalize:
            eot_features = F.normalize(eot_features, p=2, dim=-1)
        
        # reshape 回 [B, P, D]
        # 检查每个样本的phrase数是否一致
        if len(set(group_sizes)) == 1:
            # 所有样本phrase数相同
            eot_features = eot_features.view(B, P, -1)
        else:
            # phrase数不一致，需要padding（实际场景中应保证一致）
            max_p = max(group_sizes) if group_sizes else 0
            padded = torch.zeros(B, max_p, self.embed_dim, 
                               device=self.device, dtype=eot_features.dtype)
            idx = 0
            for i, size in enumerate(group_sizes):
                padded[i, :size] = eot_features[idx:idx+size]
                idx += size
            eot_features = padded
        
        return eot_features
    
    @torch.no_grad()
    def embedtext_tokenlevel(
        self,
        texts_list: List[List[str]],
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Token 模式：返回 token 级别特征 [B, P, L, D]
        SOT (index 0) 和 PAD (token == 0) 已置零
        
        Args:
            texts_list: List[List[str]]
        
        Returns:
            features: [B, P, L, D] 
            其中 L = context_length (77 for CLIP)
        """
        if not texts_list:
            return torch.empty(0, 0, self.context_length, self.embed_dim, 
                             device=self.device)
        
        B = len(texts_list)
        P = len(texts_list[0]) if B > 0 else 0
        
        # 展平处理
        flat_texts, group_sizes = self._flatten_texts(texts_list)
        if not flat_texts:
            return torch.empty(B, 0, self.context_length, self.embed_dim,
                             device=self.device)
        
        # Tokenize
        import clip
        tokens = clip.tokenize(flat_texts, truncate=True).to(self.device)  # [N, L]
        N, L = tokens.shape
        
        # 提取特征
        x = self.clip_model.token_embedding(tokens).type(self.clip_model.dtype)
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        x = x @ self.clip_model.text_projection  # [N, L, D]
        
        # 构建 mask: True 表示需要mask的位置 (SOT + PAD)
        mask = tokens == 0  # [N, L] PAD tokens
        mask[:, 0] = True   # SOT token (index 0)
        
        # 置零
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # 归一化 (每个token单独归一化)
        normalize = normalize if normalize is not None else self.normalize
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        # reshape 回 [B, P, L, D]
        if len(set(group_sizes)) == 1:
            x = x.view(B, P, L, -1)
        else:
            # phrase数不一致，需要padding
            max_p = max(group_sizes) if group_sizes else 0
            padded = torch.zeros(B, max_p, L, self.embed_dim,
                               device=self.device, dtype=x.dtype)
            idx = 0
            for i, size in enumerate(group_sizes):
                padded[i, :size] = x[idx:idx+size]
                idx += size
            x = padded
        
        return x