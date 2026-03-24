from typing import List, Literal, Optional, Union, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
)


class FlangService:
    """
    Thin service wrapper around SALT-NLP/FLANG-BERT.

    Features:
      - load model & tokenizer
      - encode texts to embeddings (CLS or mean pooling)
      - masked-token prediction for <mask>
    """

    def __init__(
        self,
        model_name: str = "SALT-NLP/FLANG-BERT",
        device: Optional[str] = None,
        use_mlm_head: bool = True,
    ):
        """
        Args:
            model_name: HF model id.
            device: "cuda", "cpu", "mps", or None (auto-detect).
            use_mlm_head: if True, also load a masked-LM head for <mask> filling.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Base encoder (for embeddings)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Optional MLM head for <mask> prediction
        self.mlm_model = (
            AutoModelForMaskedLM.from_pretrained(model_name) if use_mlm_head else None
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.encoder.to(self.device)
        if self.mlm_model is not None:
            self.mlm_model.to(self.device)

        self.encoder.eval()
        if self.mlm_model is not None:
            self.mlm_model.eval()

    @torch.inference_mode()
    def encode(
        self,
        texts: Union[str, List[str]],
        pooling: Literal["cls", "mean"] = "cls",
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        Encode one or more texts into dense embeddings.

        Args:
            texts: string or list of strings.
            pooling: "cls" = use [CLS] token, "mean" = mean pool non-pad tokens.
            max_length: transformer truncation length.

        Returns:
            Tensor of shape (batch_size, hidden_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.encoder(**enc)
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        if pooling == "cls":
            # ELECTRA-style models put [CLS] at position 0
            emb = last_hidden[:, 0]
        elif pooling == "mean":
            # mean over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb = summed / counts
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return emb  # (B, H)

    @torch.inference_mode()
    def fill_mask(
        self,
        text_with_mask: str,
        top_k: int = 5,
        max_length: int = 128,
    ):
        """
        Predict top-k tokens for a single <mask> in the input text.

        Example:
            service.fill_mask("Stocks rallied and the British pound <mask>.")

        Returns:
            List of dicts: [{"token": "...", "score": float, "token_id": int}, ...]
        """
        if self.mlm_model is None:
            raise RuntimeError("Masked-LM head not loaded (use_mlm_head=False).")

        # HF uses <mask> token; tokenizer will map this appropriately
        enc = self.tokenizer(
            text_with_mask,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("Tokenizer has no mask_token_id.")

        mask_positions = (enc["input_ids"] == mask_token_id).nonzero(as_tuple=False)
        if mask_positions.size(0) != 1:
            raise ValueError("Expected exactly one <mask> token in the input.")

        outputs = self.mlm_model(**enc)
        logits = outputs.logits  # (B, T, V)

        b_idx, pos = mask_positions[0].tolist()
        mask_logits = logits[b_idx, pos]

        probs = torch.softmax(mask_logits, dim=-1)
        scores, token_ids = torch.topk(probs, k=top_k)

        results = []
        for s, t_id in zip(scores.tolist(), token_ids.tolist()):
            tok = self.tokenizer.decode([t_id]).strip()
            results.append({"token": tok, "score": float(s), "token_id": int(t_id)})
        return results

    @torch.no_grad()
    def encode_document(self, text: str) -> Dict[str, Any]:
        """
        Run FinBERT on a long document.

        Returns a dict with:
          - "input_ids": LongTensor [T]  (deduplicated, original order)
          - "attention_mask": LongTensor [T]
          - "embeddings": FloatTensor [T, H]  (averaged over overlaps)
          - "tokens": List[str]  (word-piece tokens)
          - "n_contrib": LongTensor [T]  (# of chunk encodes per token)
        """

        # Tokenize once without special tokens
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        full_ids = enc["input_ids"][0]  # [N]
        full_mask = enc["attention_mask"][0]  # [N]
        N = full_ids.size(0)

        if N == 0:
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "embeddings": torch.empty(0, self.model.config.hidden_size),
                "tokens": [],
                "n_contrib": torch.empty(0, dtype=torch.long),
            }

        max_tokens = self.max_seq_len - 2  # reserve for [CLS],[SEP]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        # Storage for accumulation
        H = self.model.config.hidden_size
        sum_emb = torch.zeros(N, H, dtype=torch.float32)
        count = torch.zeros(N, dtype=torch.long)

        start = 0
        while start < N:
            end = min(start + max_tokens, N)

            # Slice original token range for this chunk
            chunk_ids = full_ids[start:end]
            chunk_mask = full_mask[start:end]

            # Add specials
            chunk_ids = torch.cat(
                [torch.tensor([cls_id]), chunk_ids, torch.tensor([sep_id])]
            )
            chunk_mask = torch.cat(
                [torch.tensor([1]), chunk_mask, torch.tensor([1])]
            )

            # Map chunk positions (excluding specials) back to global indices
            # global_idx[k] = original token index in [0, N)
            global_idx = torch.arange(start, end, dtype=torch.long)

            # Move to device
            chunk_ids = chunk_ids.unsqueeze(0).to(self.device)  # [1, L]
            chunk_mask = chunk_mask.unsqueeze(0).to(self.device)  # [1, L]

            outputs = self.model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
            )
            hidden = outputs.last_hidden_state.squeeze(0)  # [L, H]

            # Drop [CLS] and [SEP]
            hidden_tokens = hidden[1:-1]  # [L-2, H]

            # Accumulate embeddings for each global index
            # Move to CPU for accumulation
            hidden_tokens = hidden_tokens.cpu()
            for local_pos, g_idx in enumerate(global_idx):
                sum_emb[g_idx] += hidden_tokens[local_pos]
                count[g_idx] += 1

            if end == N:
                break
            # Overlap
            start = end - self.stride

        # Avoid division by zero (shouldn't happen if N>0)
        count_clamped = count.clone()
        count_clamped[count_clamped == 0] = 1

        avg_emb = sum_emb / count_clamped.unsqueeze(-1)

        tokens = self.tokenizer.convert_ids_to_tokens(full_ids.tolist())

        return {
            "input_ids": full_ids,  # [N]
            "attention_mask": full_mask,  # [N]
            "embeddings": avg_emb,  # [N, H]
            "tokens": tokens,  # len N
            "n_contrib": count,  # [N]
        }
    @torch.no_grad()
    def encode_news_articles(self, articles: List[str]) -> torch.Tensor:
        """
        Encode a list of news articles into embeddings.

        Args:
            articles: List of article texts.

        Returns:
            Tensor of shape (len(articles), hidden_dim).
        """
        return self.encode(articles, pooling="cls")
    

def test_flang_service():
    service = FlangService(use_mlm_head=True)

    from src.external_api.news import FinnhubNewsCollector
    import random
    collector = FinnhubNewsCollector()
    news = collector.collect(
        entities=["AAPL"],
        start_date="2026-03-05",
        end_date="2026-03-06",
        fields=["Date", "Article"]
    )
    articles = [r["Article"] for r in news]
    if not articles:
        print("No articles found for testing.")
        return
    embeddings = service.encode_news_articles(articles)
    print("Embeddings shape:", embeddings.shape)

    # Test masked token prediction
    article= articles[0] if articles else "Stocks rallied and the British pound"
    aritcle_words = article.split()
    mask_idx = random.randint(0, len(aritcle_words)-1)
    aritcle_words[mask_idx] = service.tokenizer.mask_token
    masked_text = " ".join(aritcle_words)
    print("Original text:", article)
    print("Masked text:", masked_text)
    predictions = service.fill_mask(masked_text, top_k=3)
    predictions = [p["token"] for p in predictions]
    print("Top predictions for <mask>:", predictions)

if __name__ == "__main__":
        test_flang_service()