
import sys
import os
import pickle
import numpy as np
import json
import warnings
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
import gc
from datetime import datetime
from pathlib import Path
import platform
import traceback

# Try to import FAISS
try:
    import faiss
except ImportError:
    print("FAISS not found. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
    sys.exit(1)


# =============================================================================
# AUTOMATION CONTROL
# =============================================================================

# split1/upper~ split4/lower
SPLIT = "split1/upper"

TARGET_FOLDER = "drive/MyDrive/sffd-evaluations" + "/" + SPLIT
# Model folder name, e.g., "Alibaba-NLP_gte-base-en-v1.5"
EXCLUDED_MODELS = [
    # "Alibaba-NLP_gte-base-en-v1.5"
]


# Device Checker

# Optional imports - gracefully handle if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


def probe_system_info():
    """Probe system information."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        return {
            "mem": {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent,
            },
            "disk": {
                "total": disk.total,
                "available": disk.free,
            },
        }
    return {}


def probe_gpu_info():
    """Probe GPU information."""

    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot probe GPU info")
        return {}

    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu = {
                "name": props.name,
                "total_memory": f"{props.total_memory / (1024**3):.2f} GB",
                "reserved_memory": f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB",
                "allocated_memory": f"{torch.cuda.memory_allocated(i) / (1024**3):.2f} GB",
                "compute_cap": f"{props.major}.{props.minor}",
                "multi_procs": f"{props.multi_processor_count}",
            }
            gpus.append(gpu)

        return {
            "torch": {
                "version": str(torch.__version__),
                "avilable": torch.cuda.is_available(),
                "current_device": f"{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
            },
            "cuda": {
                "version-cuda": str(torch.version.cuda),
                "version-cudnn": str(torch.backends.cudnn.version()),
                "gpus": gpus
            }
        }

    return {}


def probe_python_packages():
    """Probe installed Python packages."""
    packages_to_check = [
        ("torch", TORCH_AVAILABLE),
        ("transformers", TRANSFORMERS_AVAILABLE),
        ("accelerate", ACCELERATE_AVAILABLE),
        ("psutil", PSUTIL_AVAILABLE),
    ]

    # Add more common packages
    try:
        import huggingface_hub
        packages_to_check.append(("huggingface_hub", True))
    except ImportError:
        packages_to_check.append(("huggingface_hub", False))

    try:
        import sentence_transformers
        packages_to_check.append(("sentence_transformers", True))
    except ImportError:
        packages_to_check.append(("sentence_transformers", False))

    try:
        import numpy
        packages_to_check.append(("numpy", True))
    except ImportError:
        packages_to_check.append(("numpy", False))

    try:
        import pandas
        packages_to_check.append(("pandas", True))
    except ImportError:
        packages_to_check.append(("pandas", False))

    return packages_to_check


def probe_environment_variables():
    """Probe relevant environment variables."""

    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TRANSFORMERS_CACHE",
        "HF_HOME",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]

    res = {}
    for var in env_vars:
        value = os.environ.get(var, "(not set)")
        res[var] = value

    return res


def generate_device_info_dict():
    """Generate a dictionary containing device information for saving."""

    return {
        "sys_info": probe_system_info(),
        "gpu_info": probe_gpu_info(),
        "pkg_info": probe_python_packages(),
        "env_info": probe_environment_variables(),
    }


# =============================================================================
# 1. CONFIGURATION CLASS
# =============================================================================

@dataclass
class Config:
    """Static configuration for unified FAISS index factory."""

    # Dataset configuration
    DATASETS: List[str] = field(default_factory=lambda: [
                                "nqa", "mteb_LEMBQMSumRetrieval", "mteb_LEMBSummScreenFDRetrieval"])

    # Mode selection: "E5-SYNTH", "E5-AVERAGE", or "NATIVE-ENCODER"
    MODE: str = "NATIVE-ENCODER"
    # Chunking configuration
    CHUNK_N_LIST: List[int] = field(default_factory=lambda: [
                                    1, 2, 4, 8, 16])
    TOKEN_CHUNK_SIZE: int = 500

    # Output configuration
    OUTPUT_DIR: str = "not_set"

    # Inference settings
    ENCODE_BATCH_SIZE: int = 1  # Default for native (one query at a time)
    # Batch size for e5 chunk encoding within a single query
    E5_MODEL_ID: str = "intfloat/multilingual-e5-base"
    E5_CHUNK_BATCH_SIZE: int = 1024
    USE_FP16: bool = True
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset-specific column mappings
    DATASET_CONFIG: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "nqa": {
            "name": "deepmind/narrativeqa",
            "type": "narrativeqa",
            "doc_col": "document.text",
            "abs_col": "document.summary.text",
            "split": "test"
        },
        "mteb_LEMBQMSumRetrieval": {
            "name": "mteb_LEMBQMSumRetrieval",
            "type": "mteb",
            "split": "test"
        },
        "mteb_LEMBSummScreenFDRetrieval": {
            "name": "mteb_LEMBSummScreenFDRetrieval",
            "type": "mteb",
            "split": "validation"
        }
    })

    # Task-specific instructions for evaluation
    TASK_INSTRUCTIONS: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "nqa": {
            "forward": "Given a web search query, retrieve relevant passages that answer the query",
            "reverse": "Given a web search query, retrieve relevant passages that answer the query"
        },
        "mteb_LEMBQMSumRetrieval": {
            "forward": "Given a web search query, retrieve relevant passages that answer the query",
            "reverse": "Given a web search query, retrieve relevant passages that answer the query"
        },
        "mteb_LEMBSummScreenFDRetrieval": {
            "forward": "Given a web search query, retrieve relevant passages that answer the query",
            "reverse": "Given a web search query, retrieve relevant passages that answer the query"
        }
    })


# =============================================================================
# UTILITY FUNCTIONS (unchanged from original)
# =============================================================================

def get_detailed_instruct(header: str, contents: str) -> str:
    return f"{header}: {contents}"


def get_instructed_query_texts(texts: List[str], instruction: str) -> List[str]:
    return [f"Instruct: {instruction}\nQuery: {text}" for text in texts]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int = 512, max_chunks: int = -1) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for chunk_start_idx in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[chunk_start_idx:chunk_start_idx + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    if max_chunks > 0:
        return chunks[:max_chunks]
    return chunks


@torch.no_grad()
def encode_texts_e5(texts: List[str], model, tokenizer, batch_size: int,
                    instruction: str = "passage", device: str = "cuda") -> torch.Tensor:
    """Generates E5 embeddings for a list of texts in mini-batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        prefixed_texts = [get_detailed_instruct(
            instruction, t) for t in batch_texts]
        inputs = tokenizer(prefixed_texts, max_length=512, padding=True,
                           truncation=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0).to(device)


@torch.no_grad()
def encode_texts_native_single(text: str, tokenizer, encoder, config: Config,
                               model_cfg: Dict[str, Any], prefix: str = "") -> Optional[np.ndarray]:
    """Encode a SINGLE text with native encoder. No batching. Simulates real RAG query arrival.

    Returns:
        numpy array of shape (D,) or None if text exceeds max_ctx.
    """
    prefixed_text = prefix + text
    tokens = tokenizer.encode(prefixed_text, add_special_tokens=False)
    if len(tokens) > model_cfg['max_ctx']:
        return None

    inputs = tokenizer(
        [prefixed_text],
        max_length=model_cfg['max_ctx'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(config.DEVICE)

    with torch.cuda.amp.autocast(enabled=config.USE_FP16):
        outputs = encoder(**inputs)

        if model_cfg['pool_type'] == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embs = (outputs.last_hidden_state * mask).sum(1) / \
                mask.sum(1).clamp(1e-9)
        elif model_cfg['pool_type'] == "cls":
            embs = outputs.last_hidden_state[:, 0]
        elif model_cfg['pool_type'] == "last":
            seqlens = inputs["attention_mask"].sum(1) - 1
            embs = outputs.last_hidden_state[torch.arange(
                len(seqlens)), seqlens]
        else:
            raise ValueError(f"Unknown pool_type: {model_cfg['pool_type']}")

        embs = F.normalize(embs, p=2, dim=-1)

    return embs[0].cpu().float().numpy()


@torch.no_grad()
def encode_texts_native_batch(texts: List[str], tokenizer, encoder, config: Config,
                              model_cfg: Dict[str, Any], prefix: str = "") -> List[Optional[np.ndarray]]:
    """Batch encode texts with native encoder. Used for document indexing (outside timing)."""
    batch_size = config.ENCODE_BATCH_SIZE
    results = []
    valid_indices = []
    valid_texts = []

    for i, text in enumerate(texts):
        prefixed_text = prefix + text
        tokens = tokenizer.encode(prefixed_text, add_special_tokens=False)
        if len(tokens) > model_cfg['max_ctx']:
            results.append(None)
        else:
            valid_indices.append(i)
            valid_texts.append(prefixed_text)
            results.append(None)

    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i+batch_size]
        batch_indices = valid_indices[i:i+batch_size]
        inputs = tokenizer(batch_texts, max_length=model_cfg['max_ctx'], padding=True,
                           truncation=True, return_tensors='pt').to(config.DEVICE)

        with torch.cuda.amp.autocast(enabled=config.USE_FP16):
            outputs = encoder(**inputs)
            if model_cfg['pool_type'] == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                embs = (outputs.last_hidden_state * mask).sum(1) / \
                    mask.sum(1).clamp(1e-9)
            elif model_cfg['pool_type'] == "cls":
                embs = outputs.last_hidden_state[:, 0]
            elif model_cfg['pool_type'] == "last":
                seqlens = inputs["attention_mask"].sum(1) - 1
                embs = outputs.last_hidden_state[torch.arange(
                    len(seqlens)), seqlens]
            else:
                raise ValueError(
                    f"Unknown pool_type: {model_cfg['pool_type']}")
            embs = F.normalize(embs, p=2, dim=-1)

        embs_np = embs.cpu().float().numpy()
        for j, orig_idx in enumerate(batch_indices):
            results[orig_idx] = embs_np[j]

    return results

def aggregate_chunks_e5(chunk_embs: np.ndarray, mode: str,
                        synthesizer=None, device: str = "cuda") -> np.ndarray:
    """Aggregate multiple chunk embeddings into a single document embedding."""
    if chunk_embs.shape[0] == 0:
        d = chunk_embs.shape[1] if len(chunk_embs.shape) > 1 else 768
        return np.zeros(d, dtype='float32')

    chunk_tensor = torch.from_numpy(chunk_embs).to(device)

    if mode == "E5-AVERAGE":
        agg_emb = chunk_tensor.mean(dim=0)
        agg_emb = F.normalize(agg_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
        return agg_emb.cpu().numpy().astype('float32')

    elif mode == "E5-SYNTH":
        if synthesizer is None:
            raise ValueError("E5-SYNTH mode requires synthesizer")

        with torch.no_grad():
            chunk_batch = chunk_tensor.unsqueeze(0)  # (1, num_chunks, D)
            if chunk_batch.shape[1] > 512:
                chunk_batch = chunk_batch[:, :512, :]

            # Explicit attention_mask for clean interface compliance
            model_dtype = next(synthesizer.parameters()).dtype
            attention_mask = torch.ones(chunk_batch.shape[:2], device=device)
            outputs = synthesizer(
                inputs_embeds=chunk_batch.to(model_dtype),
                attention_mask=attention_mask
            )

            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                agg_emb = outputs.pooler_output
            else:
                agg_emb = outputs.last_hidden_state.mean(dim=1)

            agg_emb = F.normalize(agg_emb, p=2, dim=1).squeeze(0)

        return agg_emb.cpu().numpy().astype('float32')

    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def build_faiss_index(embeddings: np.ndarray, output_path: str):
    embeddings = embeddings.astype('float32')
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)
    return index


# =============================================================================
# DATA LOADERS (unchanged)
# =============================================================================

def load_dataset_universal(dataset_name: str, config: Config) -> Dict[str, Any]:
    ds_cfg = config.DATASET_CONFIG[dataset_name]
    dataset_type = ds_cfg.get('type', 'narrativeqa')
    if dataset_type == 'narrativeqa':
        return load_dataset_narrativeqa(dataset_name, ds_cfg)
    elif dataset_type == 'mteb':
        return load_dataset_mteb(dataset_name, ds_cfg)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_dataset_narrativeqa(dataset_name: str, ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset = load_dataset(ds_cfg['name'], split=ds_cfg['split'])
    documents_to_process = []
    seen_doc_ids = set()
    for item in tqdm(dataset, desc="Finding unique items"):
        doc_id = item['document'].get('id')
        if doc_id and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            doc_text = item['document']['text']
            abs_text = item['document']['summary']['text']
            if doc_text and abs_text:
                documents_to_process.append({
                    'doc_id': doc_id, 'doc_text': doc_text,
                    'abs_id': f"summary_{doc_id}", 'abs_text': abs_text
                })
    return {
        'eval_format': 'pairs',
        'documents': documents_to_process,
        'metadata': {'dataset_name': dataset_name, 'type': 'narrativeqa', 'n_documents': len(documents_to_process)}
    }


def load_dataset_mteb(dataset_name: str, ds_cfg: Dict[str, Any]) -> Dict[str, Any]:
    corpus_dataset = load_dataset(ds_cfg['name'].replace(
        '_', '/'), name="corpus", split=ds_cfg['split'])
    queries_dataset = load_dataset(ds_cfg['name'].replace(
        '_', '/'), name="queries", split=ds_cfg['split'])
    qrels_dataset = load_dataset(ds_cfg['name'].replace(
        '_', '/'), name="qrels", split=ds_cfg['split'])

    id_fields = ['_id', 'id', 'doc_id']
    corpus_fields = ['text', 'content',
                     'document', 'passage', 'title', 'abstract']

    corpus = []
    for item in corpus_dataset:
        doc_id = next((str(item[f]) for f in id_fields if f in item), None)
        doc_text = next((str(item[f])
                        for f in corpus_fields if f in item), None)
        if doc_id and doc_text:
            corpus.append({'id': doc_id, 'text': doc_text})

    queries = []
    for item in queries_dataset:
        qid = next((str(item[f]) for f in id_fields if f in item), None)
        qtext = item.get('text', item.get('query', ''))
        if qid and qtext:
            queries.append({'id': qid, 'text': str(qtext)})

    qrels = {}
    for item in qrels_dataset:
        if 'query-id' in item and 'corpus-id' in item:
            qid, did = str(item['query-id']), str(item['corpus-id'])
            qrels.setdefault(qid, {})[did] = item.get('score', 1)

    return {
        'eval_format': 'mteb', 'corpus': corpus, 'queries': queries, 'qrels': qrels,
        'metadata': {'dataset_name': dataset_name, 'type': 'mteb', 'n_corpus': len(corpus), 'n_queries': len(queries)}
    }


# =============================================================================
# EVALUATION METRICS (unchanged)
# =============================================================================

def compute_metrics_from_rankings(gt_ranks: List[int], k: int = 10) -> Dict[str, float]:
    n_queries = len(gt_ranks)
    valid_ranks = [r for r in gt_ranks if r > 0]
    if not valid_ranks:
        return {'ndcg@10': 0.0, 'n_queries': n_queries}
    dcg_sum = sum(1.0 / np.log2(r + 1) for r in valid_ranks if r <= k)
    idcg_sum = n_queries * 1.0
    return {'ndcg@10': float(dcg_sum / idcg_sum if idcg_sum > 0 else 0.0), 'n_queries': n_queries}


def evaluate_direction_pairs(index, query_embeddings, query_times=None, k_evidence=100, k_metric=10):
    query_embeddings = query_embeddings.astype('float32')
    distances, indices = index.search(query_embeddings, k_evidence)
    gt_ranks = []
    evidence_list = []
    for i in range(query_embeddings.shape[0]):
        retrieved_ids = indices[i].tolist()
        try:
            gt_rank = retrieved_ids.index(i) + 1
        except ValueError:
            gt_rank = -1
        gt_ranks.append(gt_rank)
        entry = {
            'query_id': i, 'ground_truth_id': i, 'ground_truth_rank': gt_rank,
            'retrieved_ids': retrieved_ids[:k_metric], 'scores': distances[i].tolist()[:k_metric]
        }
        if query_times is not None:
            entry['query_latency_ms'] = query_times[i]
        evidence_list.append(entry)
    return evidence_list, compute_metrics_from_rankings(gt_ranks, k=k_metric)


def evaluate_direction_mteb(index, query_embeddings, query_ids, corpus_ids, qrels, query_times=None, k_evidence=100, k_metric=10):
    query_embeddings = query_embeddings.astype('float32')
    distances, indices = index.search(query_embeddings, k_evidence)
    gt_ranks = []
    evidence_list = []
    for i in range(query_embeddings.shape[0]):
        query_id = query_ids[i]
        retrieved_corpus_ids = [corpus_ids[idx]
                                for idx in indices[i].tolist() if idx < len(corpus_ids)]
        gt_corpus_ids = set(qrels.get(query_id, {}).keys())
        gt_rank = -1
        for rank, cid in enumerate(retrieved_corpus_ids, 1):
            if cid in gt_corpus_ids:
                gt_rank = rank
                break
        gt_ranks.append(gt_rank)
        entry = {
            'query_id': query_id, 'ground_truth_ids': list(gt_corpus_ids),
            'ground_truth_rank': gt_rank, 'retrieved_corpus_ids': retrieved_corpus_ids[:k_metric],
            'scores': distances[i].tolist()[:k_metric]
        }
        if query_times is not None:
            entry['query_latency_ms'] = query_times[i]
        evidence_list.append(entry)
    return evidence_list, compute_metrics_from_rankings(gt_ranks, k=k_metric)


# =============================================================================
# REPORT I/O (unchanged)
# =============================================================================

def save_evidence(evidence, metadata, output_dir, filename):
    evidence_dir = os.path.join(output_dir, 'evidence')
    os.makedirs(evidence_dir, exist_ok=True)
    output_path = os.path.join(evidence_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'metadata': metadata, 'predictions': evidence},
                  f, indent=2, ensure_ascii=False)
    print(f"      Saved evidence: {output_path}")


def save_report(report_data, output_dir, filename):
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    output_path = os.path.join(reports_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"    Saved report: {output_path}")


# =============================================================================
# MODEL LOADING (unchanged)
# =============================================================================

def load_models(config: Config, model_id: str):
    print(f"[load_models] MODEL_ID={model_id}, DEVICE={config.DEVICE}")
    dtype = torch.float16 if config.USE_FP16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True).to(config.DEVICE)
    encoder.eval()
    return tokenizer, encoder

def load_e5_with_synthesizer(config: Config, model_id: str):
    print(f"[load_models] MODE={config.MODE}, MODEL_ID={model_id}, DEVICE={config.DEVICE}")
    dtype = torch.float16 if config.USE_FP16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.E5_MODEL_ID, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(config.E5_MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(config.DEVICE)
    backbone.eval()
    synth_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    synthesizer = AutoModel.from_pretrained(model_id, torch_dtype=dtype, config=synth_config, trust_remote_code=True).to(config.DEVICE)
    synthesizer.eval()
    return tokenizer, backbone, synthesizer

def load_e5(config:Config):
    print(f"[load_models] MODE={config.MODE}, MODEL_ID=(ignored), DEVICE={config.DEVICE}")
    dtype = torch.float16 if config.USE_FP16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.E5_MODEL_ID, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(config.E5_MODEL_ID, torch_dtype=dtype, trust_remote_code=True).to(config.DEVICE)
    backbone.eval()
    return tokenizer, backbone

# =============================================================================
# REALISTIC RAG SIMULATION - E5 MODE (SYNTH & AVERAGE)
# =============================================================================
# Principle:
#   - Document indexing time is EXCLUDED from measurement
#   - Query encoding time is INCLUDED
#   - Each query is processed independently (queries don't arrive simultaneously)
#   - Within a single query, chunks CAN be batched (they belong to the same query)
#   - Synthesizer is called once per query (serial, since one query at a time)
# =============================================================================


def build_indices_e5_synth_mode(config: Config, model_id: str):
    """Build FAISS indices for E5-based modes with realistic RAG timing."""
    print(f"\n{'='*60}")
    print(f"Realistic RAG Simulation, Model: {model_id}")
    print(f"{'='*60}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    for dataset_name in config.DATASETS:
        print(f"\n{'-'*60}\nDataset: {dataset_name}\n{'-'*60}")

        tokenizer, backbone, synthesizer = load_e5_with_synthesizer(config, model_id)

        # Warm-up
        dummy_chunks = chunk_text_by_tokens(
            "Warm-up dummy text for initialization.", tokenizer, config.TOKEN_CHUNK_SIZE)
        dummy_embs = encode_texts_e5(dummy_chunks, backbone, tokenizer, config.E5_CHUNK_BATCH_SIZE,
                                     instruction="passage", device=config.DEVICE)
        _ = aggregate_chunks_e5(dummy_embs.cpu().numpy(),
                                config.MODE, synthesizer, config.DEVICE)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        data_result = load_dataset_universal(dataset_name, config)
        eval_format = data_result.get('eval_format', 'pairs')

        if eval_format == 'pairs':
            documents = data_result['documents']
            doc_items = [{'id': doc['doc_id'], 'text': doc['doc_text']}
                         for doc in documents]
            query_items = [{'id': doc['abs_id'], 'text': doc['abs_text']}
                           for doc in documents]
        else:
            doc_items = data_result['corpus']
            query_items = data_result['queries']

        # Determine qrels
        if eval_format == 'pairs':
            forward_qrels, reverse_qrels = None, None
        else:
            forward_qrels = data_result['qrels']
            reverse_qrels = {}
            for qid, doc_dict in forward_qrels.items():
                for did in doc_dict.keys():
                    reverse_qrels.setdefault(did, {})[qid] = 1

        mode_suffix = config.MODE.lower().replace("-", "_")
        model_suffix = model_id.replace('/', '_')
        model_path = model_id.replace('/', '_')
        forward_results = {}
        reverse_results = {}

        for chunk_n in config.CHUNK_N_LIST:
            print(f"\n  CHUNK_N={chunk_n}")
            max_len = chunk_n * config.TOKEN_CHUNK_SIZE if chunk_n != -1 else 512 * 100
            chunk_suffix = f"c{chunk_n}" if chunk_n != -1 else "c_all"

            # =================================================================
            # Helper: encode + aggregate one text (for E5 modes)
            # =================================================================
            def encode_and_aggregate_e5(text: str) -> np.ndarray:
                chunks = chunk_text_by_tokens(
                    text, tokenizer, config.TOKEN_CHUNK_SIZE)
                if not chunks:
                    d = backbone.config.hidden_size
                    return np.zeros(d, dtype='float32')
                # Batch encode all chunks within this single query
                chunk_embs = encode_texts_e5(chunks, backbone, tokenizer, config.E5_CHUNK_BATCH_SIZE,
                                             instruction="passage", device=config.DEVICE)
                chunk_embs = F.normalize(chunk_embs, p=2, dim=1)
                # Single synthesizer call (or average)
                return aggregate_chunks_e5(chunk_embs.cpu().numpy(), config.MODE, synthesizer, config.DEVICE)

            # =================================================================
            # FORWARD PASS
            # =================================================================
            print(f"    === FORWARD PASS ===")

            # Phase 1: Truncation (outside timing)
            fwd_trunc_docs = []
            for item in doc_items:
                tokens = tokenizer.encode(
                    item['text'], add_special_tokens=False)[:max_len]
                fwd_trunc_docs.append(tokenizer.decode(
                    tokens, skip_special_tokens=True))

            fwd_trunc_queries = []
            for item in query_items:
                tokens = tokenizer.encode(
                    item['text'], add_special_tokens=False)[:max_len]
                fwd_trunc_queries.append(tokenizer.decode(
                    tokens, skip_special_tokens=True))

            # Phase 2a: Document indexing (OUTSIDE timing - pre-indexed)
            fwd_doc_embs = []
            fwd_valid_doc_ids = []
            for i, text in enumerate(fwd_trunc_docs):
                emb = encode_and_aggregate_e5(text)
                fwd_doc_embs.append(emb)
                fwd_valid_doc_ids.append(doc_items[i]['id'])
            fwd_doc_embs = np.stack(fwd_doc_embs, axis=0)

            doc_index_path = os.path.join(
                config.OUTPUT_DIR, f"{dataset_name}_doc_e5_{model_suffix}_{mode_suffix}_{chunk_suffix}.index")
            fwd_doc_index = build_faiss_index(fwd_doc_embs, doc_index_path)

            # Phase 2b: Query encoding (INSIDE timing - simulating real-time arrival)
            fwd_query_embs = []
            fwd_valid_query_ids = []
            query_times = []

            for i, text in enumerate(tqdm(fwd_trunc_queries, desc=f"      FWD queries (one-by-one)")):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_t = time.monotonic()

                emb = encode_and_aggregate_e5(text)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_t = time.monotonic()

                query_times.append((end_t - start_t) * 1000)
                fwd_query_embs.append(emb)
                fwd_valid_query_ids.append(query_items[i]['id'])

            fwd_query_embs = np.stack(fwd_query_embs, axis=0)
            total_query_time = sum(query_times)

            # Evaluate
            if eval_format == 'pairs':
                fwd_evidence, fwd_metrics = evaluate_direction_pairs(
                    fwd_doc_index, fwd_query_embs, query_times=query_times)
            else:
                fwd_evidence, fwd_metrics = evaluate_direction_mteb(
                    fwd_doc_index, fwd_query_embs, fwd_valid_query_ids, fwd_valid_doc_ids, forward_qrels, query_times=query_times)

            fwd_metrics['elapsed_time'] = total_query_time / 1000.0  # seconds
            fwd_metrics['mean_query_latency_ms'] = np.mean(query_times)
            fwd_metrics['median_query_latency_ms'] = np.median(query_times)
            fwd_metrics['all_query_latencies_ms'] = query_times
            forward_results[chunk_suffix] = fwd_metrics

            save_evidence(fwd_evidence,
                          {'dataset': dataset_name, 'direction': 'forward', 'mode': config.MODE,
                           'model_id': model_id, 'chunk_n': chunk_n},
                          os.path.join(config.OUTPUT_DIR, model_path),
                          f"{dataset_name}_forward_e5_{model_suffix}_{mode_suffix}_{chunk_suffix}.json")

            del fwd_doc_embs, fwd_query_embs, fwd_doc_index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # =================================================================
            # REVERSE PASS
            # =================================================================
            print(f"    === REVERSE PASS ===")

            rev_trunc_docs = fwd_trunc_docs  # Same truncation
            rev_trunc_queries = fwd_trunc_queries

            # Phase 2a: Abstract/query indexing (OUTSIDE timing - pre-indexed as "documents")
            rev_abs_embs = []
            rev_valid_abs_ids = []
            for i, text in enumerate(rev_trunc_queries):
                emb = encode_and_aggregate_e5(text)
                rev_abs_embs.append(emb)
                rev_valid_abs_ids.append(query_items[i]['id'])
            rev_abs_embs = np.stack(rev_abs_embs, axis=0)

            abs_index_path = os.path.join(
                config.OUTPUT_DIR, f"{dataset_name}_abs_e5_{model_suffix}_{mode_suffix}_{chunk_suffix}.index")
            rev_abs_index = build_faiss_index(rev_abs_embs, abs_index_path)

            # Phase 2b: Document-as-query encoding (INSIDE timing)
            rev_doc_query_embs = []
            rev_valid_doc_ids = []
            rev_query_times = []

            for i, text in enumerate(tqdm(rev_trunc_docs, desc=f"      REV queries (one-by-one)")):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_t = time.monotonic()

                emb = encode_and_aggregate_e5(text)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_t = time.monotonic()

                rev_query_times.append((end_t - start_t) * 1000)
                rev_doc_query_embs.append(emb)
                rev_valid_doc_ids.append(doc_items[i]['id'])

            rev_doc_query_embs = np.stack(rev_doc_query_embs, axis=0)
            rev_total_query_time = sum(rev_query_times)

            # Evaluate
            if eval_format == 'pairs':
                rev_evidence, rev_metrics = evaluate_direction_pairs(
                    rev_abs_index, rev_doc_query_embs, query_times=rev_query_times)
            else:
                rev_evidence, rev_metrics = evaluate_direction_mteb(
                    rev_abs_index, rev_doc_query_embs, rev_valid_doc_ids, rev_valid_abs_ids, reverse_qrels, query_times=rev_query_times)

            rev_metrics['elapsed_time'] = rev_total_query_time / 1000.0
            rev_metrics['mean_query_latency_ms'] = np.mean(rev_query_times)
            rev_metrics['median_query_latency_ms'] = np.median(rev_query_times)
            rev_metrics['all_query_latencies_ms'] = rev_query_times
            reverse_results[chunk_suffix] = rev_metrics

            save_evidence(rev_evidence,
                          {'dataset': dataset_name, 'direction': 'reverse', 'mode': config.MODE,
                           'model_id': model_id, 'chunk_n': chunk_n},
                          os.path.join(config.OUTPUT_DIR, model_path),
                          f"{dataset_name}_reverse_e5_{model_suffix}_{mode_suffix}_{chunk_suffix}.json")

            del rev_abs_embs, rev_doc_query_embs, rev_abs_index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save reports
        save_report({'dataset': dataset_name, 'direction': 'forward', 'mode': config.MODE,
                     'model_id': model_id, 'results': forward_results},
                    os.path.join(config.OUTPUT_DIR, model_path), f"{dataset_name}_forward_report.json")
        save_report({'dataset': dataset_name, 'direction': 'reverse', 'mode': config.MODE,
                     'model_id': model_id, 'results': reverse_results},
                    os.path.join(config.OUTPUT_DIR, model_path), f"{dataset_name}_reverse_report.json")

    if synthesizer is not None:
        del synthesizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}\nE5 Mode Realistic RAG Simulation Complete!\n{'='*60}")


def build_indices_e5_average_mode(config: Config):
    """Build FAISS indices for E5-based modes with realistic RAG timing."""
    print(f"\n{'='*60}")
    print(f"Realistic RAG Simulation, Model: (ignored)")
    print(f"{'='*60}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    for dataset_name in config.DATASETS:
        print(f"\n{'-'*60}\nDataset: {dataset_name}\n{'-'*60}")

        tokenizer, backbone = load_e5(config)
        synthesizer = None
        model_id = 'average'

        # Warm-up
        dummy_chunks = chunk_text_by_tokens(
            "Warm-up dummy text for initialization.", tokenizer, config.TOKEN_CHUNK_SIZE)
        dummy_embs = encode_texts_e5(dummy_chunks, backbone, tokenizer, config.E5_CHUNK_BATCH_SIZE,
                                     instruction="passage", device=config.DEVICE)
        _ = aggregate_chunks_e5(dummy_embs.cpu().numpy(),
                                config.MODE, synthesizer, config.DEVICE)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        data_result = load_dataset_universal(dataset_name, config)
        eval_format = data_result.get('eval_format', 'pairs')

        if eval_format == 'pairs':
            documents = data_result['documents']
            doc_items = [{'id': doc['doc_id'], 'text': doc['doc_text']}
                         for doc in documents]
            query_items = [{'id': doc['abs_id'], 'text': doc['abs_text']}
                           for doc in documents]
        else:
            doc_items = data_result['corpus']
            query_items = data_result['queries']

        # Determine qrels
        if eval_format == 'pairs':
            forward_qrels, reverse_qrels = None, None
        else:
            forward_qrels = data_result['qrels']
            reverse_qrels = {}
            for qid, doc_dict in forward_qrels.items():
                for did in doc_dict.keys():
                    reverse_qrels.setdefault(did, {})[qid] = 1

        mode_suffix = config.MODE.lower().replace("-", "_")
        model_suffix = model_id.replace('/', '_')
        model_path = model_id.replace('/', '_')
        forward_results = {}
        reverse_results = {}

        for chunk_n in config.CHUNK_N_LIST:
            print(f"\n  CHUNK_N={chunk_n}")
            max_len = chunk_n * config.TOKEN_CHUNK_SIZE if chunk_n != -1 else 512 * 100
            chunk_suffix = f"c{chunk_n}" if chunk_n != -1 else "c_all"

            # =================================================================
            # Helper: encode + aggregate one text (for E5 modes)
            # =================================================================
            def encode_and_aggregate_e5(text: str) -> np.ndarray:
                chunks = chunk_text_by_tokens(
                    text, tokenizer, config.TOKEN_CHUNK_SIZE)
                if not chunks:
                    d = backbone.config.hidden_size
                    return np.zeros(d, dtype='float32')
                # Batch encode all chunks within this single query
                chunk_embs = encode_texts_e5(chunks, backbone, tokenizer, config.E5_CHUNK_BATCH_SIZE,
                                             instruction="passage", device=config.DEVICE)
                chunk_embs = F.normalize(chunk_embs, p=2, dim=1)
                # Single synthesizer call (or average)
                return aggregate_chunks_e5(chunk_embs.cpu().numpy(), config.MODE, synthesizer, config.DEVICE)

            # =================================================================
            # FORWARD PASS
            # =================================================================
            print(f"    === FORWARD PASS ===")

            # Phase 1: Truncation (outside timing)
            fwd_trunc_docs = []
            for item in doc_items:
                tokens = tokenizer.encode(
                    item['text'], add_special_tokens=False)[:max_len]
                fwd_trunc_docs.append(tokenizer.decode(
                    tokens, skip_special_tokens=True))

            fwd_trunc_queries = []
            for item in query_items:
                tokens = tokenizer.encode(
                    item['text'], add_special_tokens=False)[:max_len]
                fwd_trunc_queries.append(tokenizer.decode(
                    tokens, skip_special_tokens=True))

            # Phase 2a: Document indexing (OUTSIDE timing - pre-indexed)
            fwd_doc_embs = []
            fwd_valid_doc_ids = []
            for i, text in enumerate(fwd_trunc_docs):
                emb = encode_and_aggregate_e5(text)
                fwd_doc_embs.append(emb)
                fwd_valid_doc_ids.append(doc_items[i]['id'])
            fwd_doc_embs = np.stack(fwd_doc_embs, axis=0)

            doc_index_path = os.path.join(
                config.OUTPUT_DIR, f"{dataset_name}_doc_e5_{model_suffix}_{mode_suffix}_{chunk_suffix}.index")
            fwd_doc_index = build_faiss_index(fwd_doc_embs, doc_index_path)

            # Phase 2b: Query encoding (INSIDE timing - simulating real-time arrival)
            fwd_query_embs = []
            fwd_valid_query_ids = []
            query_times = []

            for i, text in enumerate(tqdm(fwd_trunc_queries, desc=f"      FWD queries (one-by-one)")):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_t = time.monotonic()

                emb = encode_and_aggregate_e5(text)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_t = time.monotonic()

                query_times.append((end_t - start_t) * 1000)
                fwd_query_embs.append(emb)
                fwd_valid_query_ids.append(query_items[i]['id'])

            fwd_query_embs = np.stack(fwd_query_embs, axis=0)
            total_query_time = sum(query_times)

            # Evaluate
            if eval_format == 'pairs':
                fwd_evidence, fwd_metrics = evaluate_direction_pairs(
                    fwd_doc_index, fwd_query_embs, query_times=query_times)
            else:
                fwd_evidence, fwd_metrics = evaluate_direction_mteb(
                    fwd_doc_index, fwd_query_embs, fwd_valid_query_ids, fwd_valid_doc_ids, forward_qrels, query_times=query_times)

            fwd_metrics['elapsed_time'] = total_query_time / 1000.0  # seconds
            fwd_metrics['mean_query_latency_ms'] = np.mean(query_times)
            fwd_metrics['median_query_latency_ms'] = np.median(query_times)
            fwd_metrics['all_query_latencies_ms'] = query_times
            forward_results[chunk_suffix] = fwd_metrics

            save_evidence(fwd_evidence,
                          {'dataset': dataset_name, 'direction': 'forward', 'mode': config.MODE,
                           'model_id': model_id, 'chunk_n': chunk_n},
                          os.path.join(config.OUTPUT_DIR, model_path),
                          f"{dataset_name}_forward_e5_{model_suffix}_{mode_suffix}_{chunk_suffix}.json")

            del fwd_doc_embs, fwd_query_embs, fwd_doc_index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # =================================================================
            # REVERSE PASS
            # =================================================================
            print(f"    === REVERSE PASS ===")

            rev_trunc_docs = fwd_trunc_docs  # Same truncation
            rev_trunc_queries = fwd_trunc_queries

            # Phase 2a: Abstract/query indexing (OUTSIDE timing - pre-indexed as "documents")
            rev_abs_embs = []
            rev_valid_abs_ids = []
            for i, text in enumerate(rev_trunc_queries):
                emb = encode_and_aggregate_e5(text)
                rev_abs_embs.append(emb)
                rev_valid_abs_ids.append(query_items[i]['id'])
            rev_abs_embs = np.stack(rev_abs_embs, axis=0)

            abs_index_path = os.path.join(
                config.OUTPUT_DIR, f"{dataset_name}_abs_e5_{model_suffix}_{mode_suffix}_{chunk_suffix}.index")
            rev_abs_index = build_faiss_index(rev_abs_embs, abs_index_path)

            # Phase 2b: Document-as-query encoding (INSIDE timing)
            rev_doc_query_embs = []
            rev_valid_doc_ids = []
            rev_query_times = []

            for i, text in enumerate(tqdm(rev_trunc_docs, desc=f"      REV queries (one-by-one)")):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_t = time.monotonic()

                emb = encode_and_aggregate_e5(text)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_t = time.monotonic()

                rev_query_times.append((end_t - start_t) * 1000)
                rev_doc_query_embs.append(emb)
                rev_valid_doc_ids.append(doc_items[i]['id'])

            rev_doc_query_embs = np.stack(rev_doc_query_embs, axis=0)
            rev_total_query_time = sum(rev_query_times)

            # Evaluate
            if eval_format == 'pairs':
                rev_evidence, rev_metrics = evaluate_direction_pairs(
                    rev_abs_index, rev_doc_query_embs, query_times=rev_query_times)
            else:
                rev_evidence, rev_metrics = evaluate_direction_mteb(
                    rev_abs_index, rev_doc_query_embs, rev_valid_doc_ids, rev_valid_abs_ids, reverse_qrels, query_times=rev_query_times)

            rev_metrics['elapsed_time'] = rev_total_query_time / 1000.0
            rev_metrics['mean_query_latency_ms'] = np.mean(rev_query_times)
            rev_metrics['median_query_latency_ms'] = np.median(rev_query_times)
            rev_metrics['all_query_latencies_ms'] = rev_query_times
            reverse_results[chunk_suffix] = rev_metrics

            save_evidence(rev_evidence,
                          {'dataset': dataset_name, 'direction': 'reverse', 'mode': config.MODE,
                           'model_id': model_id, 'chunk_n': chunk_n},
                          os.path.join(config.OUTPUT_DIR, model_path),
                          f"{dataset_name}_reverse_e5_{model_suffix}_{mode_suffix}_{chunk_suffix}.json")

            del rev_abs_embs, rev_doc_query_embs, rev_abs_index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save reports
        save_report({'dataset': dataset_name, 'direction': 'forward', 'mode': config.MODE,
                     'model_id': model_id, 'results': forward_results},
                    os.path.join(config.OUTPUT_DIR, model_path), f"{dataset_name}_forward_report.json")
        save_report({'dataset': dataset_name, 'direction': 'reverse', 'mode': config.MODE,
                     'model_id': model_id, 'results': reverse_results},
                    os.path.join(config.OUTPUT_DIR, model_path), f"{dataset_name}_reverse_report.json")

    if synthesizer is not None:
        del synthesizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}\nE5 Mode Realistic RAG Simulation Complete!\n{'='*60}")


# =============================================================================
# REALISTIC RAG SIMULATION - NATIVE ENCODER MODE
# =============================================================================
# Principle:
#   - Document indexing time is EXCLUDED from measurement
#   - Query encoding time is INCLUDED
#   - Each query is processed ONE AT A TIME (no batching across queries)
#   - Native encoder processes the full concatenated text in a single forward pass
# =============================================================================

def build_indices_native_mode(config: Config, model_cfg: Dict[str, Any]):
    """Build FAISS indices for native encoder with realistic RAG timing."""
    model_id = model_cfg['model_id']
    print(f"\n{'='*60}")
    print(f"Realistic RAG Simulation - Native Encoder: {model_id}")
    print(f"{'='*60}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tokenizer, encoder = load_models(config, model_id)

    # Warm-up
    _ = encode_texts_native_single(
        "Warm-up dummy text.", tokenizer, encoder, config, model_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for dataset_name in config.DATASETS:
        print(f"\n{'-'*60}\nDataset: {dataset_name}\n{'-'*60}")

        data_result = load_dataset_universal(dataset_name, config)
        eval_format = data_result.get('eval_format', 'pairs')

        doc_texts, abs_texts = [], []
        doc_ids, abs_ids = [], []

        if eval_format == 'pairs':
            for doc in data_result['documents']:
                doc_texts.append(doc['doc_text'])
                abs_texts.append(doc['abs_text'])
                doc_ids.append(doc['doc_id'])
                abs_ids.append(doc['abs_id'])
            qrels, reverse_qrels = None, None
        else:
            for doc in data_result['corpus']:
                doc_texts.append(doc['text'])
                doc_ids.append(doc['id'])
            for q in data_result['queries']:
                abs_texts.append(q['text'])
                abs_ids.append(q['id'])
            qrels = data_result['qrels']
            reverse_qrels = {}
            for qid, dd in qrels.items():
                for did in dd.keys():
                    reverse_qrels.setdefault(did, {})[qid] = 1

        model_suffix = model_id.replace('/', '_')
        model_path = model_id.replace('/', '_')
        passage_prefix = model_cfg['passage_prefix']
        forward_results = {}
        reverse_results = {}

        for chunk_n in config.CHUNK_N_LIST:
            print(f"\n  CHUNK_N={chunk_n}")
            max_len = chunk_n * config.TOKEN_CHUNK_SIZE if chunk_n != - \
                1 else model_cfg['max_ctx']
            chunk_suffix = f"c{chunk_n}" if chunk_n != -1 else "c_all"

            if max_len > model_cfg['max_ctx']:
                print(
                    f"    Skipping: {max_len} > max_ctx {model_cfg['max_ctx']}")
                continue

            # Truncation (outside timing)
            trunc_docs = [tokenizer.decode(tokenizer.encode(t, add_special_tokens=False)[
                                           :max_len], skip_special_tokens=True) for t in doc_texts]
            trunc_abs = [tokenizer.decode(tokenizer.encode(t, add_special_tokens=False)[
                                          :max_len], skip_special_tokens=True) for t in abs_texts]

            # =================================================================
            # FORWARD PASS
            # =================================================================
            print(f"    === FORWARD PASS ===")

            # Document indexing (OUTSIDE timing)
            fwd_doc_list = encode_texts_native_batch(
                trunc_docs, tokenizer, encoder, config, model_cfg, prefix=passage_prefix)
            fwd_doc_embs, fwd_valid_doc_ids = [], []
            for i, emb in enumerate(fwd_doc_list):
                if emb is not None:
                    fwd_doc_embs.append(emb)
                    fwd_valid_doc_ids.append(doc_ids[i])

            if not fwd_doc_embs:
                print(f"    WARNING: No valid docs. Skipping.")
                continue

            fwd_doc_embs = np.stack(fwd_doc_embs)
            doc_index_path = os.path.join(
                config.OUTPUT_DIR, f"{dataset_name}_doc_native_{model_suffix}_{chunk_suffix}.index")
            fwd_doc_index = build_faiss_index(fwd_doc_embs, doc_index_path)

            # Query encoding (INSIDE timing - one at a time)
            fwd_query_embs = []
            fwd_valid_query_ids = []
            fwd_query_times = []

            for i, text in enumerate(tqdm(trunc_abs, desc=f"      FWD queries (one-by-one)")):
                if model_cfg.get('instructed_query', False):
                    instruction = config.TASK_INSTRUCTIONS[dataset_name]['forward']
                    text = f"Instruct: {instruction}\nQuery: {text}"
                else:
                    query_prefix = model_cfg.get('query_prefix', "")
                    text = query_prefix + text

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_t = time.monotonic()

                emb = encode_texts_native_single(
                    text, tokenizer, encoder, config, model_cfg)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_t = time.monotonic()

                if emb is not None:
                    fwd_query_times.append((end_t - start_t) * 1000)
                    fwd_query_embs.append(emb)
                    fwd_valid_query_ids.append(abs_ids[i])

            if not fwd_query_embs:
                print(f"    WARNING: No valid queries. Skipping.")
                del fwd_doc_embs, fwd_doc_index
                gc.collect()
                continue

            fwd_query_embs = np.stack(fwd_query_embs)
            fwd_total_time = sum(fwd_query_times)

            if eval_format == 'pairs':
                fwd_evidence, fwd_metrics = evaluate_direction_pairs(
                    fwd_doc_index, fwd_query_embs, query_times=fwd_query_times)
            else:
                fwd_evidence, fwd_metrics = evaluate_direction_mteb(
                    fwd_doc_index, fwd_query_embs, fwd_valid_query_ids, fwd_valid_doc_ids, qrels, query_times=fwd_query_times)

            fwd_metrics['elapsed_time'] = fwd_total_time / 1000.0
            fwd_metrics['mean_query_latency_ms'] = np.mean(fwd_query_times)
            fwd_metrics['median_query_latency_ms'] = np.median(fwd_query_times)
            fwd_metrics['all_query_latencies_ms'] = fwd_query_times
            forward_results[chunk_suffix] = fwd_metrics

            save_evidence(fwd_evidence,
                          {'dataset': dataset_name, 'direction': 'forward', 'mode': 'NATIVE-ENCODER',
                           'model_id': model_id, 'chunk_n': chunk_n},
                          os.path.join(config.OUTPUT_DIR, model_path),
                          f"{dataset_name}_forward_native_{model_suffix}_{chunk_suffix}.json")

            del fwd_doc_embs, fwd_query_embs, fwd_doc_index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # =================================================================
            # REVERSE PASS
            # =================================================================
            print(f"    === REVERSE PASS ===")

            # Abstract indexing (OUTSIDE timing)
            rev_abs_list = encode_texts_native_batch(
                trunc_abs, tokenizer, encoder, config, model_cfg, prefix=passage_prefix)
            rev_abs_embs, rev_valid_abs_ids = [], []
            for i, emb in enumerate(rev_abs_list):
                if emb is not None:
                    rev_abs_embs.append(emb)
                    rev_valid_abs_ids.append(abs_ids[i])

            if not rev_abs_embs:
                print(f"    WARNING: No valid abstracts. Skipping.")
                continue

            rev_abs_embs = np.stack(rev_abs_embs)
            abs_index_path = os.path.join(
                config.OUTPUT_DIR, f"{dataset_name}_abs_native_{model_suffix}_{chunk_suffix}.index")
            rev_abs_index = build_faiss_index(rev_abs_embs, abs_index_path)

            # Document-as-query encoding (INSIDE timing - one at a time)
            rev_doc_embs = []
            rev_valid_doc_ids = []
            rev_query_times = []

            for i, text in enumerate(tqdm(trunc_docs, desc=f"      REV queries (one-by-one)")):
                if model_cfg.get('instructed_query', False):
                    instruction = config.TASK_INSTRUCTIONS[dataset_name]['reverse']
                    text = f"Instruct: {instruction}\nQuery: {text}"
                else:
                    query_prefix = model_cfg.get('query_prefix', "")
                    text = query_prefix + text

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_t = time.monotonic()

                emb = encode_texts_native_single(
                    text, tokenizer, encoder, config, model_cfg)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_t = time.monotonic()

                if emb is not None:
                    rev_query_times.append((end_t - start_t) * 1000)
                    rev_doc_embs.append(emb)
                    rev_valid_doc_ids.append(doc_ids[i])

            if not rev_doc_embs:
                print(f"    WARNING: No valid doc queries. Skipping.")
                del rev_abs_embs, rev_abs_index
                gc.collect()
                continue

            rev_doc_embs = np.stack(rev_doc_embs)
            rev_total_time = sum(rev_query_times)

            if eval_format == 'pairs':
                rev_evidence, rev_metrics = evaluate_direction_pairs(
                    rev_abs_index, rev_doc_embs, query_times=rev_query_times)
            else:
                rev_evidence, rev_metrics = evaluate_direction_mteb(
                    rev_abs_index, rev_doc_embs, rev_valid_doc_ids, rev_valid_abs_ids, reverse_qrels, query_times=rev_query_times)

            rev_metrics['elapsed_time'] = rev_total_time / 1000.0
            rev_metrics['mean_query_latency_ms'] = np.mean(rev_query_times)
            rev_metrics['median_query_latency_ms'] = np.median(rev_query_times)
            rev_metrics['all_query_latencies_ms'] = rev_query_times
            reverse_results[chunk_suffix] = rev_metrics

            save_evidence(rev_evidence,
                          {'dataset': dataset_name, 'direction': 'reverse', 'mode': 'NATIVE-ENCODER',
                           'model_id': model_id, 'chunk_n': chunk_n},
                          os.path.join(config.OUTPUT_DIR, model_path),
                          f"{dataset_name}_reverse_native_{model_suffix}_{chunk_suffix}.json")

            del rev_abs_embs, rev_doc_embs, rev_abs_index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        save_report({'dataset': dataset_name, 'direction': 'forward', 'mode': 'NATIVE-ENCODER',
                     'model_id': model_id, 'results': forward_results},
                    os.path.join(config.OUTPUT_DIR, model_path), f"{dataset_name}_forward_report.json")
        save_report({'dataset': dataset_name, 'direction': 'reverse', 'mode': 'NATIVE-ENCODER',
                     'model_id': model_id, 'results': reverse_results},
                    os.path.join(config.OUTPUT_DIR, model_path), f"{dataset_name}_reverse_report.json")

    del encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n{'='*60}\nNative Encoder Realistic RAG Simulation Complete!\n{'='*60}")

# =============================================================================
# MAIN ENTRY POINT (AUTOMATED)
# =============================================================================


def main():
    base_path = Path(TARGET_FOLDER)
    status_log_path = base_path / "status.log"
    print(f"🔥 Starting stateful SFfD evaluation across: {base_path.resolve()}")
    print(f"   Status Log: {status_log_path}")

    # 1. Read status.log to find completed models
    completed_models = set()
    if status_log_path.exists():
        with open(status_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    completed_models.add(parts[1])

    print(
        f"Found {len(completed_models)} previously processed models in status log.")

    # 2. Find all model configs and filter out completed ones
    all_config_files = sorted(base_path.rglob('config.json'))
    models_to_evaluate = []
    for config_path in all_config_files:
        model_name = config_path.parent.name
        if model_name not in completed_models and model_name not in EXCLUDED_MODELS:
            models_to_evaluate.append(config_path)

    if not models_to_evaluate:
        print("✨ All models have already been evaluated. Nothing to do. ✨")
        return

    print(f"Found {len(models_to_evaluate)} new models to evaluate.")

    # 3. Loop through the remaining models
    progress_bar = tqdm(models_to_evaluate, desc="Evaluation Progress")

    for config_path in progress_bar:
        model_name = config_path.parent.name
        model_folder_name = config_path.parent.name
        progress_bar.set_description(f"🚀 Running SFfD for [{model_name}]")
        output_dir = str(config_path.parent)
        status = "FAIL"  # Default status

        try:
            # Instantiate the main config once
            config = Config()

            # Load model config first to determine mode
            with open(config_path, 'r', encoding='utf-8') as f:
                model_cfg = json.load(f)

            # Determine mode from config (default to NATIVE-ENCODER for backward compatibility)
            detected_mode:str = model_cfg.get("mode", "NATIVE-ENCODER")
            detected_mode = detected_mode.upper()

            if detected_mode == 'NATIVE-ENCODER' or detected_mode == 'E5-SYNTH' or detected_mode == 'E5-AVERAGE':
                config.MODE = detected_mode
            else:
                raise Exception("Mode config must be one of : 'E5-SYNTH', 'E5-AVERAGE', or 'NATIVE-ENCODER'")
            print(f"  MODE: {config.MODE}")
            print(f"  DATASETS: {config.DATASETS}")
            print(f"  DEVICE: {config.DEVICE}")


            if config.MODE == "NATIVE-ENCODER":
                model_id = model_cfg["model_name"]
                pool_type = model_cfg["pool_type"]
                query_prefix = model_cfg["query_prefix"]
                instructed_query = model_cfg["is_instruct"]
                passage_prefix = model_cfg["passage_prefix"]
                max_ctx = model_cfg["max_ctx"]

                # Map keys from lemb config to a sffd‑compatible config
                sffd_model_cfg = {
                    "model_id": model_id,
                    "pool_type": pool_type,
                    "query_prefix": query_prefix,
                    "instructed_query": instructed_query,
                    "passage_prefix": passage_prefix,
                    "max_ctx": max_ctx,
                }
            elif config.MODE == "E5-SYNTH" or config.MODE == "E5-AVERAGE":
                config.E5_MODEL_ID = model_cfg.get('e5_model_name', 'intfloat/multilingual-e5-base')


            # set output path to config path
            config.OUTPUT_DIR = config_path.parent

            start_time = time.monotonic()

            # Run evaluation based on detected mode
            if config.MODE == "E5-SYNTH":
                model_id = model_cfg["model_name"]
                build_indices_e5_synth_mode(config, model_id)
            elif config.MODE == "E5-AVERAGE":
                build_indices_e5_average_mode(config)
            elif config.MODE == "NATIVE-ENCODER":
                build_indices_native_mode(config, sffd_model_cfg)
            else:
                raise ValueError(f"Unknown mode: {config.MODE}. Supported modes: NATIVE-ENCODER, E5-SYNTH, E5-AVERAGE")

            end_time = time.monotonic()

            # Write timing and device info to report.txt
            report_path = os.path.join(output_dir, "report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                report_data = {
                    "evaluation": {"total_wall_clock_elapsed": end_time - start_time},
                    "env": generate_device_info_dict()
                }
                f.write(json.dumps(report_data, indent=4))

            status = "SUCCESS"
            print(f"  ✅ FINISHED: SFfD for [{model_name}]")

        except Exception as e:
            error_log_path = os.path.join(output_dir, 'error.log')
            error_message = f"---\nERROR during SFfD for [{model_name}]\nConfig: {config_path}\nError: {e}\n---"
            print(f"  🚨 FAILED: SFfD for [{model_name}]. See {error_log_path}")
            with open(error_log_path, "w", encoding='utf-8') as f:
                f.write(error_message)
                f.write('\n')
                f.write(traceback.format_exception())

        finally:
            # 4. Log status regardless of outcome
            with open(status_log_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp},{model_name},{status}\n")
            if status == 'FAIL':
                break

    print("\n\n⏸️ SFfD evaluations loop halted! ⏸️")


if __name__ == "__main__":
    main()
