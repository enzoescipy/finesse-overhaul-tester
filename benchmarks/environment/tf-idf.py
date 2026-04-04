"""
TF-IDF Inter-Chunk Similarity Analysis
Validates the 'semantic atom' assumption via non-circular lexical analysis.
"""

# =============================================================================
# Cell 1: Environment Setup
# =============================================================================

# !pip install -q datasets scikit-learn transformers torch

import numpy as np
import random
import datetime
from typing import List, Tuple, Dict
from collections import deque

from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("   Environment ready. Libraries imported.")

# =============================================================================
# Cell 2: Configuration & Tokenizer
# =============================================================================

# Configuration
CHUNK_SIZE_TOKENS = 500  # Target token count per chunk
POOL_SIZE = 10000        # Total chunks to collect from each corpus
SAMPLE_PAIRS = 1000      # Number of distinct pairs for similarity analysis
MAX_CHARS_PER_CHUNK = 4000  # Safety limit to avoid excessive memory

# Tokenizer (bert-base-uncased for consistent tokenization)
TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"   Tokenizer loaded: {TOKENIZER_NAME}")
print(f"   Chunk size: {CHUNK_SIZE_TOKENS} tokens")
print(f"   Pool size: {POOL_SIZE} chunks")
print(f"   Sample pairs: {SAMPLE_PAIRS}")

# =============================================================================
# Cell 3: Dataset Loading Functions
# =============================================================================

def load_culturax():
    """Load CulturaX English subset by downloading specific files."""
    print("  Loading CulturaX (downloading subset)...")
    dataset = load_dataset(
        "uonlp/CulturaX",
        "en",
        split="train",
        data_files="en/en_part_0000*.parquet",
        revision="6a8734bc69fefcbb7735f4f9250f43e4cd7a442e"
    )
    return dataset

def load_wikipedia():
    """Load Wikipedia English subset by downloading specific files."""
    print("  Loading Wikipedia (downloading subset)...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        data_files="20231101.en/train-000*-of-00041.parquet",
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
    )
    return dataset

# =============================================================================
# Cell 4: Text Chunking Functions
# =============================================================================

def chunk_text_to_tokens(text: str, target_tokens: int = CHUNK_SIZE_TOKENS) -> List[str]:
    """
    Chunk text into segments of approximately target_tokens.
    Uses character-based estimation for efficiency, then verifies with tokenizer.
    
    Strategy:
    - Estimate: ~4 chars per token for English
    - Verify actual token count and adjust
    """
    if not text or len(text.strip()) < 100:
        return []
    
    # Rough character estimate (~4 chars/token)
    chars_per_chunk = target_tokens * 4
    
    # Simple sentence-aware chunking
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_chars = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        # Check if adding this sentence exceeds target
        if current_chars + len(sent) > chars_per_chunk and current_chunk:
            # Finalize current chunk
            chunk_text = '. '.join(current_chunk) + '.'
            if len(chunk_text) <= MAX_CHARS_PER_CHUNK:
                chunks.append(chunk_text)
            current_chunk = [sent]
            current_chars = len(sent)
        else:
            current_chunk.append(sent)
            current_chars += len(sent) + 2  # +2 for '. '
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = '. '.join(current_chunk)
        if not chunk_text.endswith('.'):
            chunk_text += '.'
        if len(chunk_text) <= MAX_CHARS_PER_CHUNK:
            chunks.append(chunk_text)
    
    # Verify token counts and filter
    valid_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        # Accept chunks within 20% of target
        if target_tokens * 0.8 <= len(tokens) <= target_tokens * 1.2:
            valid_chunks.append(chunk)
    
    return valid_chunks

def collect_chunk_pool(dataset, target_pool_size: int = POOL_SIZE) -> List[str]:
    """
    Collect a pool of valid chunks from in-memory dataset.
    Shuffles and iterates through the loaded dataset.
    """
    print(f"   Collecting {target_pool_size} valid chunks...")
    
    pool = []
    
    # Shuffle the in-memory dataset for randomness
    shuffled_dataset = dataset.shuffle(seed=SEED)
    
    for example in tqdm(shuffled_dataset, desc="Processing documents"):
        if len(pool) >= target_pool_size:
            break
        
        # Extract text (handle different column names)
        text = example.get('text') or example.get('content') or example.get('article')
        
        if not text or len(text) < 1000:
            continue
        
        # Generate chunks from this document
        chunks = chunk_text_to_tokens(text)
        pool.extend(chunks)
    
    final_pool = pool[:target_pool_size]
    print(f"   Collected {len(final_pool)} chunks.")
    return final_pool


# =============================================================================
# Cell 5: TF-IDF Similarity Analysis
# =============================================================================

def calculate_tfidf_similarity(chunks: List[str], n_pairs: int = SAMPLE_PAIRS) -> Tuple[float, float, List[float]]:
    """
    Calculate TF-IDF cosine similarities for n_pairs distinct random chunk pairs.
    
    Returns:
        (mean_similarity, std_similarity, all_similarities_list)
    """
    print(f"   Analyzing {n_pairs} random chunk pairs with TF-IDF...")
    
    if len(chunks) < 100:
        raise ValueError(f"Need at least 100 chunks, got {len(chunks)}")
    
    # Initialize TF-IDF vectorizer
    # Using conservative parameters to avoid overfitting to specific terms
    vectorizer = TfidfVectorizer(
        max_features=10000,      # Limit vocabulary size
        min_df=2,              # Ignore terms appearing in <2 documents
        max_df=0.95,           # Ignore terms appearing in >95% of documents
        stop_words='english',  # Remove English stop words
        ngram_range=(1, 2),    # Unigrams and bigrams
        sublinear_tf=True      # Apply sublinear tf scaling (1 + log(tf))
    )
    
    # Fit TF-IDF on the entire pool
    print("   Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(chunks)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Sample distinct pairs
    n_chunks = len(chunks)
    similarities = []
    
    # Generate n_pairs distinct random pairs
    used_pairs = set()
    attempts = 0
    max_attempts = n_pairs * 10
    
    while len(similarities) < n_pairs and attempts < max_attempts:
        i, j = random.sample(range(n_chunks), 2)
        pair_key = tuple(sorted([i, j]))
        
        if pair_key in used_pairs:
            attempts += 1
            continue
        
        used_pairs.add(pair_key)
        
        # Calculate cosine similarity for this pair
        vec_i = tfidf_matrix[i]
        vec_j = tfidf_matrix[j]
        sim = cosine_similarity(vec_i, vec_j)[0, 0]
        similarities.append(sim)
        
        if len(similarities) % 100 == 0:
            print(f"   Processed {len(similarities)}/{n_pairs} pairs...")
    
    similarities = np.array(similarities)
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    return mean_sim, std_sim, similarities.tolist()

# =============================================================================
# Cell 6: Main Execution
# =============================================================================

def run_analysis(dataset_name: str, dataset_loader) -> Dict:
    """Run complete analysis pipeline for one dataset."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    dataset = dataset_loader()
    
    # Collect chunk pool
    chunk_pool = collect_chunk_pool(dataset, POOL_SIZE)
    
    if len(chunk_pool) < SAMPLE_PAIRS * 2:
        print(f"   Warning: Only collected {len(chunk_pool)} chunks, need more for reliable analysis")
    
    # Calculate similarities
    mean_sim, std_sim, all_sims = calculate_tfidf_similarity(chunk_pool, SAMPLE_PAIRS)
    
    results = {
        'dataset': dataset_name,
        'pool_size': len(chunk_pool),
        'n_pairs': len(all_sims),
        'mean_similarity': mean_sim,
        'std_similarity': std_sim,
        'min_similarity': np.min(all_sims),
        'max_similarity': np.max(all_sims),
        'median_similarity': np.median(all_sims),
        'all_similarities': all_sims
    }
    
    return results

def print_results(results: Dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {results['dataset']}")
    print(f"{'='*60}")
    print(f"   Pool size:           {results['pool_size']:,} chunks")
    print(f"   Pairs analyzed:      {results['n_pairs']:,}")
    print(f"   Mean similarity:     {results['mean_similarity']:.4f}")
    print(f"   Std deviation:       {results['std_similarity']:.4f}")
    print(f"   Median similarity:   {results['median_similarity']:.4f}")
    print(f"   Min / Max:           {results['min_similarity']:.4f} / {results['max_similarity']:.4f}")
    print(f"{'='*60}")

# =============================================================================
# Cell 7: Run Both Datasets
# =============================================================================


# Run CulturaX
results_culturax = run_analysis("CulturaX-en", load_culturax)
print_results(results_culturax)

# Run Wikipedia
results_wikipedia = run_analysis("Wikipedia-en", load_wikipedia)
print_results(results_wikipedia)


# =============================================================================
# Cell 8: Generate Report
# =============================================================================

def generate_report(res_cx: Dict, res_wp: Dict, output_path: str = "report.txt"):
    """Generate a plain-text report summarizing the TF-IDF analysis."""
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = []
    lines.append("=" * 70)
    lines.append("TF-IDF Inter-Chunk Similarity Analysis Report")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Purpose: Validate semantic atom assumption")
    lines.append("=" * 70)
    lines.append("")
    
    # Configuration
    lines.append("CONFIGURATION")
    lines.append("-" * 70)
    lines.append(f"  Tokenizer:          {TOKENIZER_NAME}")
    lines.append(f"  Chunk size:         {CHUNK_SIZE_TOKENS} tokens")
    lines.append(f"  Pool size:          {POOL_SIZE} chunks per corpus")
    lines.append(f"  Sample pairs:       {SAMPLE_PAIRS}")
    lines.append(f"  Random seed:        {SEED}")
    lines.append("")
    
    # Results for each dataset
    for res in [res_cx, res_wp]:
        lines.append(f"RESULTS: {res['dataset']}")
        lines.append("-" * 70)
        lines.append(f"  Pool size:          {res['pool_size']:,} chunks")
        lines.append(f"  Pairs analyzed:     {res['n_pairs']:,}")
        lines.append(f"  Mean similarity:    {res['mean_similarity']:.6f}")
        lines.append(f"  Std deviation:      {res['std_similarity']:.6f}")
        lines.append(f"  Median similarity:  {res['median_similarity']:.6f}")
        lines.append(f"  Min similarity:     {res['min_similarity']:.6f}")
        lines.append(f"  Max similarity:     {res['max_similarity']:.6f}")
        lines.append("")
    
    report_text = "\n".join(lines)
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    # Also print to console
    print(report_text)
    print(f"\n[DONE] Report saved to: {output_path}")

generate_report(results_culturax, results_wikipedia)
