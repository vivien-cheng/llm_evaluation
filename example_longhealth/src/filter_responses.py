import argparse
import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import torch

# --- Path Setup ---
# Add project root for potential future imports if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# from hta_evaluation_harness.utils import save_json, load_json # Using basic file handling here
# --- End Path Setup ---


# --- Semantic Filtering Logic ---

def filter_semantic_responses(
    responses_for_query: list,
    query_text: str,
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    max_responses: int
    ) -> list:
    """
    Filters responses based on semantic similarity using embeddings.
    1. Encodes responses.
    2. Clusters similar responses.
    3. Selects the best representative from each cluster (closest to query).
    4. Ranks representatives by similarity to query.
    5. Returns top N responses.

    Args:
        responses_for_query: List of response dictionaries for a single query.
                             Each dict must contain 'response_text'.
        query_text: The original query text.
        embedding_model: An initialized SentenceTransformer model.
        similarity_threshold: Cosine similarity threshold for clustering (e.g., 0.85).
                              Responses with similarity > threshold might be clustered.
        max_responses: The maximum number of responses to return.

    Returns:
        A list of selected response dictionaries, ranked by similarity to the query.
    """
    if not responses_for_query:
        return []

    response_texts = [r.get("response_text", "") for r in responses_for_query]
    valid_indices = [i for i, text in enumerate(response_texts) if text and not text.startswith("[Generation Error")]
    
    if not valid_indices:
        print("  - Warning: No valid response texts found for this query.")
        return []

    # Filter responses and texts to only include valid ones
    valid_responses = [responses_for_query[i] for i in valid_indices]
    valid_texts = [response_texts[i] for i in valid_indices]

    print(f"  - Encoding {len(valid_texts)} valid responses...")
    try:
        # Encode query and valid responses
        query_embedding = embedding_model.encode(query_text, convert_to_tensor=True)
        response_embeddings = embedding_model.encode(valid_texts, convert_to_tensor=True)

        # --- Clustering for Deduplication ---
        # Calculate cosine similarity matrix
        similarity_matrix = util.pytorch_cos_sim(response_embeddings, response_embeddings).cpu().numpy()

        # Convert similarity to distance for clustering (distance = 1 - similarity)
        distance_matrix = 1 - similarity_matrix
        # Ensure distances are non-negative (floating point issues might make them slightly negative)
        distance_matrix[distance_matrix < 0] = 0

        # Perform Agglomerative Clustering
        # Cluster responses where distance is less than (1 - similarity_threshold)
        # Note: affinity='precomputed' requires a distance matrix, not similarity
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=(1 - similarity_threshold),
            metric='precomputed', # Use precomputed distance matrix
            linkage='average' # Or 'complete', 'single' - average often works well
        ).fit(distance_matrix)

        num_clusters = clustering.n_clusters_
        print(f"  - Found {num_clusters} semantic clusters (threshold={similarity_threshold:.2f}).")

        # --- Select Best Representative from Each Cluster ---
        cluster_representatives = []
        processed_indices = set() # Keep track of indices already added

        for cluster_id in range(num_clusters):
            cluster_member_indices = np.where(clustering.labels_ == cluster_id)[0]
            
            if not cluster_member_indices.size: continue # Skip empty clusters if any

            best_response_in_cluster = None
            highest_query_sim = -1

            # Find the response in this cluster most similar to the query
            cluster_embeddings = response_embeddings[cluster_member_indices]
            query_similarities = util.pytorch_cos_sim(query_embedding, cluster_embeddings).cpu().numpy()[0]

            best_member_index_in_cluster = np.argmax(query_similarities)
            original_index = cluster_member_indices[best_member_index_in_cluster]

            # Ensure we haven't already processed this index via another path (shouldn't happen with clustering)
            if original_index not in processed_indices:
                 best_response_in_cluster = valid_responses[original_index]
                 # Store similarity score for ranking
                 best_response_in_cluster['similarity_to_query'] = float(np.max(query_similarities))
                 cluster_representatives.append(best_response_in_cluster)
                 processed_indices.add(original_index)


        # --- Rank Representatives and Select Top N ---
        # Sort representatives by their similarity to the query (descending)
        cluster_representatives.sort(key=lambda r: r.get('similarity_to_query', -1), reverse=True)

        # Select top N
        selected_responses = cluster_representatives[:max_responses]
        print(f"  - Selected {len(selected_responses)} final responses after clustering and ranking.")

        return selected_responses

    except Exception as e:
        print(f"  - Error during semantic filtering: {e}")
        # Fallback to simple selection if semantic filtering fails
        print("  - Warning: Falling back to simple filtering (first N unique valid responses).")
        unique_valid_responses = []
        seen_texts = set()
        for resp in valid_responses:
            text = resp.get("response_text", "")
            if text not in seen_texts:
                unique_valid_responses.append(resp)
                seen_texts.add(text)
        return unique_valid_responses[:max_responses]


# --- End Semantic Filtering Logic ---


def main():
    parser = argparse.ArgumentParser(description="Filter LLM responses using semantic similarity.")
    parser.add_argument(
        "--model_response_files",
        nargs="+",
        required=True,
        help="List of JSONL files containing model responses."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the final filtered responses JSONL file."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="all-MiniLM-L6-v2", # Use a standard, efficient model
        help="Hugging Face model name for sentence-transformers embeddings."
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.85, # Threshold for considering responses as near-duplicates
        help="Cosine similarity threshold for clustering responses."
    )
    parser.add_argument(
        "--max_responses_per_query",
        type=int,
        default=5,
        help="Maximum number of responses to keep per query after filtering."
    )
    args = parser.parse_args()

    print("--- Starting Semantic Response Filtering ---")
    print(f"Input files: {args.model_response_files}")
    print(f"Output file: {args.output_file}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    print(f"Max Responses per Query: {args.max_responses_per_query}")
    print(f"Embedding Model: {args.embedding_model_name}")

    # --- Load Embedding Model ---
    try:
        print(f"Loading embedding model: {args.embedding_model_name}...")
        # Determine device for sentence transformer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        embedding_model = SentenceTransformer(args.embedding_model_name, device=device)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model '{args.embedding_model_name}': {e}")
        print("Please ensure 'sentence-transformers' is installed and the model name is correct.")
        sys.exit(1)

    # --- Load all responses ---
    all_responses = []
    total_loaded = 0
    print("Loading responses from files...")
    for file_path in args.model_response_files:
        if not os.path.exists(file_path):
            print(f"Warning: Input response file not found: {file_path}. Skipping.")
            continue
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                count_before = len(all_responses)
                for line in f:
                    if line.strip():
                        try:
                            all_responses.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON line in {file_path}: {line[:100]}...")
                count_after = len(all_responses)
                print(f"  - Loaded {count_after - count_before} responses from {os.path.basename(file_path)}")
                total_loaded += (count_after - count_before)
        except IOError as e:
            print(f"Warning: Could not read file {file_path}: {e}. Skipping.")

    if not all_responses:
        print("Error: No responses loaded. Cannot perform filtering.")
        sys.exit(1)
    print(f"Total responses loaded: {total_loaded}")

    # --- Group responses by query_id ---
    responses_by_query = defaultdict(list)
    query_text_map = {} # Store query text for each query_id
    print("Grouping responses by query...")
    for r_data in all_responses:
        query_id = r_data.get("id") # Use the unique question ID from data_utils
        if not query_id: # Fallback if 'id' is missing, try 'query_id'
             query_id = r_data.get("query_id")

        if query_id:
            responses_by_query[query_id].append(r_data)
            if query_id not in query_text_map:
                query_text_map[query_id] = r_data.get("query", "") # Store query text once
        else:
            print(f"Warning: Skipping response with missing 'id' or 'query_id': {r_data.get('response_text', '')[:50]}...")

    print(f"Grouped responses into {len(responses_by_query)} unique queries.")

    # --- Filter responses for each query ---
    final_filtered_responses = []
    print("Filtering responses for each query using semantic similarity...")
    for query_id, resp_list in tqdm(responses_by_query.items(), desc="Filtering queries"):
        if not resp_list: continue
        query_text = query_text_map.get(query_id, "")
        if not query_text:
             print(f"Warning: No query text found for query_id {query_id}. Skipping filtering.")
             continue

        # Apply the semantic filtering logic
        filtered_for_query = filter_semantic_responses(
            resp_list,
            query_text,
            embedding_model,
            args.similarity_threshold,
            args.max_responses_per_query
        )
        final_filtered_responses.extend(filtered_for_query)

    print(f"Total responses after semantic filtering: {len(final_filtered_responses)}")

    # --- Save the filtered responses ---
    try:
        print(f"Saving filtered responses to {args.output_file}...")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding='utf-8') as outfile:
            for entry in final_filtered_responses:
                 # Keep relevant info for evaluation
                 # Remove the temporary similarity score if desired
                 entry.pop('similarity_to_query', None)
                 outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Filtered responses saved successfully.")
        print("--- Semantic Filtering Finished ---")
    except IOError as e:
        print(f"Error writing output file {args.output_file}: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred during saving: {e}")
         sys.exit(1)


if __name__ == "__main__":
    main()
