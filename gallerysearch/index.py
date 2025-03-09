import json
from collections import defaultdict
from pathlib import Path
import faiss
import numpy as np
import pickle
import platformdirs
from typing import List, Optional, Tuple

from tqdm import tqdm

from gallerysearch.embed import CLIPEmbedder
from gallerysearch.gather import gather_gallery_files, GalleryPair

import hashlib


def stable_hash(input_string: str) -> str:
    encoded_string = input_string.encode('utf-8')
    sha1_hash = hashlib.sha1(encoded_string)
    return sha1_hash.hexdigest()


class TagIndex:
    def __init__(self, directory: str):
        self.gallery_pairs = gather_gallery_files(Path(directory))

        self.tags_by_pair = {}
        self.rating_by_pair = {}
        self.pairs_by_tag = defaultdict(set)
        self.pairs_by_rating = defaultdict(set)

        for pair in tqdm(self.gallery_pairs, desc="Loading tags"):
            with open(pair.json_file) as f:
                data = json.load(f)
                tag_keys = (
                    "tags_general",
                    "gen_tags",
                    "tags"
                )
                tag_key = next(
                    (key for key in tag_keys if key in data),
                    None
                )
                if tag_key:
                    tags = set(data[tag_key])
                    self.tags_by_pair[pair] = tags
                    for tag in tags:
                        self.pairs_by_tag[tag].add(pair)

                if "rating" in data:
                    rating = {
                        "General": "s",
                        "R-18": "e",
                        "R-18G": "e",
                        "g": "s",
                    }.get(data["rating"], data["rating"])
                    self.rating_by_pair[pair] = rating
                    self.pairs_by_rating[rating].add(pair)


    def search(self, text: str) -> List[GalleryPair]:
        query_parts = [
            x.strip() for x in text.lower().split()
        ]
        rating_text = next((x for x in query_parts if x.startswith("rating:")), None)
        tags = [x for x in query_parts if not x.startswith("rating:") and not x.startswith('-')]
        negative_tags = [x[1:] for x in query_parts if x.startswith('-')]
        # we want only the images that match all tags
        output = [
            pair for pair in self.pairs_by_tag[tags[0]]
            if all(tag in self.tags_by_pair[pair] for tag in tags)
            and all(tag not in self.tags_by_pair[pair] for tag in negative_tags)
        ]

        if rating_text:
            rating = rating_text.split(":")[1].strip()
            output = [
                pair for pair in output
                if self.rating_by_pair.get(pair) == rating
            ]

        return output


class CLIPIndex:
    def __init__(self, embedder: CLIPEmbedder, directory: str):
        self.embedder: CLIPEmbedder = embedder
        key = stable_hash(directory + embedder.model_name)
        self.cache_dir: Path = (Path(platformdirs.user_cache_dir()) / 'gallerysearch' ) / key
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_file: Path = self.cache_dir / "faiss_index.bin"
        self.meta_file: Path = self.cache_dir / "image_paths.pkl"

        # Gather gallery pairs
        self.gallery_pairs: List[GalleryPair] = gather_gallery_files(Path(directory))

        # Load or build the FAISS index
        self.index: Optional[faiss.Index] = None
        self._load_or_build_index()

    def _build_index(self, gallery_pairs: List[GalleryPair]) -> faiss.Index:
        """Builds a FAISS index from a list of images."""
        print("Building FAISS index...")

        embeddings: List[np.ndarray] = []
        image_paths: List[str] = []

        for pair in tqdm(gallery_pairs, desc="Embedding images"):
            emb = self.embedder.embed_image(pair)
            embeddings.append(emb)
            image_paths.append(str(pair.img_file.resolve()))

        if not embeddings:
            raise ValueError("No images found to build the index.")

        # Convert to NumPy array
        embeddings_array: np.ndarray = np.array(embeddings, dtype=np.float32)
        embedding_dim: int = embeddings_array.shape[1]

        # Create and populate FAISS index
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_array)

        # Save index and metadata
        self._save_index(index, image_paths)
        print("FAISS index built and saved.")

        return index

    def _load_or_build_index(self) -> None:
        """Loads an existing FAISS index or builds a new one if necessary."""
        if self.index_file.exists() and self.meta_file.exists():
            self.index, image_paths = self._load_index()
            self._update_index(image_paths)
        else:
            self.index = self._build_index(self.gallery_pairs)

    def _load_index(self) -> Tuple[faiss.Index, List[str]]:
        """Loads the FAISS index and metadata from disk."""
        print("Loading existing FAISS index...")
        index = faiss.read_index(str(self.index_file))

        with open(self.meta_file, "rb") as f:
            image_paths = pickle.load(f)

        if index.ntotal != len(image_paths):
            raise ValueError("FAISS index and metadata file are inconsistent.")

        print(f"Loaded FAISS index with {index.ntotal} images.")
        return index, image_paths

    def _update_index(self, existing_paths: List[str]) -> None:
        """Adds new images to an existing FAISS index."""
        print("Checking for new images...")

        existing_set = set(existing_paths)
        new_pairs = [p for p in self.gallery_pairs if str(p.img_file.resolve()) not in existing_set]

        if not new_pairs:
            print("No new images found.")
            return

        print(f"Adding {len(new_pairs)} new images to FAISS index...")

        new_embeddings: List[np.ndarray] = []
        for pair in tqdm(new_pairs, desc="Embedding new images"):
            emb = self.embedder.embed_image(pair)
            new_embeddings.append(emb)
            existing_paths.append(str(pair.img_file.resolve()))

        new_embeddings_array: np.ndarray = np.array(new_embeddings, dtype=np.float32)
        self.index.add(new_embeddings_array)

        # Save updated index and metadata
        self._save_index(self.index, existing_paths)
        print("Index updated successfully.")

    def _save_index(self, index: faiss.Index, image_paths: List[str]) -> None:
        """Saves FAISS index and image metadata to disk."""
        faiss.write_index(index, str(self.index_file))

        with open(self.meta_file, "wb") as f:
            pickle.dump(image_paths, f)

    def search(self, text: str, top_k: int = 5) -> List[GalleryPair]:
        """Searches the FAISS index using a text query and returns GalleryPairs."""
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("FAISS index is empty or not initialized.")

        # Embed text query
        query_embedding: np.ndarray = np.array(self.embedder.embed_text(text), dtype=np.float32)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve matching images
        image_paths: List[str]
        with open(self.meta_file, "rb") as f:
            image_paths = pickle.load(f)

        results: List[GalleryPair] = [
            GalleryPair(Path(image_paths[i]), Path(image_paths[i]).with_suffix(".json"))
            for i in indices[0]
        ]
        return results
