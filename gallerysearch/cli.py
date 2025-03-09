import argparse
import subprocess
from pprint import pp

from gallerysearch.embed import CLIPEmbedder
from gallerysearch.index import Index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--model', type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument('--print', type=bool, default=True)
    parser.add_argument('--open_with', type=str, default=None)

    args = parser.parse_args()

    embedder = CLIPEmbedder(model_name=args.model)
    index = Index(embedder, args.dir)

    results = index.search(args.query)
    image_paths = [t.img_file for t in results]

    if args.print:
        pp(image_paths)

    if args.open_with:
        subprocess.run([args.open_with, *image_paths])


