import argparse
import json
import subprocess
import os
from pathlib import Path
from pprint import pp

import PIL
from tqdm import tqdm
from imgutils.tagging import get_wd14_tags

from gallerysearch.embed import CLIPEmbedder
from gallerysearch.gather import gather_gallery_files
from gallerysearch.index import CLIPIndex, TagIndex

def run_opener(opener: str, image_paths: list[Path], page_size: int):
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    if opener and image_paths:
        if page_size == 0:
            pages = [image_paths]
        else:
            pages = [image_paths[i:i + page_size] for i in range(0, len(image_paths), page_size)]

        processes = [
            subprocess.Popen([*opener.split(), *page])
            for page in pages
        ]
        for proc in processes:
            proc.wait()

def imgsearch(args):
    embedder = CLIPEmbedder(model_name=args.model)
    index = CLIPIndex(embedder, args.dir)

    results = index.search(args.query)
    image_paths = [t.img_file for t in results]

    if args.print:
        pp(image_paths)

    run_opener(args.open_with, image_paths, args.page_size)

def tag(args):
    pairs = gather_gallery_files(Path(args.dir))
    for img_file, json_file in tqdm(pairs, 'Tagging'):
        with json_file.open('r+') as f:
            json_data = json.load(f)

            if 'gen_tags' in json_data:
                continue

            try:
                rating, features, chars = get_wd14_tags(img_file)
                json_data['gen_tags'] = {
                    'rating': sorted(list(rating.keys()), key=lambda x: rating[x], reverse=True)[0],
                    'features': list(features.keys()),
                    'chars': list(chars.keys()),
                }
            except PIL.UnidentifiedImageError:
                json_data['gen_tags'] = {
                    'rating': '',
                    'features': [],
                    'chars': [],
                }

            f.seek(0)
            json.dump(json_data, f, indent=4)

def tagsearch(args):
    i = TagIndex(args.dir)
    results = i.search(args.query)

    image_paths = [t.img_file for t in results]

    if args.print:
        pp(image_paths)

    run_opener(args.open_with, image_paths, args.page_size)

def generate_jsonfiles(args):
    dir = Path(args.dir)
    for file in dir.glob('*'):
        if file.suffix == '.json':
            continue
        json_file = dir / (file.name + '.json')
        if not json_file.exists():
            with json_file.open('w') as f:
                json.dump({'category': 'manual'}, f, indent=4)



def main():
    parser = argparse.ArgumentParser(prog='gallerysearch')
    subparsers = parser.add_subparsers(dest='command', required=True)

    imgsearch_parser = subparsers.add_parser('imgsearch', help='Search for images using CLIP')
    imgsearch_parser.add_argument('--dir', type=str, required=True, help='Directory containing images')
    imgsearch_parser.add_argument('--query', type=str, required=True, help='Search query')
    imgsearch_parser.add_argument('--model', type=str, default="openai/clip-vit-base-patch16", help='CLIP model to use')
    imgsearch_parser.add_argument('--print', type=bool, default=True, help='Print results')
    imgsearch_parser.add_argument('--open_with', type=str, default=None, help='Program to open results with')
    imgsearch_parser.add_argument('--page_size', type=int, default=50, help='Number of results to return')
    imgsearch_parser.set_defaults(func=imgsearch)

    tag_parser = subparsers.add_parser('tag', help='Tag images')
    tag_parser.add_argument('--dir', type=str, required=True, help='Directory containing images')
    tag_parser.set_defaults(func=tag)

    ts_parser = subparsers.add_parser('tagsearch', help='Tag search')
    ts_parser.add_argument('--dir', type=str, required=True, help='Directory containing images')
    ts_parser.add_argument('--query', type=str, required=True, help='Space-separated list of tags')
    ts_parser.add_argument('--print', type=bool, default=True, help='Print results')
    ts_parser.add_argument('--open_with', type=str, default=None, help='Program to open results with')
    ts_parser.add_argument('--page_size', type=int, default=50, help='Number of results to return')
    ts_parser.set_defaults(func=tagsearch)

    jsonfile_parser = subparsers.add_parser('jsonfile', help='Generate json files')
    jsonfile_parser.add_argument('--dir', type=str, required=True, help='Directory containing images')
    jsonfile_parser.set_defaults(func=generate_jsonfiles)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
