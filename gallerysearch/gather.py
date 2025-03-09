from pathlib import Path
from collections import namedtuple
from typing import Optional

GalleryPair = namedtuple('GalleryPair', ['img_file', 'json_file'])

def gather_gallery_files(directory: Path, curr_list: Optional[list[GalleryPair]] = None) -> list[GalleryPair]:
    if curr_list is None:
        curr_list = []

    for item in directory.iterdir():
        if item.is_dir():
            gather_gallery_files(item, curr_list)
        elif item.suffix != '.json':
            img_file = item
            json_file = item.parent / (item.name + '.json')
            assert json_file.exists(), f'Missing JSON file {json_file}'
            curr_list.append(GalleryPair(img_file.resolve(), json_file.resolve()))

    return curr_list
