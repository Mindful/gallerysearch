# Gallery Search

A super simple semantic search application using CLIP and FAISS for semantic search.
Mostly intended for use with [gallery-dl](https://github.com/mikf/gallery-dl), although the image 
search can be used with images of any kind.

```shell
conda create -n gallerysearch python=3.13
pip install -r requirements.txt
# "eog" is to open with eye of gnome on linux
galsearch --dir /home/Pictures --query "dog" --open_with eog
```