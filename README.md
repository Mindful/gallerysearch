# Gallery Search

A super simple semantic search application using CLIP and FAISS for semantic search.
Mostly intended for use with [gallery-dl](https://github.com/mikf/gallery-dl), although the image 
search can be used with images of any kind.

```shell
conda create -n gallerysearch python=3.13
pip install -e .

# image search with CLIP 
# feh is a lightweight image viewer, "eog" also works for gnome-based desktops
galsearch imgsearch --dir /home/Pictures --query "dog" --open_with "feh -t -P --scale-down"

# tag illustrations and then search by tags
galsearch tag --dir /home/Pictures/illustrations
galsearch tagsearch --dir /home/Pictures/illustrations --tags "landscape, mountains"
```