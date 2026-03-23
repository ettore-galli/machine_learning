from pathlib import Path
from typing import Iterable

from ddgs import DDGS

from fastai.vision.utils import resize_images, download_images

from fastai_work.base import IMAGE_CATEGORIES, IMAGES_PATH, MAX_IMAGES

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger()


def search_images(keywords, max_images=1):
    log.info("Looking for %s, max images = %s", keywords, max_images)
    return DDGS().images(keywords, max_results=max_images)


def get_image_path(images_path: Path, subpath_label: str) -> Path:
    return images_path / subpath_label


def perform_source_data_search(
    images_path: Path, searches: Iterable[str], max_images: int
):
    for subpath_label in searches:
        dest = get_image_path(images_path, subpath_label)
        dest.mkdir(exist_ok=True, parents=True)
        image_urls = [
            result["image"]
            for result in search_images(
                f"{subpath_label} photos", max_images=max_images
            )
        ]
        download_images(
            urls=image_urls,
            dest=dest,
        )
        resize_images(images_path / subpath_label, max_size=400)


if __name__ == "__main__":
    perform_source_data_search(
        images_path=IMAGES_PATH,
        searches=IMAGE_CATEGORIES,
        max_images=MAX_IMAGES,
    )
