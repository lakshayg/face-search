#!/usr/bin/env python
import argparse
import face_recognition
import filetype
import io
import logging
import numpy
import pathlib
import pillow_heif
import sqlite3
import tqdm

pillow_heif.register_heif_opener()


FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)


# Converts np.array to and from TEXT when inserting
# https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
def adapt_numpy_array(arr):
    out = io.BytesIO()
    numpy.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_numpy_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return numpy.load(out)


sqlite3.register_adapter(numpy.ndarray, adapt_numpy_array)
sqlite3.register_converter("numpy_ndarray", convert_numpy_array)


class AlbumIndex(object):
    def __init__(self, album_path: pathlib.Path):
        self.album_path = album_path
        self.index_path = album_path / "index.sqlite3"
        self.index = None
        if self.index_path.exists():
            log.info(f"Reading '{self.index_path.as_posix()}'")
            self.index = sqlite3.connect(
                self.index_path, detect_types=sqlite3.PARSE_DECLTYPES
            )

    def exists(self) -> bool:
        return self.index is not None

    def initialize(self):
        if self.exists():
            raise Exception("index already exists")
        log.info(f"Initializing '{self.index_path.as_posix()}'")
        self.index = sqlite3.connect(
            self.index_path, detect_types=sqlite3.PARSE_DECLTYPES
        )
        cursor = self.index.cursor()
        cursor.execute("CREATE TABLE files(name TEXT)")
        cursor.execute("CREATE TABLE link(filename TEXT, face_vector numpy_ndarray)")
        self.index.commit()

    def insert(self, filename: str):
        encodings = face_recognition.face_encodings(
            face_recognition.load_image_file(self.album_path / filename)
        )
        cursor = self.index.cursor()
        cursor.execute("INSERT INTO files(name) VALUES (?)", (filename,))
        for encoding in encodings:
            cursor.execute(
                "INSERT INTO link(filename, face_vector) VALUES (?, ?)",
                (filename, encoding),
            )
        self.index.commit()

    def list_files(self) -> set[str]:
        cursor = self.index.cursor()
        cursor.execute("SELECT * FROM files")
        return set(x[0] for x in cursor.fetchall())

    def list_links(self):
        cursor = self.index.cursor()
        cursor.execute("SELECT * FROM link")
        return cursor.fetchall()

    def find_face(self, path: pathlib.Path) -> list[str]:
        encodings = face_recognition.face_encodings(
            face_recognition.load_image_file(path)
        )
        if len(encodings) == 0:
            print("No face was found in the input image")
            return []
        if len(encodings) > 1:
            raise Exception("Support for multiple faces is input is unimplemented")
        return [
            filename
            for filename, known_encoding in self.list_links()
            if face_recognition.compare_faces([known_encoding], encodings[0])[0]
        ]


class Album(object):
    def __init__(self, path: pathlib.Path):
        if not path.exists():
            raise FileNotFoundError(f"Could not find '{path.as_posix()}'")
        if not path.is_dir():
            raise TypeError(f"album needs to be a directory")
        self.path = path

    def list_files(self) -> list[str]:
        return [
            file.relative_to(self.path).as_posix()
            for file in self.path.rglob("*")
            if file.is_file() and filetype.is_image(file)
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("album")
    parser.add_argument("--find-face")
    parser.add_argument("--delete-index", action="store_true")
    args = parser.parse_args()

    album = Album(pathlib.Path(args.album))
    index = AlbumIndex(album.path)

    if args.delete_index:
        if index.exists():
            log.info(f"Deleting '{index.index_path.as_posix()}'")
            index.index_path.unlink(missing_ok=False)
        return

    if not index.exists():
        index.initialize()

    indexed_files = index.list_files()
    new_files = [file for file in album.list_files() if file not in indexed_files]
    if len(new_files) > 0:
        log.info(f"Updating index at '{index.index_path.as_posix()}'")
        for file in tqdm.tqdm(new_files):
            index.insert(file)

    if args.find_face is not None:
        images = index.find_face(args.find_face)
        print(f"Found {len(images)} image(s)")
        for image in images:
            print((album.path / image).as_posix())


if __name__ == "__main__":
    main()
