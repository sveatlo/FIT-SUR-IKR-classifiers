#!/usr/bin/env python3

import sys
from pathlib import Path

import PIL
import face_recognition
import numpy as np

class Dataloader():
    """ Dataloader is responsible for loading samples in a format for Recognizer """
    def __init__(self, dataset_path, sample_type="image"):
        """
            Creates new instance of Dataloader

            :param dataset_path: path to directory containing 3 subdirectories (dev, train, eval) with data samples
            :param dataset_type: train/dev/eval
            :param sample_type: sample type - could be 'image', 'audio'
        """
        super().__init__()

        self.path = Path(dataset_path)
        self.sample_type = sample_type

    def train_samples(self):
        return self._samples("train")

    def test_samples(self):
        return self._samples("dev")

    def eval_samples(self):
        for (_, sample) in self._samples("eval"):
            yield sample

    def _samples(self, subdir="train"):
        """
        Generator function yielding samples

        :returns: (id, sample_path). sample can be image or audio depending on the type

        """

        glob = self._glob()

        for sample_path in (self.path / subdir).glob(glob):
            yield (sample_path.parent.name, sample_path)

    def _glob(self):
        globs = {
            "audio": "**/*.wav",
            "image": "**/*.png",
            #  "both": "**/*"
        }
        return globs[self.sample_type]

class Recognizer():

    """ Recognizer ... TBD  """

    def __init__(self):
        self.known_encodings = []
        self.known_ids = []

    def learn(self, face_id, face_img_path):
        img = face_recognition.load_image_file(face_img_path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) != 1:
            print(f"{face_img_path} has a weird number of faces ({len(encodings)})", file=sys.stderr)
            return

        self.known_encodings.append(encodings[0])
        self.known_ids.append(face_id)

    def recognize(self, face_img_path):
        img = face_recognition.load_image_file(face_img_path)
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        if len(face_encodings) != 1:
            print(f"{face_img_path} has a weird number of faces ({len(face_encodings)})", file=sys.stderr)
            return -1, []
        face_encoding = face_encodings[0]

        matches = face_recognition.compare_faces(self.known_encodings, face_encoding)

        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            face_id = self.known_ids[best_match_index]
            return face_id, face_distances
        else:
            return -1, []


def main():
    recognizer = Recognizer()

    dataset = Dataloader("./dataset")

    for (face_id, face_img_path) in dataset.train_samples():
        recognizer.learn(face_id, face_img_path)

    print("======================================================")

    #  total, correct = 0, 0
    #  for (face_id, face_img_path) in dataset.test_samples():
    #      recognized_face_id, _ = recognizer.recognize(face_img_path)
    #
    #      total+=1
    #      if face_id == recognized_face_id:
    #          correct += 1
    #
    #  print(f"{correct}/{total}")

    for face_img_path in dataset.eval_samples():
        recognized_face_id, distances = recognizer.recognize(face_img_path)

        sample_id = face_img_path.stem

        print(f"{sample_id} {recognized_face_id} {' '.join([str(d) for d in distances])}")




if __name__ == "__main__":
    main()
