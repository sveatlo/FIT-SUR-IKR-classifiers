#!/usr/bin/env python3

import sys
from pathlib import Path

import PIL
import face_recognition
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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

        class_distances = {}
        for i, dist in enumerate(face_distances):
            face_id = self.known_ids[i]

            if face_id not in class_distances:
                class_distances[face_id] = dist

            class_distances[face_id] = min(class_distances[face_id], dist)
        distances = list(class_distances.values())

        face_id = -1
        if matches[best_match_index]:
            face_id = self.known_ids[best_match_index]

        return face_id, distances


EVAL = True

def face_recognition_main():
    dataset = Dataloader("./dataset")

    recognizer = Recognizer()
    for (face_id, face_img_path) in dataset.train_samples():
        recognizer.learn(face_id, face_img_path)

    print("======================================================", file=sys.stderr)

    if not EVAL:
        print("Running test", file=sys.stderr)
        total, correct = 0, 0
        for (face_id, face_img_path) in dataset.test_samples():
            recognized_face_id, _ = recognizer.recognize(face_img_path)

            total+=1
            if face_id == recognized_face_id:
                correct += 1

        print(f"{correct/total} ({correct}/{total})")
    else:
        with open('image_face_recognition.txt', 'w') as out:
            for face_img_path in dataset.eval_samples():
                recognized_face_id, distances = recognizer.recognize(face_img_path)

                sample_id = face_img_path.stem

                probs = ["NaN" for d in range(31)]

                print(f"{sample_id} {recognized_face_id} {' '.join([str(d) for d in probs])}", file=out)

def face_recognition_scikit(classifier='mlp'):
    dataset = Dataloader("./dataset")

    classifiers = {
        'mlp': MLPClassifier(hidden_layer_sizes=(800,), activation='tanh', learning_rate='adaptive', alpha=1e-4, max_iter=400, random_state=1),
        'svc': SVC(C=1e2, probability=True),
        'knn': KNeighborsClassifier(),
    }
    clf = classifiers[classifier]

    X_train = []
    Y_train = []
    for (face_id, face_img_path) in dataset.train_samples():
        img = imread(face_img_path, plugin='pil')
        out, hog_img = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L1', visualize=True)
        Y_train.append(face_id)
        X_train.append(out)

    clf = clf.fit(X_train, Y_train)

    print("======================================================", file=sys.stderr)

    if not EVAL:
        print(f"Running test [{classifier}]", file=sys.stderr)
        X_test = []
        Y_test = []
        for (face_id, face_img_path) in dataset.test_samples():
            img = imread(face_img_path, plugin='pil')
            out, hog_img = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L1', visualize=True)
            Y_test.append(face_id)
            X_test.append(out)

        Y_pred = clf.predict(X_test)

        total, correct = 0, 0
        for gt, pred in zip(Y_test, Y_pred):
            total += 1
            if gt == pred:
                correct += 1
        correct, total, correct/total

        print(f"{correct/total} ({correct}/{total})")
    else:
        names = []
        X_eval = []
        for face_img_path in dataset.eval_samples():
            img = imread(face_img_path, plugin='pil')
            out, hog_img = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L1', visualize=True)
            names.append(face_img_path.stem)
            X_eval.append(out)

        pred_proba = clf.predict_proba(X_eval)

        with open('image_{}.txt'.format(classifier), 'w') as out:
            for name, proba in zip(names, pred_proba):
                print(name,
                      1 + np.argmax(proba),
                      ' '.join([str(x) for x in proba.tolist()]),
                      file=out)

if __name__ == "__main__":
    face_recognition_main()
    for classifier in ('mlp', 'svc', 'knn'):
        face_recognition_scikit(classifier)
