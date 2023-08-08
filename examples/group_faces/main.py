import cv2
import numpy as np
from numpy.linalg import norm
import insightface
from insightface.app import FaceAnalysis
import time
import os

# prepare insightface model
def prepareEmbedding():
    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    return app


# get faces from image path
def getFaces(app, imagePath):
    img = cv2.imread(imagePath)
    faces = app.get(img)

    return faces


# calculate cosine simularity of two embedding vectors
def getCosSimularity(embed1, embed2):
    return np.dot(embed1, embed2) / (norm(embed1) * norm(embed2))


# calculate simularity of all faces in images
def calcSimularityAll(app, paths):
    # (embedding, bbox, id, image) of all faces
    embedBoxes = []

    # incremental id of faces
    faceId = 0

    # extract embedding vecotor and bbox of all faces and grant id
    s = time.time()
    for path in paths:
        debug(f' < "{path}" >')
        faces = getFaces(app, path)
        for face in faces:
            debug(f" - Face no.{faceId} detected!")
            embedBoxes.append([face.embedding, face.bbox, faceId, path])
            faceId += 1
        debug()
    debug(f" - face extraction elapsed time O(F*model): {time.time() - s:.2f} seconds")

    if len(embedBoxes) < 2:
        debug("no faces or only one face detected!")
        exit(-1)

    # group of (embedding, bbox, id, image) by simularity
    groups = {}

    # [TODO] more effective way
    # calculate simularity of all pairs of faces
    s = time.time()
    for i, (embed1, bbox1, id1, path1) in enumerate(embedBoxes):
        for j, (embed2, bbox2, id2, path2) in enumerate(embedBoxes):
            if i == j:
                continue
            if f"person{id2}" in groups:
                continue

            sim = getCosSimularity(embed1, embed2)

            if sim >= 0.6:
                key = f"person{id1}"
                if key in groups:
                    groups[key].append((embed2, bbox2, id2, path2))
                else:
                    groups[key] = [
                        (embed1, bbox1, id1, path1),
                        (embed2, bbox2, id2, path2),
                    ]

                embedBoxes[j][2] = id1
    debug(
        f" - simularity calculation elapsed time O(F^2): {time.time() - s:.2f} seconds"
    )

    # [TODO] remove (move into simularity calculation loop)
    # add single person
    s = time.time()
    for (embed1, bbox1, id1, path1) in embedBoxes:
        key = f"person{id1}"
        if key not in groups:
            groups[key] = [(embed1, bbox1, id1, path1)]
    debug(f" - single person addition elapsed time O(F): {time.time() - s:.2f} seconds")

    formatedGroups = {}

    # [TODO] remove (move into simularity calculation loop)
    # regenerate person key
    personId = 0  # incremental id of persons

    s = time.time()
    for person in groups:
        formatedGroups[f"person{personId}"] = groups[person]
        personId += 1
    debug(f" - person key regeneration O(F): {time.time() - s:.2f} seconds")

    return formatedGroups


# for tests
def getFileNames(directory):
    names = os.listdir(directory)
    names = [os.path.join(directory, name) for name in names if name != ".gitignore"]

    return names


# for tests
def mergedFileNames(directories):
    names = []
    for directory in directories:
        names.extend(getFileNames(directory))

    return names


# for tests
def debug(message=""):
    # for immediate print in docker
    print(message, flush=True)


# demo
if __name__ == "__main__":
    app = prepareEmbedding()

    directories = getFileNames("images")
    imgPaths = mergedFileNames(directories)
    debug(f"target images: \n{imgPaths}")
    debug(f"target images count: {len(imgPaths)}\n")

    start = time.time()
    groups = calcSimularityAll(app, imgPaths)
    debug(f"total elapsed time: {time.time() - start:.2f} seconds\n")

    for person in groups:
        debug(f"==== {person} ====")
        for face in groups[person]:
            debug(face[3])  # image path
            # debug(face[1])  # bbox (bounding box)
