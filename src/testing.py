from src.facedetector import face_detector

images = ['/home/dandy/PycharmProjects/facenet/data/images/t1.jpg', '/home/dandy/PycharmProjects/facenet/data/images/t2.jpg', '/home/dandy/PycharmProjects/facenet/data/images/t3.jpg']
aligned = face_detector.align_face(images)
comparisons = face_detector.compare(aligned, 0.3)

print("Is image 1 and 2 similar? ", bool(comparisons[0][1]))
print("Is image 1 and 3 similar? ", bool(comparisons[0][2]))