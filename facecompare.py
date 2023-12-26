import cv2
import dlib
import numpy as np

def compare_images(image_path1, image_path2):
    # Load the pre-trained face detector from dlib
    face_detector = dlib.get_frontal_face_detector()

    # Load the pre-trained facial landmarks predictor from dlib
    shape_predictor = dlib.shape_predictor(r"C:\Users\b\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat") # download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

    # Load the pre-trained face recognition model from dlib
    face_recognizer = dlib.face_recognition_model_v1(r"C:\Users\b\Downloads\dlib_face_recognition_resnet_model_v1.dat\dlib_face_recognition_resnet_model_v1.dat") # download: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

    # Load the images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    print("Loaded images.")

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print("Converted images to grayscale.")

    # Detect faces in the images
    faces1 = face_detector(gray1)
    faces2 = face_detector(gray2)

    print(f"Detected {len(faces1)} faces in the first image.")
    print(f"Detected {len(faces2)} faces in the second image.")

    # Get facial landmarks for the detected faces
    landmarks1 = [shape_predictor(gray1, face) for face in faces1]
    landmarks2 = [shape_predictor(gray2, face) for face in faces2]

    print("Got facial landmarks for both images.")

    # Print facial landmarks
    for i, landmarks in enumerate([landmarks1, landmarks2]):
        print(f"\nFacial landmarks for Image {i + 1}:\n")
        for landmark in landmarks:
            for point in landmark.parts():
                print(f"({point.x}, {point.y})", end=" ")
            print()

    # Compute face descriptors for the detected faces
    descriptors1 = [face_recognizer.compute_face_descriptor(img1, landmark) for landmark in landmarks1]
    descriptors2 = [face_recognizer.compute_face_descriptor(img2, landmark) for landmark in landmarks2]

    print("Computed face descriptors for both images.")

    # Compare the face descriptors
    for desc1 in descriptors1:
        for desc2 in descriptors2:
            # Convert descriptors to NumPy arrays
            desc1_np = np.array(desc1)
            desc2_np = np.array(desc2)

            # Calculate the Euclidean distance between the descriptors
            distance = np.linalg.norm(desc1_np - desc2_np)

            print(f"Face distance: {distance}")

            # Define a threshold for face similarity
            threshold = 0.6

            # Compare the distance with the threshold
            if distance < threshold:
                print("Same person detected!")
                return True

    print("Different persons detected.")
    return False


# Example usage
image1_path = r"C:\Users\b\Downloads\w2000_h2000_fmax.jpg"
image2_path = r"C:\Users\b\Downloads\Screenshots\Screenshot 2023-12-26 134007.png"
compare_images(image1_path, image2_path)
