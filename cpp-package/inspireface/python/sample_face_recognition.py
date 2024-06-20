import os
import cv2
import inspireface as ifac
from inspireface.param import *
import click

@click.command()
@click.argument("resource_path")
@click.argument('test_data_folder')
def case_face_recognition(resource_path, test_data_folder):
    """
    Launches the face recognition system, inserts face features into a database, and performs searches.
    Args:
        resource_path (str): Path to the resource directory for face recognition algorithms.
        test_data_folder (str): Path to the test data containing images for insertion and recognition tests.
    """
    # Initialize the face recognition system with provided resources.
    ret = ifac.launch(resource_path)
    assert ret, "Launch failure. Please ensure the resource path is correct."

    # Enable face recognition features.
    opt = HF_ENABLE_FACE_RECOGNITION
    session = ifac.InspireFaceSession(opt, HF_DETECT_MODE_ALWAYS_DETECT)

    # Configure the feature management system.
    feature_hub_config = ifac.FeatureHubConfiguration(
        feature_block_num=10,
        enable_use_db=False,
        db_path="",
        search_threshold=0.48,
        search_mode=HF_SEARCH_MODE_EAGER,
    )
    ret = ifac.feature_hub_enable(feature_hub_config)
    assert ret, "Failed to enable FeatureHub."

    # Insert face features from 'bulk' directory.
    bulk_path = os.path.join(test_data_folder, "bulk")
    assert os.path.exists(bulk_path), "Bulk directory does not exist."

    insert_images = [os.path.join(bulk_path, path) for path in os.listdir(bulk_path) if path.endswith(".jpg")]
    for idx, image_path in enumerate(insert_images):
        name = os.path.basename(image_path).replace(".jpg", "")
        image = cv2.imread(image_path)
        assert image is not None, f"Failed to load image {image_path}"
        faces = session.face_detection(image)
        if faces:
            face = faces[0]  # Assume the most prominent face is what we want.
            feature = session.face_feature_extract(image, face)
            identity = ifac.FaceIdentity(feature, custom_id=idx, tag=name)
            ret = ifac.feature_hub_face_insert(identity)
            assert ret, "Failed to insert face."

    count = ifac.feature_hub_get_face_count()
    print(f"Number of faces inserted: {count}")

    # Process faces from 'RD' directory and insert them.
    RD = os.path.join(test_data_folder, "RD")
    assert os.path.exists(RD), "RD directory does not exist."
    RD_images = [os.path.join(RD, path) for path in os.listdir(RD) if path.endswith(".jpeg")]

    for idx, image_path in enumerate(RD_images[:-1]):
        name = os.path.basename(image_path).replace(".jpeg", "")
        image = cv2.imread(image_path)
        assert image is not None, f"Failed to load image {image_path}"
        faces = session.face_detection(image)
        if faces:
            face = faces[0]
            feature = session.face_feature_extract(image, face)
            identity = ifac.FaceIdentity(feature, custom_id=idx+count+1, tag=name)
            ret = ifac.feature_hub_face_insert(identity)
            assert ret, "Failed to insert face."

    count = ifac.feature_hub_get_face_count()
    print(f"Total number of faces after insertion: {count}")

    # Search for a similar face using the last image in RD directory.
    remain = cv2.imread(RD_images[-1])
    assert remain is not None, f"Failed to load image {RD_images[-1]}"
    faces = session.face_detection(remain)
    assert faces, "No faces detected."
    face = faces[0]
    feature = session.face_feature_extract(remain, face)

    search = ifac.feature_hub_face_search(feature)
    if search.similar_identity.custom_id != -1:
        print(f"Found similar identity with ID: {search.similar_identity.custom_id}, Tag: {search.similar_identity.tag}, Confidence: {search.confidence:.2f}")
    else:
        print("No similar identity found.")

    # Display top-k similar face identities.
    print("Top-k similar identities:")
    search_top_k = ifac.feature_hub_face_search_top_k(feature, 10)
    for idx, (conf, custom_id) in enumerate(search_top_k):
        identity = ifac.feature_hub_get_face_identity(custom_id)
        print(f"Top-{idx + 1}: {identity.tag}, ID: {custom_id}, Confidence: {conf:.2f}")


if __name__ == '__main__':
    case_face_recognition()
