from test import *
import unittest
import cv2


@optional(ENABLE_LFW_PRECISION_TEST, "LFW dataset precision tests have been closed.")
class LFWPrecisionTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.quick = QuickComparison()

    def test_lfw_precision(self):
        pairs_path = os.path.join(LFW_FUNNELED_DIR_PATH, 'pairs.txt')
        pairs = read_pairs(pairs_path)
        self.assertEqual(True, len(pairs) > 0)
        if os.path.exists(LFW_PREDICT_DATA_CACHE_PATH):
            print("Loading results from cache")
            cache = np.load(LFW_PREDICT_DATA_CACHE_PATH, allow_pickle=True)
            similarities = cache[0]
            labels = cache[1]
        else:
            similarities = []
            labels = []

            for pair in tqdm(pairs):
                if len(pair) == 3:
                    person, img_num1, img_num2 = pair
                    img_path1 = os.path.join(LFW_FUNNELED_DIR_PATH, person, f"{person}_{img_num1.zfill(4)}.jpg")
                    img_path2 = os.path.join(LFW_FUNNELED_DIR_PATH, person, f"{person}_{img_num2.zfill(4)}.jpg")
                    match = True
                else:
                    person1, img_num1, person2, img_num2 = pair
                    img_path1 = os.path.join(LFW_FUNNELED_DIR_PATH, person1, f"{person1}_{img_num1.zfill(4)}.jpg")
                    img_path2 = os.path.join(LFW_FUNNELED_DIR_PATH, person2, f"{person2}_{img_num2.zfill(4)}.jpg")
                    match = False

                img1 = cv2.imread(img_path1)
                img2 = cv2.imread(img_path2)

                if not self.quick.setup(img1, img2):
                    print("not detect face")
                    continue

                cosine_similarity = self.quick.comp()
                similarities.append(cosine_similarity)
                labels.append(match)

            similarities = np.array(similarities)
            labels = np.array(labels)
            # save cache file
            np.save(LFW_PREDICT_DATA_CACHE_PATH, [similarities, labels])

        # find best threshold
        best_threshold, best_accuracy = find_best_threshold(similarities, labels)
        print(f"Best Threshold: {best_threshold:.2f}, Best Accuracy: {best_accuracy:.3f}")


if __name__ == '__main__':
    unittest.main()
