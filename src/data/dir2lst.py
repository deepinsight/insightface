
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image

input_dir = sys.argv[1]

dataset = face_image.get_dataset_common(input_dir, 2)

for item in dataset:
  print("%d\t%s\t%d" % (1, item.image_path, int(item.classname)))


