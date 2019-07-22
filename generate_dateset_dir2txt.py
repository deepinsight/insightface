import os

'''
Point to a root directory where the structure is like this:
Person A:
 - image_1.jpg
 - image_2.jpg
 - image_3.jpg
Person B:
 - image_4.jpg
 - image_5.jpg
 - image_6.jpg
Person C:
 - image_7.jpg
 - image_8.jpg
 - image_9.jpg
 - image_10.jpg
...
Person N:
 - image_100.jpg
 - image_101.jpg
'''

output_text_file = open('face_annotations.txt', 'w')

ROOT_IMAGE_DIR = './data/msceleb_retina_crop'

for root, dirs, filenames in os.walk(ROOT_IMAGE_DIR):
	for filename in filenames:
		image_path = os.path.join(root, filename)
		print(image_path)
