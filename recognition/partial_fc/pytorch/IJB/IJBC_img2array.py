import torch

def get_image_feature(img_path, files_list, model_path, epoch, gpu_id):
    batch_size = args.batch_size
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    img_feats = np.empty((len(files), 1024), dtype=np.float32)

    batch_data = np.empty((2 * batch_size, 3, 112, 112))
    embedding = Embedding(model_path, epoch, data_shape, batch_size, gpu_id)
    for img_index, each_line in enumerate(files[:len(files) - rare_size]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)
        # print(2*(img_index-batch*batch_size), 2*(img_index-batch*batch_size)+1)
        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        if (img_index + 1) % batch_size == 0:
            print('batch', batch)
            img_feats[batch * batch_size:batch * batch_size +
                      batch_size][:] = embedding.forward_db(batch_data)
            batch += 1
        faceness_scores.append(name_lmk_score[-1])