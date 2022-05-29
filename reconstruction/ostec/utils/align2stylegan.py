# Copyright (c) 2020, Baris Gecer. All rights reserved.
#
# This work is made available under the CC BY-NC-SA 4.0.
# To view a copy of this license, see LICENSE

import numpy as np
import scipy.ndimage
import PIL.Image

def create_perspective_transform_matrix(src, dst):
    """ Creates a perspective transformation matrix which transforms points
        in quadrilateral ``src`` to the corresponding points on quadrilateral
        ``dst``.

        Will raise a ``np.linalg.LinAlgError`` on invalid input.
        """
    # See:
    # * http://xenia.media.mit.edu/~cwren/interpolator/
    # * http://stackoverflow.com/a/14178717/71522
    in_matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        in_matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    A = np.matrix(in_matrix, dtype=np.float)
    B = np.array(dst).reshape(8)
    af = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.append(np.array(af).reshape(8), 1).reshape((3, 3))


def create_perspective_transform(src, dst, round=False, splat_args=False):
    """ Returns a function which will transform points in quadrilateral
        ``src`` to the corresponding points on quadrilateral ``dst``::

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ... )
            >>> transform((5, 5))
            (74.99999999999639, 74.999999999999957)

        If ``round`` is ``True`` then points will be rounded to the nearest
        integer and integer values will be returned.

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ...     round=True,
            ... )
            >>> transform((5, 5))
            (75, 75)

        If ``splat_args`` is ``True`` the function will accept two arguments
        instead of a tuple.

            >>> transform = create_perspective_transform(
            ...     [(0, 0), (10, 0), (10, 10), (0, 10)],
            ...     [(50, 50), (100, 50), (100, 100), (50, 100)],
            ...     splat_args=True,
            ... )
            >>> transform(5, 5)
            (74.99999999999639, 74.999999999999957)

        If the input values yield an invalid transformation matrix an identity
        function will be returned and the ``error`` attribute will be set to a
        description of the error::

            >>> tranform = create_perspective_transform(
            ...     np.zeros((4, 2)),
            ...     np.zeros((4, 2)),
            ... )
            >>> transform((5, 5))
            (5.0, 5.0)
            >>> transform.error
            'invalid input quads (...): Singular matrix
        """
    try:
        transform_matrix = create_perspective_transform_matrix(src, dst)
        error = None
    except np.linalg.LinAlgError as e:
        transform_matrix = np.identity(3, dtype=np.float)
        error = "invalid input quads (%s and %s): %s" %(src, dst, e)
        error = error.replace("\n", "")

    to_eval = "def perspective_transform(%s):\n" %(
        splat_args and "*pt" or "pt",
    )
    to_eval += "  res = np.dot(transform_matrix, ((pt[0], ), (pt[1], ), (1, )))\n"
    to_eval += "  res = res / res[2]\n"
    if round:
        to_eval += "  return (int(round(res[0][0])), int(round(res[1][0])))\n"
    else:
        to_eval += "  return (res[0][0], res[1][0])\n"
    locals = {
        "transform_matrix": transform_matrix,
    }
    locals.update(globals())
    exec(to_eval,locals,locals)
    res = locals["perspective_transform"]
    res.matrix = transform_matrix
    res.error = error
    return res


def align_mesh2stylegan(temp_tcoords, transformation_params):
    temp_tcoords = temp_tcoords.copy()
    temp_tcoords[:, 0] = temp_tcoords[:, 0] - transformation_params['crop'][1]
    temp_tcoords[:, 1] = temp_tcoords[:, 1] - transformation_params['crop'][0]

    temp_tcoords[:, 0] = temp_tcoords[:, 0] + transformation_params['pad'][1]
    temp_tcoords[:, 1] = temp_tcoords[:, 1] + transformation_params['pad'][0]

    h, w = (4096, 4096)  # transformation_params['new_size']
    transform = create_perspective_transform(
        transformation_params['quad'],
        [(0, 0), (0, h), (h, w), (w, 0)],
        splat_args=True,
    )
    for i in range(len(temp_tcoords)):
        temp_tcoords[i, 1], temp_tcoords[i, 0] = transform(temp_tcoords[i, 1], temp_tcoords[i, 0])

    new_tcoords = temp_tcoords[:, ::-1] / (h, w)  # transformation_params['new_size']
    new_tcoords[:, 1] = 1 - new_tcoords[:, 1]
    return new_tcoords

def align_im2stylegan(src_im, src_mask, face_landmarks, output_size=1024, transform_size=4096,
                    enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= x_scale
        y = np.flipud(x) * [-y_scale, y_scale]
        c = eye_avg + eye_to_mouth * em_scale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2
        rsize = None

        img = src_im.convert('RGBA').convert('RGB')

        img_mask = src_mask.convert('L')

        img.putalpha(img_mask)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = np.uint8(np.clip(np.rint(img), 0, 255))
            if alpha:
                mask = 1 - np.clip(3.0 * mask, 0.0, 1.0)
                mask = np.uint8(np.clip(np.rint(mask * 255), 0, 255))
                img = np.concatenate((img, mask), axis=2)
                img = PIL.Image.fromarray(img, 'RGBA')
            else:
                img = PIL.Image.fromarray(img, 'RGBA')
            quad += pad[:2]

        # Transform.
        aligned_mask = PIL.Image.fromarray(np.uint8(img)[:, :, 3])
        img = PIL.Image.fromarray(np.uint8(img)[:, :, :3])

        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        aligned_mask = aligned_mask.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                                              PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
            aligned_mask = aligned_mask.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        transformation_params = {
            'rsize': rsize,
            'crop': crop,
            'pad': pad,
            'quad': quad + 0.5,
            'new_size': (output_size, output_size)
        }
        # Save aligned image.
        return img, aligned_mask, transformation_params
