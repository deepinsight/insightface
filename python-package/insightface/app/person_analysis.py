"""Single-image person features with explicit, bounded reference updates."""
import json
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from .common import Face as NativeFace
from .person import PersonConfig, Face, Person, MatchResult, UpdateResult, RegistrationResult
from .person.matrix import ReferenceMatrix
from .person.types import normalize_feature, validate_image
from ..model_zoo.person_package import prepare_models


def _iou(first, second):
    intersection = np.maximum(0., np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])).prod()
    union = np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - intersection
    return float(intersection / union) if union > 0 else 0.


def _associate_faces(persons, faces, margin):
    """Use one-frame geometry only; an ambiguous face remains independent."""
    candidates = []
    for face_index, face in enumerate(faces):
        box = face.bbox
        area = float(np.prod(box[2:] - box[:2]))
        center = (box[:2] + box[2:]) / 2
        for person_index, person in enumerate(persons):
            body = person.body_bbox
            width, height = body[2:] - body[:2]
            if not (body[0] <= center[0] <= body[2] and body[1] <= center[1] <= body[3]):
                continue
            overlap = np.maximum(0., np.minimum(box[2:], body[2:]) - np.maximum(box[:2], body[:2])).prod()
            coverage = float(overlap / area)
            if coverage < .75 or area > 1.5 * width * height:
                continue
            offset = abs(center[0] - (body[0] + body[2]) / 2) / width
            score = .75 * coverage + .25 * max(0., 1. - 2. * offset)
            candidates.append((score, face_index, person_index))
    used_faces, used_people = set(), set()
    for score, face_index, person_index in sorted(candidates, reverse=True):
        if face_index in used_faces or person_index in used_people:
            continue
        rivals = [other for other, fi, pi in candidates
                  if (fi == face_index and pi != person_index) or (pi == person_index and fi != face_index)]
        if any(score - rival < margin for rival in rivals):
            continue
        persons[person_index].face = faces[face_index]
        persons[person_index]._association_valid = True
        used_faces.add(face_index)
        used_people.add(person_index)
    return [face for index, face in enumerate(faces) if index not in used_faces]


def _images(value):
    if value is None:
        return []
    if isinstance(value, (str, Path, np.ndarray)):
        return [value]
    return list(value)


def _read_image(value):
    if isinstance(value, (str, Path)):
        try:
            encoded = np.fromfile(Path(value).expanduser(), dtype=np.uint8)
            value = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        except (OSError, cv2.error) as error:
            raise ValueError('image could not be read') from error
    return validate_image(value)


class PersonAnalysis:
    """Detect and extract features; match and update references explicitly.

    References live only in this object's memory. There are no tracks, automatic
    anonymous identities, input readers, callbacks, clocks or database writes.
    """
    def __init__(self, name='cheetah_l', root='~/.insightface', config=None, providers=None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError('name must be a nonempty model package name')
        if not isinstance(root, (str, Path)) or not str(root).strip():
            raise ValueError('root must be a nonempty path')
        if config is not None and not isinstance(config, PersonConfig):
            raise TypeError('config must be PersonConfig')
        if providers is not None and (not isinstance(providers, (list, tuple)) or not providers):
            raise ValueError('providers must be a nonempty sequence')
        self.name, self.root = name, str(root)
        self.config = config or PersonConfig()
        self.providers = [tuple(provider) if isinstance(provider, list) else provider
                          for provider in providers] if providers is not None else ['CPUExecutionProvider']
        self._lock = RLock()
        self._prepared = self._closed = False
        self._owner, self._reference_version = object(), 0
        self._face_references = self._body_references = None
        self.detector = self.reid = self.face_detector = self.face_recognizer = None
        self._face_input_size = None
        self.face_model_id = self.reid_model_id = None

    @classmethod
    def from_config(cls, source, **overrides):
        """Load constructor options from a JSON file or an explicit mapping."""
        directory = None
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser().resolve()
            directory = path.parent
            with path.open(encoding='utf-8') as handle:
                options = json.load(handle)
        else:
            options = source
        if not isinstance(options, dict):
            raise ValueError('configuration must be a JSON object')
        options = dict(options)
        allowed = {'name', 'root', 'providers', 'config'}
        if (options.keys() | overrides.keys()) - allowed:
            raise ValueError('unknown PersonAnalysis configuration option')
        if directory is not None and 'root' in options and 'root' not in overrides:
            root = Path(options['root']).expanduser()
            options['root'] = str(root if root.is_absolute() else directory / root)
        options.update(overrides)
        config = options.get('config')
        if isinstance(config, dict):
            options['config'] = PersonConfig(**config)
        return cls(**options)

    def prepare(self):
        """Load the selected package once. get/register also prepare on first use."""
        with self._lock:
            self._check_open()
            if self._prepared:
                return
            models = prepare_models(self.name, self.root, self.providers, threads=self.config.cpu_threads,
                                    body_det_size=self.config.body_det_size, face_det_size=self.config.face_det_size)
            for name in ('detector', 'reid', 'face_detector', 'face_recognizer', 'face_model_id', 'reid_model_id'):
                setattr(self, name, models[name])
            self._face_input_size = tuple(self.face_detector.input_size)
            config = self.config
            common = dict(initial_capacity=config.reference_capacity,
                          duplicate_threshold=config.duplicate_similarity_threshold)
            self._face_references = ReferenceMatrix(int(self.face_recognizer.output_shape[-1]),
                self.face_model_id, **common)
            self._body_references = ReferenceMatrix(int(self.reid.descriptor['outputs'][0]['shape'][-1]),
                self.reid_model_id, max_samples=config.max_body_samples, **common)
            self._prepared = True

    def _check_open(self):
        if self._closed:
            raise RuntimeError('PersonAnalysis is closed')

    def _faces(self, image, min_size):
        fixed = getattr(self.face_detector, 'static_input_size', None)
        if fixed is not None and tuple(fixed) != self._face_input_size:
            raise ValueError(f'the face detector must support the configured input size {self._face_input_size}')
        boxes, landmarks = self.face_detector.detect(image, input_size=self._face_input_size, max_num=0, metric='default')
        faces = []
        height, width = image.shape[:2]
        for index, row in enumerate(boxes):
            if (len(row) != 5 or not np.isfinite(row).all() or not 0 <= row[4] <= 1 or
                    np.any(row[2:4] <= row[:2])):
                raise RuntimeError('face detector returned an invalid box')
            bbox = np.clip(row[:4], [0, 0, 0, 0], [width, height, width, height]).astype(np.float32)
            if np.any(bbox[2:] <= bbox[:2]):
                continue
            kps = None if landmarks is None else np.asarray(landmarks[index], np.float32).copy()
            face = Face(bbox, float(row[4]), kps)
            if kps is None or kps.shape != (5, 2) or not np.isfinite(kps).all():
                face._reason = 'invalid_landmarks'
            elif min(bbox[2:] - bbox[:2]) < min_size:
                face._reason = 'face_too_small'
            elif face.det_score < self.config.face_min_score:
                face._reason = 'low_face_score'
            faces.append(face)
        # Deduplicate before recognition, preferring a usable face sample.
        unique = []
        for face in sorted(faces, key=lambda item: (item._reason is not None, -item.det_score)):
            if all(_iou(face.bbox, previous.bbox) <= .5 for previous in unique):
                unique.append(face)
        for face in unique:
            if face._reason is not None:
                continue
            native = NativeFace(bbox=face.bbox.copy(), kps=face.kps.copy(), det_score=face.det_score)
            feature = self.face_recognizer.get(image, native)
            try:
                face.embedding = normalize_feature(feature if feature is not None else native.embedding)
                if face.embedding.size != self._face_references.dimension:
                    raise ValueError('face embedding dimension mismatch')
            except ValueError:
                face.embedding, face._reason = None, 'invalid_embedding'
        return unique

    def get(self, image):
        """Return independent single-frame observations in original image coordinates."""
        with self._lock:
            self.prepare()
            validate_image(image)
            rows = np.asarray(self.detector.detect(image), dtype=np.float32)
            if not rows.size:
                rows = np.empty((0, 5), np.float32)
            if rows.ndim != 2 or rows.shape[1] != 5 or not np.isfinite(rows).all():
                raise RuntimeError('person detector must return finite Nx5 boxes')
            height, width = image.shape[:2]
            persons = []
            for row in rows:
                if not 0 <= row[4] <= 1 or np.any(row[2:4] <= row[:2]):
                    raise RuntimeError('person detector returned an invalid box')
                bbox = np.clip(row[:4], [0, 0, 0, 0], [width, height, width, height]).astype(np.float32)
                if np.any(bbox[2:] <= bbox[:2]):
                    continue
                person = Person(body_bbox=bbox, det_score=float(row[4]))
                if min(bbox[2:] - bbox[:2]) >= self.config.body_min_size and row[4] >= self.config.body_min_score:
                    x1, y1 = np.floor(bbox[:2]).astype(int)
                    x2, y2 = np.ceil(bbox[2:]).astype(int)
                    person.reid_feature = self.get_reid(image[y1:y2, x1:x2])
                persons.append(person)
            faces = self._faces(image, self.config.face_min_size)
            unbound = _associate_faces(persons, faces, self.config.face_body_margin)
            persons.extend(Person(face=face) for face in unbound)
            for person in persons:
                person._face_model_id = self.face_model_id
                person._reid_model_id = self.reid_model_id
            return persons

    def get_reid(self, crop):
        """Embed one already cropped person; no additional detector is run."""
        with self._lock:
            self.prepare()
            validate_image(crop)
            if min(crop.shape[:2]) < self.config.body_min_size:
                raise ValueError('body_too_small')
            feature = normalize_feature(self.reid.embed(crop))
            if feature.size != self._body_references.dimension:
                raise RuntimeError('body embedding dimension mismatch')
            return feature

    @staticmethod
    def _person_id(person_id):
        if (isinstance(person_id, bool) or not isinstance(person_id, (str, int)) or
                isinstance(person_id, str) and not person_id.strip() or
                isinstance(person_id, int) and person_id <= 0):
            raise ValueError('person_id must be a positive integer or a nonempty string')

    def register(self, person_id, images, *, body_images=None):
        """Add labeled face photos and optional already cropped, labeled bodies.

        Each face photo must contain exactly one detected face. Explicit body
        references are trusted user labels; update() never learns from a body match.
        """
        with self._lock:
            self.prepare()
            self._person_id(person_id)
            face_images, bodies = _images(images), _images(body_images)
            if not face_images and not bodies:
                raise ValueError('provide at least one reference image')
            accepted, rejected = 0, []
            for kind, values in (('face', face_images), ('body', bodies)):
                matrix = self._face_references if kind == 'face' else self._body_references
                for index, value in enumerate(values):
                    try:
                        image = _read_image(value)
                        if kind == 'face':
                            faces = self._faces(image, self.config.face_registration_min_size)
                            if len(faces) != 1:
                                raise ValueError('no_face' if not faces else 'multiple_faces')
                            if faces[0].embedding is None:
                                raise ValueError(faces[0]._reason)
                            feature = faces[0].embedding
                        else:
                            feature = self.get_reid(image)
                        action, reason = matrix.add(person_id, feature, manual=True)
                        if action == 'skipped':
                            raise ValueError(reason)
                        accepted += 1
                        self._reference_version += 1
                    except ValueError as error:
                        rejected.append(dict(index=index, kind=kind, reason=str(error)))
            return RegistrationResult(person_id, accepted, rejected)

    def _validate_observation(self, person):
        if not isinstance(person, Person):
            raise TypeError('match expects observations returned by PersonAnalysis.get')
        if person.face is not None and person.face.embedding is not None:
            if person._face_model_id != self.face_model_id:
                raise ValueError('face feature model mismatch')
        if person.reid_feature is not None and person._reid_model_id != self.reid_model_id:
            raise ValueError('body feature model mismatch')

    def match(self, observations):
        """Read references without modifying them. Face evidence has priority."""
        with self._lock:
            self.prepare()
            observations = list(observations)
            for person in observations:
                self._validate_observation(person)
            results = []
            for person in observations:
                candidates = [('face', None if person.face is None else person.face.embedding,
                               self._face_references, self.config.face_similarity_threshold, self.config.face_margin),
                              ('body', person.reid_feature, self._body_references,
                               self.config.reid_similarity_threshold, self.config.reid_margin)]
                result = MatchResult(person)
                for kind, feature, matrix, threshold, margin in candidates:
                    if feature is None:
                        continue
                    candidate = matrix.match(feature)
                    if candidate and candidate['similarity'] >= threshold and candidate['margin'] >= margin:
                        result = MatchResult(person, candidate['person_id'], kind, candidate['similarity'])
                        break
                object.__setattr__(result, '_owner', self._owner)
                object.__setattr__(result, '_version', self._reference_version)
                if (result.matched_by == 'face' and person._association_valid and
                        person.body_bbox is not None and person.reid_feature is not None):
                    body = normalize_feature(person.reid_feature).copy()
                    if body.size != self._body_references.dimension:
                        raise ValueError('body embedding dimension mismatch')
                    body.setflags(write=False)
                    object.__setattr__(result, '_body_feature', body)
                results.append(result)
            return results

    def update(self, matches):
        """Learn at most one body sample from each explicitly face-matched result."""
        with self._lock:
            self.prepare()
            matches = list(matches)
            for result in matches:
                if not isinstance(result, MatchResult) or result._owner is not self._owner:
                    raise ValueError('match result belongs to another PersonAnalysis instance')
                if result._version != self._reference_version:
                    raise ValueError('references changed; match the observations again before update')
            counts = dict(added=0, replaced=0, skipped=0)
            for result in matches:
                if result._used or result._body_feature is None:
                    counts['skipped'] += 1
                    continue
                action, _ = self._body_references.add(result.person_id, result._body_feature, manual=False)
                counts[action] += 1
                object.__setattr__(result, '_used', True)
            return UpdateResult(**counts)

    def remove_person(self, person_id):
        with self._lock:
            self.prepare()
            self._person_id(person_id)
            count = self._face_references.remove(person_id) + self._body_references.remove(person_id)
            if count:
                self._reference_version += 1
            return count

    def clear_references(self):
        with self._lock:
            self.prepare()
            self._face_references.clear()
            self._body_references.clear()
            self._reference_version += 1

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._reference_version += 1
            self._face_references = self._body_references = None
            self.detector = self.reid = self.face_detector = self.face_recognizer = None

    def __enter__(self):
        self.prepare()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
