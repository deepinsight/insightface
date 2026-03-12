# -*- coding: utf-8 -*-
"""Face visualization utilities.

This module provides visualization functions for face analysis results,
decoupled from the core business logic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from .common import Face


class FaceVisualizer:
    """Visualizer for face analysis results.
    
    This class provides methods to draw bounding boxes, keypoints,
    and other face attributes on images.
    """
    
    COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
    COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)
    COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)
    COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
    
    def __init__(
        self,
        box_color: Tuple[int, int, int] = COLOR_RED,
        kps_color: Tuple[int, int, int] = COLOR_RED,
        kps_highlight_color: Tuple[int, int, int] = COLOR_GREEN,
        text_color: Tuple[int, int, int] = COLOR_GREEN,
        box_thickness: int = 2,
        kps_radius: int = 2,
        text_scale: float = 0.7,
        text_thickness: int = 1,
    ) -> None:
        self.box_color = box_color
        self.kps_color = kps_color
        self.kps_highlight_color = kps_highlight_color
        self.text_color = text_color
        self.box_thickness = box_thickness
        self.kps_radius = kps_radius
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self._cv2 = None
    
    def _get_cv2(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2
    
    def draw_faces(
        self,
        img: np.ndarray,
        faces: List[Face],
        draw_box: bool = True,
        draw_kps: bool = True,
        draw_gender_age: bool = True,
    ) -> np.ndarray:
        cv2 = self._get_cv2()
        result = img.copy()
        
        for face in faces:
            if draw_box:
                self._draw_bbox(result, face)
            if draw_kps:
                self._draw_kps(result, face)
            if draw_gender_age:
                self._draw_gender_age(result, face)
        
        return result
    
    def _draw_bbox(self, img: np.ndarray, face: Face) -> None:
        cv2 = self._get_cv2()
        bbox = face.get_bbox()
        if bbox is None:
            return
        box = bbox.astype(int)
        if len(box) >= 4:
            cv2.rectangle(
                img,
                (box[0], box[1]),
                (box[2], box[3]),
                self.box_color,
                self.box_thickness,
            )
    
    def _draw_kps(self, img: np.ndarray, face: Face) -> None:
        cv2 = self._get_cv2()
        kps = face.get_kps()
        if kps is None:
            return
        kps_int = kps.astype(int)
        for idx, kp in enumerate(kps_int):
            color = self.kps_color
            if idx == 0 or idx == 3:
                color = self.kps_highlight_color
            cv2.circle(
                img,
                (kp[0], kp[1]),
                self.kps_radius,
                color,
                self.kps_radius,
            )
    
    def _draw_gender_age(self, img: np.ndarray, face: Face) -> None:
        cv2 = self._get_cv2()
        gender, age = face.get_gender_age()
        if gender is None or age is None:
            return
        bbox = face.get_bbox()
        if bbox is None:
            return
        box = bbox.astype(int)
        sex = 'M' if gender == 1 else 'F'
        text = f'{sex},{age}'
        cv2.putText(
            img,
            text,
            (box[0] - 1, box[1] - 4),
            cv2.FONT_HERSHEY_COMPLEX,
            self.text_scale,
            self.text_color,
            self.text_thickness,
        )


def draw_faces(
    img: np.ndarray,
    faces: List[Face],
    box_color: Tuple[int, int, int] = (0, 0, 255),
    kps_color: Tuple[int, int, int] = (0, 0, 255),
    kps_highlight_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (0, 255, 0),
    draw_box: bool = True,
    draw_kps: bool = True,
    draw_gender_age: bool = True,
) -> np.ndarray:
    visualizer = FaceVisualizer(
        box_color=box_color,
        kps_color=kps_color,
        kps_highlight_color=kps_highlight_color,
        text_color=text_color,
    )
    return visualizer.draw_faces(
        img,
        faces,
        draw_box=draw_box,
        draw_kps=draw_kps,
        draw_gender_age=draw_gender_age,
    )
