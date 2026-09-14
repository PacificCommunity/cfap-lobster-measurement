"""
===============================================================================
Lobster Measurement and Calibration System
===============================================================================

Description
-----------
Automated lobster morphometric measurement pipeline using computer vision,
YOLO object detection, calibration board rectification, and optional
coin-based measurement validation.

The system performs:

    • Calibration board detection and perspective correction
    • Automatic board orientation detection
    • Coin detection and extraction
    • Lobster landmark detection (P0, P1, P2)
    • Carapace length (CL) measurement
    • Total length (TL) measurement
    • Annotated image generation
    • JSON output generation for downstream analysis

Credits
-------
Author:
    Franck Magron

Programme:
    Coastal Fisheries and Aquaculture Programme (CFAP)

Division:
    Fisheries, Aquaculture and Marine Ecosystems (FAME) Division

Institution:
    Pacific Community (SPC)

Purpose
-------
Developed to support fisheries monitoring, stock assessment,
and digital fisheries data collection workflows through the
automated extraction of lobster measurements from calibrated
photographs.

Copyright
---------
© Pacific Community (SPC)

This software was developed as part of activities of the
Coastal Fisheries and Aquaculture Programme within the
Fisheries, Aquaculture and Marine Ecosystems (FAME) Division.

Disclaimer
----------
This software is provided "as is" without warranty of any kind.
Users are responsible for validating measurements before use
in scientific, regulatory, or management applications.

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

CALIBRATION_ZOOM_FACTOR = 1.0
MEASUREMENT_ZOOM_FACTOR = 4.0
EXPORT_SCALE = 2

BOARD_WIDTH = 1590
BOARD_HEIGHT = 380 
CALIBRATION_BORDER = 200
DESTINATION_BORDER_Y = 50


@dataclass
class AppOptions:
    input_path: str
    calibrated_output_folder: str
    json_output_folder: str
    models_root: str
    coin_output_folder: str | None
    coin_diameter_mm: float
    gpu_index: int


@dataclass
class CalibrationPointOutput:
    index: int
    label: str
    confidence: float
    image_x: float
    image_y: float
    real_x: float
    real_y: float


@dataclass
class CalibrationOutput:
    success: bool
    message: str
    orientation: str | None = None
    step: int = 0
    residual: float = 0.0
    zoom_factor: float = 0.0
    calibration_border: int = 0
    destination_border_y: int = 0
    output_width: int = 0
    output_height: int = 0
    detected_points: list[CalibrationPointOutput] = field(default_factory=list)

    @staticmethod
    def failed(message: str) -> "CalibrationOutput":
        return CalibrationOutput(success=False, message=message)


@dataclass
class BoundingBoxOutput:
    x: int
    y: int
    width: int
    height: int


@dataclass
class CoinDetectionResult:
    detected: bool = False
    confidence: float = 0.0
    bounding_box: BoundingBoxOutput | None = None
    coin_bounding_box_width_mm: float = 0.0
    coin_bounding_box_height_mm: float = 0.0
    coin_based_size_correction: float | None = None
    extracted_coin_file: str | None = None


@dataclass
class DetectedPointSimple:
    x: float
    y: float
    confidence: float


@dataclass
class LobsterMeasurementResult:
    detected: bool = False
    carapace_length_mm: float | None = None
    total_length_mm: float | None = None
    coin_based_carapace_length_mm: float | None = None
    coin_based_total_length_mm: float | None = None
    coin_correction_factor: float | None = None
    p0: DetectedPointSimple | None = None
    p1: DetectedPointSimple | None = None
    p2: DetectedPointSimple | None = None


@dataclass
class ImageProcessingOutput:
    input_file: str
    calibrated_image_file: str | None = None
    calibration: CalibrationOutput | None = None
    coin: CoinDetectionResult | None = None
    lobster: LobsterMeasurementResult | None = None
    warnings: list[str] | None = None
    error: str | None = None


@dataclass
class YoloItem:
    type: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass
class CalibrationClass:
    index: int
    label: str
    real_x: int
    real_y: int
    image_x: float = 0.0
    image_y: float = 0.0
    confidence: float = 0.0


@dataclass
class RotatedImage:
    filename: str
    board_points: list[CalibrationClass] = field(default_factory=list)
    model_points_after_step: list[CalibrationClass] = field(default_factory=list)
    likelihood: int = 0
    orientation: str = ""


@dataclass
class StepEvaluation:
    residuals: float
    temp_file_path: str
    step: int
    board_points: list[CalibrationClass] | None = None


@dataclass
class AlignedPoints:
    objects: list[YoloItem]
    std_dev: float
    std_dev_distances: float


class YoloDetector:
    def __init__(self, cfg_path: str, weights_path: str, names_path: str, gpu_index: int) -> None:
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        self.class_names = self._read_class_names(names_path)

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            if gpu_index >= 0:
                cv2.cuda.setDevice(gpu_index)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def _read_class_names(path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    def detect(self, image_path: str, confidence_threshold: float = 0.25, nms_threshold: float = 0.45) -> list[YoloItem]:
        image = cv2.imread(image_path)
        if image is None:
            return []

        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < confidence_threshold:
                    continue

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes:
            return []

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        if len(indexes) == 0:
            return []

        selected: list[YoloItem] = []
        for idx in indexes.flatten():
            class_name = self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else str(class_ids[idx])
            x, y, w, h = boxes[idx]
            selected.append(
                YoloItem(
                    type=class_name,
                    confidence=confidences[idx],
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                )
            )

        return selected


def main() -> int:
    try:
        options = parse_args(sys.argv[1:])
        if options is None:
            print_usage()
            return 1

        os.makedirs(options.calibrated_output_folder, exist_ok=True)
        os.makedirs(options.json_output_folder, exist_ok=True)
        if options.coin_output_folder and options.coin_output_folder.strip():
            os.makedirs(options.coin_output_folder, exist_ok=True)

        input_files = list(get_input_files(options.input_path))
        if len(input_files) == 0:
            print("No JPG files found to process.")
            return 2

        ruler_model = build_model_configuration(options.models_root, "yolov4-rulerplus.cfg", "yolov4-rulerplus_last.weights", "rulerplus.names")
        coin_model = build_model_configuration(options.models_root, "yolov4-coin.cfg", "yolov4-coin_final.weights", "coin.names")
        lobster_model = build_model_configuration(options.models_root, "yolov4-lobster_lengths.cfg", "yolov4-lobster_lengths_last.weights", "lobster_lengths.names")

        ruler_yolo = YoloDetector(ruler_model[0], ruler_model[1], ruler_model[2], options.gpu_index)
        coin_yolo = YoloDetector(coin_model[0], coin_model[1], coin_model[2], options.gpu_index)
        lobster_yolo = YoloDetector(lobster_model[0], lobster_model[1], lobster_model[2], options.gpu_index)

        for idx, input_file in enumerate(input_files, start=1):
            print(f"[{idx}/{len(input_files)}] {Path(input_file).name}")
            output = ImageProcessingOutput(input_file=input_file, warnings=[])

            try:
                calibrated_file = str(Path(options.calibrated_output_folder) / Path(input_file).name)
                output.calibrated_image_file = calibrated_file

                calibration = calibrate_board_image(ruler_yolo, input_file, calibrated_file)
                output.calibration = calibration

                if not calibration.success:
                    output.error = "Calibration failed: not enough board points or residual too high."
                    write_output_json(options.json_output_folder, input_file, output)
                    continue

                coin_result = detect_and_extract_coin(
                    coin_yolo,
                    calibrated_file,
                    options.coin_output_folder,
                    options.coin_diameter_mm,
                )
                output.coin = coin_result

                lobster_result = measure_lobster(
                    lobster_yolo,
                    calibrated_file,
                    options.coin_diameter_mm,
                    coin_result,
                )
                output.lobster = lobster_result

                draw_measurements_on_image(
                    calibrated_file,
                    lobster_result,
                    coin_result,
                )

                write_output_json(options.json_output_folder, input_file, output)
            except Exception as ex:
                output.error = str(ex)
                write_output_json(options.json_output_folder, input_file, output)

        return 0
    except Exception as ex:
        print(ex)
        return 99


def parse_args(args: list[str]) -> AppOptions | None:
    if len(args) < 3:
        return None

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("inputJpgOrFolder")
    parser.add_argument("calibratedOutputFolder")
    parser.add_argument("jsonOutputFolder")
    parser.add_argument("--models", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--coinDiameterMm", type=float, default=24.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--coinOutputFolder", default=None)

    parsed, _ = parser.parse_known_args(args)

    return AppOptions(
        input_path=parsed.inputJpgOrFolder,
        calibrated_output_folder=parsed.calibratedOutputFolder,
        json_output_folder=parsed.jsonOutputFolder,
        models_root=parsed.models,
        coin_output_folder=parsed.coinOutputFolder,
        coin_diameter_mm=parsed.coinDiameterMm,
        gpu_index=parsed.gpu,
    )


def print_usage() -> None:
    print("Usage:")
    print(
        "  python measure_lobster.py <inputJpgOrFolder> <calibratedOutputFolder> <jsonOutputFolder> "
        "[--models <modelsRoot>] [--coinDiameterMm <diameter>] [--gpu <index>] [--coinOutputFolder <folder>]"
    )


def get_input_files(input_path: str) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in (".jpg", ".jpeg"):
            return [str(path.resolve())]
        return []

    if not path.is_dir():
        return []

    files = []
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in (".jpg", ".jpeg"):
            files.append(str(file))

    return files


def build_model_configuration(models_root: str, cfg_name: str, weights_name: str, names_name: str) -> tuple[str, str, str]:
    return (
        resolve_model_path(models_root, cfg_name),
        resolve_model_path(models_root, weights_name),
        resolve_model_path(models_root, names_name),
    )


def resolve_model_path(models_root: str, file_name: str) -> str:
    candidates = [
        Path(models_root) / file_name,
        Path(models_root) / "configs" / file_name,
        Path(models_root) / "models" / file_name,
        Path(models_root) / "names" / file_name,
        Path(models_root) / "cfg" / file_name,
        Path(models_root) / "weights" / file_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Model file not found: {file_name}")


def measure_lobster(
    lobster_yolo: YoloDetector,
    calibrated_board_file: str,
    coin_diameter_mm: float,
    coin: CoinDetectionResult,
) -> LobsterMeasurementResult:
    _ = coin_diameter_mm
    result = LobsterMeasurementResult()

    items = sorted(lobster_yolo.detect(calibrated_board_file), key=lambda p: p.confidence, reverse=True)

    p0 = next((it for it in items if it.type == "p0"), None)
    p1 = next((it for it in items if it.type == "p1"), None)
    p2 = next((it for it in items if it.type == "p2"), None)

    if p0 is None or p1 is None or p2 is None:
        result.detected = False
        return result

    x1 = p0.x + p0.width / 2.0
    y1 = p0.y + p0.height / 2.0
    x2 = p1.x + p1.width / 2.0
    y2 = p1.y + p1.height / 2.0
    x3 = p2.x + p2.width / 2.0
    y3 = p2.y + p2.height / 2.0

    cl_mm = round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / MEASUREMENT_ZOOM_FACTOR, 0)
    tl_mm = round(math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2) / MEASUREMENT_ZOOM_FACTOR, 0)

    result.detected = True
    result.carapace_length_mm = cl_mm
    result.total_length_mm = tl_mm
    result.p0 = DetectedPointSimple(x=x1, y=y1, confidence=p0.confidence)
    result.p1 = DetectedPointSimple(x=x2, y=y2, confidence=p1.confidence)
    result.p2 = DetectedPointSimple(x=x3, y=y3, confidence=p2.confidence)

    if coin.detected and coin.coin_based_size_correction is not None:
        result.coin_based_carapace_length_mm = round(cl_mm * coin.coin_based_size_correction, 1)
        result.coin_based_total_length_mm = round(tl_mm * coin.coin_based_size_correction, 1)
        result.coin_correction_factor = coin.coin_based_size_correction

    return result


def detect_and_extract_coin(
    coin_yolo: YoloDetector,
    board_file: str,
    coin_output_folder: str | None,
    coin_diameter_mm: float,
) -> CoinDetectionResult:
    result = CoinDetectionResult()

    detections = sorted(coin_yolo.detect(board_file), key=lambda p: p.confidence, reverse=True)
    item = detections[0] if detections else None

    if item is None:
        result.detected = False
        return result

    ratio = max(item.width, item.height) / min(item.width, item.height) if min(item.width, item.height) > 0 else float("inf")
    if ratio > 1.2:
        result.detected = False
        return result

    result.detected = True
    result.confidence = item.confidence
    result.bounding_box = BoundingBoxOutput(x=item.x, y=item.y, width=item.width, height=item.height)
    result.coin_bounding_box_width_mm = item.width / MEASUREMENT_ZOOM_FACTOR
    result.coin_bounding_box_height_mm = item.height / MEASUREMENT_ZOOM_FACTOR
    result.coin_based_size_correction = coin_diameter_mm / result.coin_bounding_box_width_mm if result.coin_bounding_box_width_mm > 0 else None

    if coin_output_folder and coin_output_folder.strip():
        dst_file = str(Path(coin_output_folder) / Path(board_file).name)
        extract_coin(board_file, dst_file, item, MEASUREMENT_ZOOM_FACTOR)
        result.extracted_coin_file = dst_file

    return result


def calibrate_board_image(
    yolo_wrapper: YoloDetector,
    input_file: str,
    calibrated_output_file: str,
) -> CalibrationOutput:
    rotated_images: list[RotatedImage] = []
    scores: list[StepEvaluation] = []

    try:
        rotated_images = generate_rotated_images(input_file)

        # Match ReadMarketBoardInverts:
        # detect points on all four rotations, but do not run a calibration
        # step on every rotation before selecting the best one.
        for image in rotated_images:
            items = yolo_wrapper.detect(image.filename)

            image.board_points = get_board2_points(
                items,
                CALIBRATION_BORDER,
                clean=True,
                zoom_factor=1.0,
            )

        # Match:
        # OrderByDescending(rt => rt.ModelPoints.Count)
        #     .ThenBy(rt => rt.Temporary)
        #
        # The Python likelihood field preserves the preferred rotation order.
        best_image = sorted(
            rotated_images,
            key=lambda image: (
                -len(image.board_points),
                image.likelihood,
            ),
        )[0]

        model_points = best_image.board_points
        filename = best_image.filename

        if len(model_points) < 4:
            return CalibrationOutput.failed(
                "Insufficient board points on all rotations."
            )

        step = 1

        # ReadMarketBoardInverts omits zoomFactor here, so it defaults to 1.
        model_points_after_step = process_step_board2(
            yolo_wrapper=yolo_wrapper,
            src_file_path=filename,
            border=CALIBRATION_BORDER,
            src_model_points=model_points,
            scores=scores,
            step=step,
            zoom_factor=CALIBRATION_ZOOM_FACTOR,
        )

        if len(model_points_after_step) > 4:
            if not scores:
                return CalibrationOutput.failed(
                    "First calibration step did not produce a valid score."
                )

            previous_score = scores[-1]
            step += 1

            model_points_after_step = process_step_board2(
                yolo_wrapper=yolo_wrapper,
                src_file_path=previous_score.temp_file_path,
                border=CALIBRATION_BORDER,
                src_model_points=model_points_after_step,
                scores=scores,
                step=step,
                zoom_factor=CALIBRATION_ZOOM_FACTOR,
            )

            # Direct translation of the C# condition:
            #
            # while (
            #     step < 3 ||
            #     (
            #         modelPointsAfterStep.Count >= 4 &&
            #         scores.Last().Residuals + 1 <
            #             previousScore.Residuals &&
            #         step <= 5
            #     )
            # )
            while (
                step < 3
                or (
                    len(model_points_after_step) >= 4
                    and len(scores) > 0
                    and scores[-1].residuals + 1
                    < previous_score.residuals
                    and step <= 5
                )
            ):
                if not scores:
                    break

                previous_score = scores[-1]
                step += 1

                model_points_after_step = process_step_board2(
                    yolo_wrapper=yolo_wrapper,
                    src_file_path=previous_score.temp_file_path,
                    border=CALIBRATION_BORDER,
                    src_model_points=model_points_after_step,
                    scores=scores,
                    step=step,
                    zoom_factor=CALIBRATION_ZOOM_FACTOR,
                )

        if len(scores) == 0:
            return CalibrationOutput.failed(
                "No calibration step succeeded."
            )

        best_step = min(
            scores,
            key=lambda score: score.residuals,
        )

        if best_step.residuals >= 100:
            return CalibrationOutput.failed(
                f"Residual too high: {best_step.residuals:.2f}"
            )

        # Match System.Drawing.Graphics.DrawImage source and destination
        # rectangles. Both are 1590 x 480, so no resize is required.
        source_x = CALIBRATION_BORDER - 100
        source_y = CALIBRATION_BORDER - DESTINATION_BORDER_Y
        output_width=int(BOARD_WIDTH * EXPORT_SCALE),
        output_height=int(
            (BOARD_HEIGHT + DESTINATION_BORDER_Y * 2)
            * EXPORT_SCALE
        ),

        with Image.open(best_step.temp_file_path) as temp_bitmap:

            crop_rect = (
                int((CALIBRATION_BORDER - 100) * EXPORT_SCALE),
                int((CALIBRATION_BORDER - DESTINATION_BORDER_Y)
                    * EXPORT_SCALE),

                int((CALIBRATION_BORDER - 100 + BOARD_WIDTH)
                    * EXPORT_SCALE),

                int(
                    (
                        CALIBRATION_BORDER
                        - DESTINATION_BORDER_Y
                        + BOARD_HEIGHT
                        + DESTINATION_BORDER_Y * 2
                    ) * EXPORT_SCALE
                ),
            )
            cropped = temp_bitmap.crop(crop_rect)

            dest_bitmap = temp_bitmap.crop(crop_rect)

            dest_bitmap.save(
                calibrated_output_file,
                format="JPEG",
                quality=95,
            )

        return CalibrationOutput(
            success=True,
            message="Calibration succeeded",
            orientation=best_image.orientation,
            step=best_step.step,
            residual=best_step.residuals,
            zoom_factor=CALIBRATION_ZOOM_FACTOR,
            calibration_border=CALIBRATION_BORDER,
            destination_border_y=DESTINATION_BORDER_Y,
            output_width=output_width,
            output_height=output_height,
            detected_points=[
                CalibrationPointOutput(
                    label=point.label,
                    index=point.index,
                    confidence=point.confidence,
                    image_x=point.image_x,
                    image_y=point.image_y,
                    real_x=point.real_x,
                    real_y=point.real_y,
                )
                for point in (best_step.board_points or [])
            ],
        )

    finally:
        for image in rotated_images:
            try_delete_file(image.filename)

        for score in scores:
            try_delete_file(score.temp_file_path)

def process_step_board2(
    yolo_wrapper: YoloDetector,
    src_file_path: str,
    border: int,
    src_model_points: list[CalibrationClass],
    scores: list[StepEvaluation],
    step: int,
    zoom_factor: float = 1,
) -> list[CalibrationClass]:
    fd, dest_file_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    save_unwarped_image(src_file_path, dest_file_path, src_model_points, border, zoom_factor)

    items_after_step = yolo_wrapper.detect(dest_file_path)
    model_points_after_step = get_board2_points(items_after_step, border, True, zoom_factor)
    if len(model_points_after_step) >= 4:
        board_points = get_board2_points(items_after_step, border, True, zoom_factor)
        evaluation = StepEvaluation(eval_detected_points(board_points), dest_file_path, step, board_points)
        scores.append(evaluation)

    return model_points_after_step


def save_unwarped_image(
    src_file_path: str,
    dest_file_path: str,
    model_points: list[CalibrationClass],
    border: int,
    zoom_factor: float = 1,
) -> None:
    warp_mat = get_warp_mat(model_points)
    input_mat = cv2.imread(src_file_path, cv2.IMREAD_COLOR)
    if input_mat is None:
        raise ValueError(f"Unable to read image: {src_file_path}")

    cut_image_portion = cv2.warpPerspective(
        input_mat,
        warp_mat,
        (int(EXPORT_SCALE * zoom_factor * (BOARD_WIDTH + border * 2)), int(EXPORT_SCALE * zoom_factor * (BOARD_HEIGHT + border * 2))),
        flags=cv2.INTER_CUBIC,
    )

    cv2.imwrite(dest_file_path, cut_image_portion, [cv2.IMWRITE_JPEG_QUALITY, 95])


def get_warp_mat(model_points: list[CalibrationClass]) -> np.ndarray:
    srcs = np.array([[p.image_x, p.image_y] for p in model_points], dtype=np.float32)
    dsts = np.array([[p.real_x, p.real_y] for p in model_points], dtype=np.float32)

    warp_mat, _ = cv2.findHomography(srcs, dsts, method=0)
    if warp_mat is None:
        raise ValueError("Unable to compute homography matrix.")

    return warp_mat


def get_board2_points(
    items: list[YoloItem],
    border: int,
    clean: bool = True,
    zoom_factor: float = 1.0,
) -> list[CalibrationClass]:
    objects = [
        item
        for item in items
        if safe_type_int(item.type) is not None
        and safe_type_int(item.type) < 14
    ]

    if clean:
        top_objects = find_aligned(
            [
                item
                for item in objects
                if safe_type_int(item.type) is not None
                and safe_type_int(item.type) < 6
            ]
        ).objects

        bottom_objects = find_aligned(
            [
                item
                for item in objects
                if safe_type_int(item.type) is not None
                and 7 <= safe_type_int(item.type) < 13
            ]
        ).objects

        if (
            len(top_objects) < 2
            or len(bottom_objects) < 2
            or max(item.y for item in top_objects)
            > min(item.y for item in bottom_objects)
        ):
            return []

        objects = [*top_objects, *bottom_objects]

    classes = [
        CalibrationClass(0, "T10", 200, 0),
        CalibrationClass(1, "T20", 400, 0),
        CalibrationClass(2, "T30", 600, 0),
        CalibrationClass(3, "T40", 800, 0),
        CalibrationClass(4, "T50", 1000, 0),
        CalibrationClass(5, "T60", 1200, 0),
        CalibrationClass(6, "T70", 1400, 0),
        CalibrationClass(7, "B10", 200, 379),
        CalibrationClass(8, "B20", 400, 379),
        CalibrationClass(9, "B30", 600, 379),
        CalibrationClass(10, "B40", 800, 379),
        CalibrationClass(11, "B50", 1000, 379),
        CalibrationClass(12, "B60", 1200, 379),
        CalibrationClass(13, "B70", 1400, 379),
    ]

    # Match the first C# loop.
    for calibration_class in classes:
        calibration_class.real_x = int(
            calibration_class.real_x * zoom_factor * EXPORT_SCALE
        )
        calibration_class.real_y = int(
            calibration_class.real_y * zoom_factor * EXPORT_SCALE
        )

    for calibration_class in list(classes):
        matching_objects = [
            item
            for item in objects
            if safe_type_int(item.type)
            == calibration_class.index
        ]

        main_object = (
            max(matching_objects, key=lambda item: item.confidence)
            if matching_objects
            else None
        )

        if main_object is None:
            classes.remove(calibration_class)
            continue

        calibration_class.confidence = main_object.confidence

        # Match the second C# loop. Border is added after zooming and is
        # deliberately not multiplied by zoom_factor.
        calibration_class.real_x += int(border * EXPORT_SCALE)
        calibration_class.real_y += int(border * EXPORT_SCALE)

        object_type = safe_type_int(main_object.type)

        if object_type is not None and object_type < 7:
            calibration_class.image_x = (
                main_object.x + main_object.width / 2.0
            )
            calibration_class.image_y = main_object.y
        else:
            calibration_class.image_x = (
                main_object.x + main_object.width / 2.0
            )
            calibration_class.image_y = (
                main_object.y + main_object.height
            )

    return classes

def safe_type_int(value: str) -> int | None:
    token = value.split(" ")[0]
    try:
        return int(token)
    except ValueError:
        return None


def find_aligned(items: list[YoloItem]) -> AlignedPoints:
    if len(items) <= 2:
        return AlignedPoints(items, 0, 0)

    new_list: list[YoloItem] = []
    labels = sorted(set(o.type for o in items))

    for label in labels:
        best = sorted((o for o in items if o.type == label), key=lambda o: o.confidence, reverse=True)[0]
        new_list.append(best)

    ordered = sorted(new_list, key=lambda o: safe_type_int(o.type.split(" ")[0]) or 0)

    tangents: list[float] = []
    distances: list[float] = []

    for i in range(len(ordered) - 1):
        for j in range(i + 1, len(ordered)):
            obj1 = ordered[i]
            obj2 = ordered[j]

            dy = (obj2.y + obj2.height / 2.0) - (obj1.y + obj1.height / 2.0)
            dx = (obj2.x + obj2.width / 2.0) - (obj1.x + obj1.width / 2.0)
            tan = math.copysign(math.inf, dy) if dx == 0 else dy / dx

            distances.append(calc_distance(obj1, obj2))
            tangents.append(tan)

    aligned = AlignedPoints(ordered, std_dev(tangents), std_dev(distances))

    if len(ordered) > 3:
        for i in range(len(ordered)):
            list2 = [ordered[j] for j in range(len(ordered)) if j != i]
            res = find_aligned(list2)
            if res.std_dev_distances < aligned.std_dev_distances:
                aligned = res

    return aligned


def calc_distance(obj1: YoloItem, obj2: YoloItem) -> float:
    idx1 = safe_type_int(obj1.type.split(" ")[0])
    idx2 = safe_type_int(obj2.type.split(" ")[0])
    if idx1 is None or idx2 is None:
        return float("inf")

    denominator = abs(idx2 - idx1)
    if denominator == 0:
        return float("inf")

    return (
        math.sqrt(
            ((obj2.x + obj2.width / 2.0) - (obj1.x + obj1.width / 2.0)) ** 2
            + ((obj2.y + obj2.height / 2.0) - (obj1.y + obj1.height / 2.0)) ** 2
        )
        / denominator
    )


def std_dev(values: list[float]) -> float:
    if len(values) == 0:
        return 0

    finite_values = [v for v in values if math.isfinite(v)]
    if len(finite_values) == 0:
        return 0

    avg = sum(finite_values) / len(finite_values)
    sum_sq = sum((d - avg) ** 2 for d in finite_values)
    return math.sqrt(sum_sq / len(finite_values))


def eval_detected_points(items: list[CalibrationClass]) -> float:
    score = 0.0
    for pclass in items:
        score += (pclass.image_x - pclass.real_x) ** 2 + (pclass.image_y - pclass.real_y) ** 2

    return score / len(items) if len(items) > 0 else float("inf")


def generate_rotated_images(filename: str) -> list[RotatedImage]:
    rotated_images = [RotatedImage(filename=create_temp_jpg()) for _ in range(4)]

    with Image.open(filename) as img:
        img2 = img.copy()
        img2.save(rotated_images[0].filename, format="JPEG", quality=95)

        img2 = img2.rotate(-90, expand=True)
        img2.save(rotated_images[1].filename, format="JPEG", quality=95)

        img2 = img2.rotate(-90, expand=True)
        img2.save(rotated_images[2].filename, format="JPEG", quality=95)

        img2 = img2.rotate(-90, expand=True)
        img2.save(rotated_images[3].filename, format="JPEG", quality=95)

    rotated_images[0].likelihood = 0
    rotated_images[1].likelihood = 2
    rotated_images[2].likelihood = 1
    rotated_images[3].likelihood = 3

    rotated_images[0].orientation = "Straight"
    rotated_images[1].orientation = "RotateLeft"
    rotated_images[2].orientation = "Rotate180"
    rotated_images[3].orientation = "Rotate270"

    return rotated_images


def extract_coin(src_file: str, dst_file: str, bbox: YoloItem, resolution: float) -> None:
    _ = resolution

    with Image.open(src_file) as img:
        obj_center_x = bbox.x + bbox.width / 2.0
        obj_center_y = bbox.y + bbox.height / 2.0
        obj_width = bbox.width * 1.2
        obj_height = bbox.height * 1.2

        src_width = max(obj_height, obj_width)
        scale = 150 / src_width if src_width > 0 else 1

        target_width = max(1, int(obj_width * scale))
        target_height = max(1, int(obj_height * scale))

        src_x = obj_center_x - obj_width / 2
        src_y = obj_center_y - obj_height / 2

        cropped = img.crop((src_x, src_y, src_x + obj_width, src_y + obj_height)).resize(
            (target_width, target_height), Image.Resampling.BILINEAR
        )

        draw = ImageDraw.Draw(cropped)
        delta_x = obj_width * scale * 0.1
        delta_y = obj_height * scale * 0.1
        draw.rectangle(
            (
                int(delta_x),
                int(delta_y),
                int(delta_x + (obj_width * scale / 1.2)),
                int(delta_y + (obj_height * scale / 1.2)),
            ),
            outline="red",
            width=1,
        )

        cropped.save(dst_file, format="JPEG", quality=95)


def write_output_json(output_folder: str, input_file: str, output: ImageProcessingOutput) -> None:
    json_file = Path(output_folder) / (Path(input_file).stem + ".json")
    payload = to_pascal_dict(output)
    json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def to_pascal_dict(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, list):
        return [to_pascal_dict(item) for item in obj]

    if hasattr(obj, "__dataclass_fields__"):
        raw = asdict(obj)
        output: dict[str, Any] = {}
        for key, value in raw.items():
            output[to_pascal_case(key)] = to_pascal_dict(value)
        return output

    if isinstance(obj, dict):
        return {to_pascal_case(key): to_pascal_dict(value) for key, value in obj.items()}

    return obj


def to_pascal_case(snake_name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in snake_name.split("_"))


def create_temp_jpg() -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    return path


def draw_measurements_on_image(
    calibrated_image_file: str,
    lobster: LobsterMeasurementResult,
    coin: CoinDetectionResult,
) -> None:

    image = cv2.imread(calibrated_image_file)

    if image is None:
        return

    #
    # Coin bounding box
    #
    if coin and coin.detected and coin.bounding_box:
        bb = coin.bounding_box

        cv2.rectangle(
            image,
            (bb.x, bb.y),
            (bb.x + bb.width, bb.y + bb.height),
            (255, 255, 0),      # cyan
            3,
        )

        cv2.putText(
            image,
            "COIN",
            (bb.x, bb.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    #
    # Lobster points and lines
    #
    if lobster and lobster.detected:

        p0 = lobster.p0
        p1 = lobster.p1
        p2 = lobster.p2

        if p0 and p1 and p2:

            pt0 = (int(round(p0.x)), int(round(p0.y)))
            pt1 = (int(round(p1.x)), int(round(p1.y)))
            pt2 = (int(round(p2.x)), int(round(p2.y)))

            #
            # CL line
            #
            cv2.line(
                image,
                pt0,
                pt1,
                (0, 255, 255),
                4,
                cv2.LINE_AA,
            )

            #
            # TL line
            #
            cv2.line(
                image,
                pt0,
                pt2,
                (255, 0, 255),
                4,
                cv2.LINE_AA,
            )

            #
            # Points
            #
            for name, pt in [("P0", pt0), ("P1", pt1), ("P2", pt2)]:

                cv2.circle(
                    image,
                    pt,
                    10,
                    (0, 255, 255),
                    -1,
                )

                cv2.putText(
                    image,
                    name,
                    (pt[0] + 12, pt[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            #
            # Length text
            #
            if lobster.carapace_length_mm is not None:

                mid_cl = (
                    (pt0[0] + pt1[0]) // 2,
                    (pt0[1] + pt1[1]) // 2,
                )

                text = f"CL={lobster.carapace_length_mm:.0f} mm"
                (font_w, font_h), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                2,
                )

                text_x = int(mid_cl[0] - font_w / 2)
                text_y = int(mid_cl[1] + font_h / 2) +  30

                cv2.putText(
                image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
                )

            if lobster.total_length_mm is not None:

                mid_tl = (
                    (pt1[0] + pt2[0]) // 2,
                    (pt1[1] + pt2[1]) // 2,
                )

                text = f"TL={lobster.total_length_mm:.0f} mm"
                (font_w, font_h), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                2,
                )
                text_x = int(mid_tl[0] - font_w / 2)
                text_y = int(mid_tl[1] + font_h / 2) + 30
                cv2.putText(
                image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
                )

    cv2.imwrite(
        calibrated_image_file,
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )


def try_delete_file(path: str | None) -> None:
    if not path:
        return

    file_path = Path(path)
    if not file_path.exists():
        return

    try:
        file_path.unlink()
    except (PermissionError, OSError) as ex:
        print(f"Unable to delete temp file '{path}': {ex}")


if __name__ == "__main__":
    raise SystemExit(main())
