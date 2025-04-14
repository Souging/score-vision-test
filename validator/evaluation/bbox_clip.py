from argparse import ArgumentParser
from pathlib import Path
from json import load
from random import sample
from enum import Enum
from asyncio import get_event_loop, run
from logging import getLogger, basicConfig, INFO, DEBUG
from math import cos, acos
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
from time import time

from numpy import ndarray, zeros
from pydantic import BaseModel, Field
from transformers import CLIPProcessor, CLIPModel
from cv2 import VideoCapture
from torch import no_grad

FRAMES_PER_VIDEO = 750
logger = getLogger("Bounding Box Evaluation Pipeline")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
data_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class BoundingBoxObject(Enum):
    #possible classifications identified by miners and CLIP
    FOOTBALL = "football"
    GOALKEEPER = "goalkeeper"
    PLAYER = "football player"
    REFEREE = "referee"
    #possible additional classifications identified by CLIP only
    CROWD = "crowd"
    GRASS = "grass"
    GOAL = "goal"
    BACKGROUND = "background"
    BLANK = "blank"
    OTHER = "other"

OBJECT_ID_TO_ENUM = {
    0:BoundingBoxObject.FOOTBALL,
    1:BoundingBoxObject.GOALKEEPER,
    2:BoundingBoxObject.PLAYER,
    3:BoundingBoxObject.REFEREE,
}


class BBox(BaseModel):
    x1:int
    y1:int
    x2:int
    y2:int

    @property
    def width(self) -> int:
        return abs(self.x2-self.x1)

    @property
    def height(self) -> int:
        return abs(self.y2-self.y1)

class BBoxScore(BaseModel):
    predicted_label:BoundingBoxObject = Field(..., description="Object type classified by the Miner's Model")
    expected_label:BoundingBoxObject = Field(..., description="Object type classified by the Validator's Model (CLIP)")
    occurrence:int = Field(..., description="The number of times an of object of this type has been seen up to now")

    def __str__(self) -> str:
        return f"""
expected: {self.expected_label.value}
predicted: {self.predicted_label.value}
    correctness: {self.correctness}
    validity: {self.validity}
        score: {self.score}
        weight: {self.weight}
            points = {self.points}
"""

    @property
    def validity(self) -> bool:
        """Is the Object captured by the Miner
        a valid object of interest (e.g. player, goalkeeper, football, ref)
        or is it another object we don't care about?""
        """
        return self.expected_label in OBJECT_ID_TO_ENUM.values()

    @property
    def correctness(self) -> bool:
        """"Does the Object type classified by the Miner's Model
        match the classification given by the Validator?"""
        return self.predicted_label==self.expected_label

    @property
    def weight(self) -> float:
        """The first time an object of a certain type is seen the weight is 1.0
        Thereafter it decreases exponentially.
        The infinite sum of 1/(2**n) converges to 2
        Which is useful to know for normalising total scores"""
        return 1/(2**self.occurrence)

    @property
    def score(self) -> float:
        """
        1.0 = bbox is correct (e.g. a goalkeeper was captured and identified as a goalkeeper)
        0.5 = bbox is incorrect but contains a valid object (e.g. a referee was captured but identified as a goalkeeper)
        -0.5 = bbox is incorrect and contains no objects of interest (e.g. the background was captured and identified as a goalkeeper)
        """
        return float(self.correctness) or (self.validity-0.5)

    @property
    def points(self) -> float:
        return self.weight*self.score



async def stream_frames(video_path:Path):
    cap = VideoCapture(str(video_path))
    try:
        frame_count = 0
        while True:
            ret, frame = await get_event_loop().run_in_executor(None, cap.read)
            if not ret:
                break
            yield frame_count, frame
            frame_count += 1
    finally:
        cap.release()

def multiplication_factor(image_array:ndarray, bboxes:list[dict[str,int|tuple[int,int,int,int]]]) -> float:
    """Reward more targeted bbox predictions
    while penalising excessively large or numerous bbox predictions
    """
    total_area_bboxes = 0.0
    for bbox in bboxes:
        if 'bbox' not in bbox:
            continue
        x1,y1,x2,y2 = bbox['bbox']
        w = abs(x2-x1)
        h = abs(y2-y1)
        a = w*h
        total_area_bboxes += a
    height,width,_ = image_array.shape
    area_image = height*width
    logger.debug(f"Total BBox Area: {total_area_bboxes:.2f} pxl^2\nImage Area: {area_image:.2f} pxl^2")

    percentage_image_area_covered = total_area_bboxes / area_image
    scaling_factor = cos(acos(0)*max(0.0,min(1.0,percentage_image_area_covered)))
    logger.info(f"The predicted bboxes cover {percentage_image_area_covered*100:.2f}% of the image so the resulting scaling factor = {scaling_factor:.2f}")
    return scaling_factor

def batch_classify_rois(regions_of_interest:list[ndarray]) -> list[BoundingBoxObject]:
    """Use CLIP to classify a batch of images"""
    model_inputs = data_processor(
        text=[key.value for key in BoundingBoxObject],
        images=regions_of_interest,
        return_tensors="pt",
        padding=True
    ).to("cpu")
    with no_grad():
        model_outputs = clip_model(**model_inputs)
        probabilities = model_outputs.logits_per_image.softmax(dim=1)
        object_ids = probabilities.argmax(dim=1)
    logger.debug(f"Indexes predicted by CLIP: {object_ids}")
    return [
        OBJECT_ID_TO_ENUM.get(object_id.item(), BoundingBoxObject.OTHER)
        for object_id in object_ids
    ]

def extract_regions_of_interest_from_image(bboxes:list[dict[str,int|tuple[int,int,int,int]]], image_array:ndarray) -> list[ndarray]:
    bboxes_ = [
        BBox(
            x1=int(bbox['bbox'][0]),
            y1=int(bbox['bbox'][1]),
            x2=int(bbox['bbox'][2]),
            y2=int(bbox['bbox'][3])
        )
        for bbox in bboxes
    ]
    max_height = max(bbox.height for bbox in bboxes_)
    max_width = max(bbox.width for bbox in bboxes_)

    rois = []
    for bbox in bboxes_:
        roi = zeros(shape=(max_height,max_width,3),dtype="uint8")
        roi[:bbox.height,:bbox.width,:] = image_array[bbox.y1:bbox.y2,bbox.x1:bbox.x2,:]
        rois.append(roi)
        image_array[bbox.y1:bbox.y2,bbox.x1:bbox.x2,:] = zeros(shape=(bbox.height,bbox.width,3)) #We mask the bbox on the original image to prevent repeat predictions for the same object
    return rois

def evaluate_frame(
    frame_id:int,
    image_array:ndarray,
    bboxes:list[dict[str,int|tuple[int,int,int,int]]]
) -> float:
    """
    bboxes = [
      {"id": int,"class_id": int,"bbox": [x1,y1,x2,y2]},
      ...
    ]
    """
    object_counts = {
        BoundingBoxObject.FOOTBALL:0,
        BoundingBoxObject.GOALKEEPER:0,
        BoundingBoxObject.PLAYER:0,
        BoundingBoxObject.REFEREE:0,
        BoundingBoxObject.OTHER:0
    }
    rois = extract_regions_of_interest_from_image(
        bboxes=bboxes,
        image_array=image_array[:,:,::-1] #flip colour channels: BGR to RGB
    )
    expecteds = batch_classify_rois(regions_of_interest=rois)
    scores = [
        BBoxScore(
            predicted_label=OBJECT_ID_TO_ENUM[bboxes[i]['class_id']],
            expected_label=expected,
            occurrence=len([
                prior_expected for prior_expected in expecteds[:i]
                if prior_expected==expected
            ])
        )
        for i,expected in enumerate(expecteds)
    ]
    logger.debug('\n'.join(map(str,scores)))
    points = [score.points for score in scores]
    total_points = sum(points)
    n_unique_classes_detected = len(set(expecteds))
    normalised_score = total_points/n_unique_classes_detected
    scale = multiplication_factor(image_array=image_array, bboxes=bboxes)
    scaled_score = scale*normalised_score
    logger.info(f"Frame {frame_id}:\n\t-> {len(bboxes)} Bboxes predicted\n\t-> sum({', '.join(f'{point:.2f}' for point in points)}) = {total_points:.2f}\n\t-> (normalised by {n_unique_classes_detected} classes detected: [{', '.join(expected.value for expected in set(expecteds))}]) = {normalised_score:.2f}\n\t-> (Scaled by a factor of {scale:.2f}) = {scaled_score:.2f}")
    return scaled_score


async def evaluate_bboxes(prediction:dict, path_video:Path, n_frames:int, n_valid:int) -> float:
    frames = prediction
    
    # Skip evaluation if no frames or all frames are empty
    if not frames or all(not frame.get("objects") for frame in frames.values()):
        logger.warning("No valid frames with objects found in prediction — skipping evaluation.")
        return 0.0
        
    if isinstance(frames, list):
        logger.warning("Legacy formatting detected. Updating...")
        frames = {
            frame.get('frame_number',str(i)):frame
            for i, frame in enumerate(frames)
        }

    frames_ids_which_can_be_validated = [
        frame_id for frame_id,predictions in frames.items()
        if any(predictions.get('objects',[]))
    ]
    frame_ids_to_evaluate=sample(
        frames_ids_which_can_be_validated,
        k=min(n_frames,len(frames_ids_which_can_be_validated))
    )

    if len(frame_ids_to_evaluate)/n_valid<0.7:
        logger.waning(f"Only having {len(frame_ids_to_evaluate)} which is not enough for the threshold")
        return 0.0
        
    if not any(frame_ids_to_evaluate):
        logger.warning("""
            We are currently unable to validate frames with no bboxes predictions
            It may be correct that there are no objects of interest within a frame
            and so we cannot simply give a score of 0.0 for no bbox predictions
            However, with our current method, we cannot confirm nor deny this
            So any frames without any predictions are skipped in favour of those which can be verified

            However, after skipping such frames (i.e. without bbox predictions),
            there were insufficient frames remaining upon which to base an accurate evaluation
            and so were forced to return a final score of 0.0
        """)
        return 0.0

    n_threads = min(cpu_count(),len(frame_ids_to_evaluate))
    logger.info(f"Loading Video: {path_video} to evaluate {len(frame_ids_to_evaluate)} frames (using {n_threads} threads)...")
    scores = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        async for frame_id,image in stream_frames(video_path=path_video):
            if str(frame_id) not in frame_ids_to_evaluate:
                continue
            futures.append(
                executor.submit(
                    evaluate_frame,frame_id,image,frames[str(frame_id)]['objects']
                )
            )
    for future in as_completed(futures):
        try: 
            score = future.result()
            scores.append(score)
        except Exception as e:
            print(f"Error while getting score from future: {e}")

    average_score = sum(scores)/len(scores) if any(scores) else 0.0
    logger.info(f"Average Score: {average_score:.2f} when evaluated on {len(scores)} frames")
    return max(0.0,min(1.0,round(average_score,2)))
