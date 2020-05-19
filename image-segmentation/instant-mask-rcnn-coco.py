import pixellib
from pixellib.instance import instance_segmentation

segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5") 
# segment_image.segmentImage("path_to_image", output_image_name = "output_image_path")
segment_image.segmentImage("sample2.jpg", output_image_name = "image_new.jpg", show_bboxes = True)


"""
import pixellib
from pixellib.instance import instance_segmentation
import cv2

instance_seg = instance_segmentation()
instance_seg.load_model("mask_rcnn_coco.h5")
# segmask, output = instance_seg.segmentImage("sample2.jpg")
segmask, output = instance_seg.segmentImage("sample2.jpg", show_bboxes= True)
cv2.imwrite("img.jpg", output)
print(output.shape)
"""