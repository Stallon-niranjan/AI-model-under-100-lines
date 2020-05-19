import pixellib
from pixellib.semantic import semantic_segmentation

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 
# segment_image.segmentAsPascalvoc("path_to_image", output_image_name = "path_to_output_image")

segment_image.segmentAsPascalvoc("sample1.jpg", output_image_name = "image_new.jpg", overlay = True)



"""
import pixellib
from pixellib.semantic import semantic_segmentation
import cv2

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("pascal.h5")
# output, segmap = segment_image.segmentAsPascalvoc("sample1.jpg")
# cv2.imwrite("img.jpg", output)
# print(output.shape)
segmap, segoverlay = segment_image.segmentAsPascalvoc("sample1.jpg", overlay= True)
cv2.imwrite("img.jpg", segoverlay)
print(segoverlay.shape)
"""
