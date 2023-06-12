# !sudo apt-get install poppler-utils #pdf2image dependency -- restart runtime/kernel after installation
# !sudo apt-get install tesseract-ocr-eng #install Tesseract OCR Engine --restart runtime/kernel after installation

# !pip install layoutparser torchvision && pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
# !pip install pdf2img
# !pip install "layoutparser[ocr]"


import pdf2image
import numpy as np
import layoutparser as lp
import torchvision.ops.boxes as bops
import torch


""" ======================================================== """

""" Layout Detection """

pdf_file= 'Three-ColumnNotes.pdf' # Adjust the filepath of your input image accordingly
img = np.asarray(pdf2image.convert_from_path(pdf_file)[0])

model1 = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

model2 = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

model3 = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

model4 = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})


""" ======================================================== """


""" Loading Images to the Detectron models """

layout_result1 = model1.detect(img)
layout_result2 = model2.detect(img)
layout_result3 = model3.detect(img)
layout_result4 = model4.detect(img)

lp.draw_box(img, layout_result1,  box_width=5, box_alpha=0.2, show_element_type=True)

lp.draw_box(img, layout_result2,  box_width=5, box_alpha=0.2, show_element_type=True)

lp.draw_box(img, layout_result3,  box_width=5, box_alpha=0.2, show_element_type=True)

lp.draw_box(img, layout_result4,  box_width=5, box_alpha=0.2, show_element_type=True)


""" ======================================================== """

""" Drawing boxes """

text_blocks1 = lp.Layout([b for b in layout_result1 ])
title_blocks1 = lp.Layout([b for b in layout_result1 if b.type=='Title'])

lp.draw_box(img, text_blocks1,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)

text_blocks2 = lp.Layout([b for b in layout_result2 ])
title_blocks2 = lp.Layout([b for b in layout_result2 if b.type=='Title'])

lp.draw_box(img, text_blocks2,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)

text_blocks3 = lp.Layout([b for b in layout_result3 ])
title_blocks3 = lp.Layout([b for b in layout_result3 if b.type=='Title'])

lp.draw_box(img, text_blocks3,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)

text_blocks4 = lp.Layout([b for b in layout_result4 ])
title_blocks4 = lp.Layout([b for b in layout_result4 if b.type=='Title'])

lp.draw_box(img, text_blocks4,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)


""" ======================================================== """


""" Sorting the Text"""

ocr_agent = lp.TesseractAgent(languages='eng')

image_width = len(img[0])

# Sort element ID of the left column based on y1 coordinate
left_interval = lp.Interval(0, image_width/2, axis='x').put_on_canvas(img)
left_blocks = text_blocks1.filter_by(left_interval, center=True)._blocks
left_blocks.sort(key = lambda b:b.coordinates[1])

# Sort element ID of the right column based on y1 coordinate
right_blocks = [b for b in text_blocks1 if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1])

# Sort the overall element ID starts from left column
text_blocks1 = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

# lp.draw_box(img, text_blocks,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)

image_width = len(img[0])

# Sort element ID of the left column based on y1 coordinate
left_interval = lp.Interval(0, image_width/2, axis='x').put_on_canvas(img)
left_blocks = text_blocks2.filter_by(left_interval, center=True)._blocks
left_blocks.sort(key = lambda b:b.coordinates[1])

# Sort element ID of the right column based on y1 coordinate
right_blocks = [b for b in text_blocks2 if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1])

# Sort the overall element ID starts from left column
text_blocks2 = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

# lp.draw_box(img, text_blocks,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)

image_width = len(img[0])

# Sort element ID of the left column based on y1 coordinate
left_interval = lp.Interval(0, image_width/2, axis='x').put_on_canvas(img)
left_blocks = text_blocks3.filter_by(left_interval, center=True)._blocks
left_blocks.sort(key = lambda b:b.coordinates[1])

# Sort element ID of the right column based on y1 coordinate
right_blocks = [b for b in text_blocks3 if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1])

# Sort the overall element ID starts from left column
text_blocks3 = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

# lp.draw_box(img, text_blocks,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)

image_width = len(img[0])

# Sort element ID of the left column based on y1 coordinate
left_interval = lp.Interval(0, image_width/2, axis='x').put_on_canvas(img)
left_blocks = text_blocks4.filter_by(left_interval, center=True)._blocks
left_blocks.sort(key = lambda b:b.coordinates[1])

# Sort element ID of the right column based on y1 coordinate
right_blocks = [b for b in text_blocks4 if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1])

# Sort the overall element ID starts from left column
text_blocks4 = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

# lp.draw_box(img, text_blocks,  box_width=5, box_alpha=0.2, show_element_type=True, show_element_id=True)


""" ======================================================== """

"""# Performing OCR"""

for block in text_blocks1:

    # Crop image around the detected layout
    segment_image = (block
                       .pad(left=15, right=15, top=5, bottom=5)
                       .crop_image(img))
    
    # Perform OCR
    text = ocr_agent.detect(segment_image)

    # Save OCR result
    block.set(text=text, inplace=True)

for block in text_blocks2:

    # Crop image around the detected layout
    segment_image = (block
                       .pad(left=15, right=15, top=5, bottom=5)
                       .crop_image(img))
    
    # Perform OCR
    text = ocr_agent.detect(segment_image)

    # Save OCR result
    block.set(text=text, inplace=True)

for block in text_blocks3:

    # Crop image around the detected layout
    segment_image = (block
                       .pad(left=15, right=15, top=5, bottom=5)
                       .crop_image(img))
    
    # Perform OCR
    text = ocr_agent.detect(segment_image)

    # Save OCR result
    block.set(text=text, inplace=True)

for block in text_blocks4:

    # Crop image around the detected layout
    segment_image = (block
                       .pad(left=15, right=15, top=5, bottom=5)
                       .crop_image(img))
    
    # Perform OCR
    text = ocr_agent.detect(segment_image)

    # Save OCR result
    block.set(text=text, inplace=True)


""" ======================================================== """

''' Printing the Text'''

for txt in text_blocks1:
    print("Text = ",txt.text)
    print("x_1=",txt.block,end='\n---\n')

for txt in text_blocks2:
    print("Text = ",txt.text)
    print("x_1=",txt.block,end='\n---\n')

for txt in text_blocks3:
    print("Text = ",txt.text)
    print("x_1=",txt.block,end='\n---\n')

for txt in text_blocks4:
    print("Text = ",txt.text)
    print("x_1=",txt.block,end='\n---\n')


""" ======================================================== """