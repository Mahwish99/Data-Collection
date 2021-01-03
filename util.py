import cv2
from xml.dom import minidom
import xml.etree.ElementTree as xml
import os
import json
import atexit

labels = open('./coco.names').read().strip().split('\n')
json_object = dict()
info_object = {
    "description": "Object detection dataset for cars",
    "url": "oursite-isdown.com",
    "version": 1.0,
    "year": 2020,
    "contributor": "Mahwish, Aaqasha, Muneeb",
    "date-created": "2020/11/1"
}
license_object = [{
    "url": "our-site-isdown.come",
    "id": 1,
    "name": "MIT Opensource license"
}]
categories_object = []
label_id = 0
for label in labels:
    label_object = dict()
    label_object["id"] = label_id
    label_object["name"] = label
    categories_object.append(label_object)
boxes_array = []
json_object["info"] = info_object
json_object["license"] = license_object
json_object["categories"] = categories_object


def voting(all_boxes, all_detected_classes):
    max_figure = -100
    box_index = 0
    for i in range(len(all_boxes)):
        if len(all_boxes[i]) > max_figure:
            max_figure = len(all_boxes[i])
            box_index = i
    boxes_max = all_boxes.pop(box_index)
    classes_max = all_detected_classes.pop(box_index)
    dic = {}
    output_list = []

    # for example yolo had max boxes
    for x in range(len(boxes_max)):
        current_box = boxes_max[x]

        # store the bounding box
        dic["box"] = current_box
        # store the class along with it's count
        dic[classes_max[x]] = 1

        upper_x_max = current_box[0][0] + 50
        upper_y_max = current_box[0][1] + 50

        lower_x_max = current_box[1][0] + 50
        lower_y_max = current_box[1][1] + 50

        upper_x_min = current_box[0][0] - 50
        upper_y_min = current_box[0][1] - 50

        lower_x_min = current_box[1][0] - 50
        lower_y_min = current_box[1][1] - 50

        for i in range(len(all_boxes)):
            boxes = all_boxes[i]
            for j in range(len(boxes)):
                other_box = boxes[j]
                # check if it's in top x range
                if (other_box[0][0] >= upper_x_min) and (other_box[0][0] <= upper_x_max):
                    # check if it's in top y range
                    if (other_box[0][1] >= upper_y_min) and (other_box[0][1] <= upper_y_max):
                        # check if it's in bottom x range
                        if (other_box[1][0] >= lower_x_min) and (other_box[1][0] <= lower_x_max):
                            # check if it's in bottom y range
                            if (other_box[1][1] >= lower_y_min) and (other_box[1][1] <= lower_y_max):
                                # match class ids
                                # increment class ids count
                                if all_detected_classes[i][j] in dic:
                                    dic[all_detected_classes[i][j]] = dic[all_detected_classes[i][j]] + 1
                                else:

                                    dic[all_detected_classes[i][j]] = 1

        output_list.append(dic.copy())
        # empty dic for next
        dic.clear()
    return output_list
def createXML(img1, output_list, j, path_xml):
    labels = open('./coco.names').read().strip().split('\n')
    root = xml.Element("Annotaions")
    for i in range(len(output_list)):
        text = ""
        box = output_list[i]["box"]
        left, top = box[0]
        right, bottom = box[1]
        cv2.rectangle(img1, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        w = int(right) - int(left)
        h = int(bottom) - int(top)
        max = 0
        for d in output_list[i]:
            if d != "box":
                if output_list[i][d] > max:
                    max = output_list[i][d]
                    class_F = labels[int(d)]
        text = text + class_F + " : " + str(max)
        cl = xml.SubElement(root, "Box")
        class_Final = xml.SubElement(cl, "Class")
        class_Final.text = class_F

        height = xml.SubElement(cl, "Height")
        height.text = str(h)

        weight = xml.SubElement(cl, "Weight")
        weight.text = str(w)
        print('Final output', 'label: ', class_F, '\theight: ', h, '\twidth: ', w)
        cv2.putText(img1, text, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    mydata = xml.tostring(root)
    dom = minidom.parseString(mydata)
    myfile = open(os.path.join(path_xml,str(j) + '.xml'), "w").write(dom.toprettyxml('\t'))
    cv2.imshow("Output", img1)
    cv2.waitKey(0)

def create_json(img, output_list, i, path_xml):
    labels = open('./coco.names').read().strip().split('\n')
    for i in range(len(output_list)):
        text = ""
        box = output_list[i]["box"]
        left, top = box[0]
        right, bottom = box[1]
        w = int(right) - int(left)
        h = int(bottom) - int(top)
        max = 0
        max_id = 0
        for d in output_list[i]:
            if d != "box":
                if output_list[i][d] > max:
                    max = output_list[i][d]
                    class_F = labels[int(d)]
                    max_id = int(d)
        box_object = dict()
        box_object["image_id"] = i
        box_object["bbox"] = [left, top, w, h]
        box_object["category_id"] = max_id
        boxes_array.append(box_object)

images_array = []
def utility(all_boxes, all_detected_classes, img, i, path_xml):
    output_list = voting(all_boxes, all_detected_classes)
    images_array.append({
        "id":i,
        "license": 1,
        "path": "path/to_path",
        "width":200,
        "height":200,
        "filename":"0.jpg"
    })
    img_json = img
    img_XML = img
    create_json(img_json, output_list, i, path_xml)
    createXML(img_XML, output_list, i, path_xml)

def write_json(data, filename='data.json'):
    with open(filename,'w') as f:
        json.dump(data, f)

def end_function():
    json_object["annotations"] = boxes_array
    json_object["images"] = images_array
    print(json_object)
    write_json(json_object, 'train.json')
