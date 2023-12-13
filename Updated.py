import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import tkinter as tk
from tkinter import filedialog
import cv2
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk
import sys
import pandas as pd
from datetime import date


today = date.today()

# Load the CSV file
csv_file = 'Details.csv'


# Create a dictionary with image_number as keys and the corresponding values for actual_number, owner_name, and location
data_dict = {}
with open(csv_file, mode='r') as file1:
    reader1 = csv.reader(file1)
    next(reader1)  # Skip header row
    for row1 in reader1:
        image_number = row1[0]
        actual_number = row1[1]
        extracted_number = row1[2]
        owner_name = row1[3]
        location = row1[4]
        data_dict[image_number] = {'actual_number': actual_number, 'extracted_number': extracted_number, 'owner_name': owner_name, 'location': location}
        
        
# Load the CSV file
live_csv_file = 'live_stolen_list.csv'

# Create a dictionary with Stolen_bike_number as keys and the corresponding values for Owner_name, and Location
live_data_dict = {}
with open(live_csv_file, mode='r') as file2:
    reader2 = csv.reader(file2)
    next(reader2)  # Skip header row
    for row2 in reader2:
        Stolen_bike_number = row2[0]
        Owner_name = row2[1]
        Location = row2[2]
        
        live_data_dict[Stolen_bike_number] = {'Owner_name': Owner_name, 'Location': Location }
        




# read the CSV file into a pandas DataFrame
df = pd.read_csv('stolen_list.csv')

# extract the column with "stolen_bike_list"
stolen_bike_list_col = df['Stolen_bike_number']

# convert the column to a list
stolen_bike_list = stolen_bike_list_col.tolist()        

# stolen_bike_list = ["BROLOK8852", "B6212N044", "AP2112", "MP543959", "KLOTC4755", "B6212N04L9", "0B496", "KA04K7684", "F8474BO", "RLE88", "1825E5", "U3255MH46"]

        
net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", r"./yolov3_custom_final.weights")
classes = ['Bike', 'nohelmet', 'numberplate', 'helmet']

global file_path



file_path = None
img = None
image_label = None



def select_image():
    global file_path, img, image_label
    # Open a file dialog to select an image file
    new_file_path = filedialog.askopenfilename()
    if new_file_path:
        file_path = new_file_path
        # Load the selected image and display it in the GUI
        img = Image.open(file_path)
        img = img.resize((400, 400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        if image_label is not None:
            # Remove the previous image label from the GUI
            image_label.pack_forget()
        image_label = tk.Label(root, image=img)
        image_label.pack()







def detect_helmet():
    global file_path
    # Perform helmet detection on the selected image
    
    if file_path:
        img = cv2.imread(file_path)
        root = Tk()
        root.geometry('800x800')
        root.title("Helmet Detection")

        

        img = cv2.resize(img, (1280, 720))
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        numberplate_class_index = classes.index('numberplate')

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
                
        if len(indexes) > 0:
            for i in indexes.flatten():
                if class_ids[i] == numberplate_class_index:
                    x, y, w, h = boxes[i]
                    numberplate = img[y:y+h, x:x+w]
                    cv2.imshow('numberplate', numberplate)
                    cv2.imwrite('numberplate.jpg', numberplate)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Extract text from numberplate using EasyOCR
                    reader = easyocr.Reader(['en'])
                    result = reader.readtext('numberplate.jpg')
                    
                    
                    
                    if result:
                        reader = easyocr.Reader(['en'])
                        result = reader.readtext(numberplate, paragraph="False")
                        c=[]
                        for i in result:
                            c.append(i[-1:][0])

                        k = "".join(c)
                        l=[]
                        for i in k:
                            if i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                                l.append(i)
                            else:
                                continue

                        l_no = "".join(l)
                        labels=tk.Label(root,text=f"Numberplate text: {l_no}", font=("Arial", 14))
                        labels.pack(pady=10)
                        # print('Numberplate text:', l_no)
                        

                        # Check if the extracted number plate text matches the actual number plate text from the CSV file
                        for image_number, row_data in data_dict.items():
                            extracted_number = row_data['extracted_number']
                            
                            if l_no in extracted_number:
                                # labels = tk.Label(root, text=f"Match found for {l_no} in row {image_number}:", font=("Arial", 14))
                                # labels.pack(pady=10)
                                print(f"Match found for '{l_no}' in row {image_number}:")
                                img_num = image_number

                                # labels = tk.Label(root, text=f"Actual Number: {row_data['actual_number']}", font=("Arial", 14))
                                # labels.pack(pady=10)
                                act_num = row_data['actual_number']                            
                                print(f"Actual Number: {row_data['actual_number']}")

                                labels = tk.Label(root, text=f"Extracted Number: {row_data['extracted_number']}", font=("Arial", 14))
                                labels.pack(pady=10)
                                ext_num = row_data['extracted_number']
                                print(f"Extracted Number: {row_data['extracted_number']}")

                                #labels = tk.Label(root, text=f"Owner Name: {row_data['owner_name']}", font=("Arial", 14))
                                #labels.pack(pady=10)
                                #own_name = row_data['owner_name']
                                #print(f"Owner Name: {row_data['owner_name']}")

                                #labels = tk.Label(root, text=f"Location: {row_data['location']}", font=("Arial", 14))
                                #labels.pack(pady=10)
                                #loc = row_data['location']
                                #print(f"Location: {row_data['location']}\n")
                            

                    elif result==[]:
                        l_no = []
                        
                    else:
                        labels = tk.Label(root, text='Unable to extract numberplate text', font=("Arial", 14))
                        labels.pack(pady=10)
                        print('Unable to extract numberplate text')
                        
                    

                    break
        else:
            labels = tk.Label(root, text='Not able to detect Numberplate', font=("Arial", 14))
            labels.pack(pady=10)
            print('Not able to detect Numberplate')
            
        helmet_detected = False
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)
                if label == "helmet":
                    helmet_detected = True

        if helmet_detected:
            labels = tk.Label(root, text="Helmet detected in the image.", font=("Arial", 14))
            labels.pack(pady=10)
            print("Helmet detected in the image.")
        else:
            
            # account_sid = 'ACd3030ecce26c1dae0ba5ae2d5a3da41f'
            # auth_token = '1820b093638c123372e0d628d3258615'

            # twilio_number = '+15075955850'
            # my_phone_number = '+919075234639'

            # from twilio.rest import Client
            # # import keys

            # client = Client(account_sid, auth_token)

            # message = client.messages.create(
            #     body = f"{own_name} owner of bike {act_num} found without helmet at {loc}. Please pay the fine amount of Rs:250/-.",
            #     from_ = twilio_number,
            #     to = '+919075234639')
            # print(message.body)

            labels = tk.Label(root, text=f"{own_name} owner of bike {act_num} found without helmet at {loc}. Please pay the fine amount of Rs:250/-.", font=("Arial", 14))
            labels.pack(pady=10)
            labels = tk.Label(root, text="No helmet detected in the image.", font=("Arial", 14))
            labels.pack(pady=10)
            print(f"{own_name} owner of bike {act_num} found without helmet at {loc}. Please pay the fine amount of Rs:250/-.")
            print("No helmet detected in the image.")
            
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)
                if label == "helmet":
                    helmet_detected = True



        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

            


                    
        if l_no in stolen_bike_list:
            

            
            print(f'{act_num} stolen bike found at {loc}')
            # labels=tk.Label(root,text=f"{act_num} stolen bike found at {loc}", font=("Arial", 14))
            # labels.pack(pady=10)

            print("Bike is detected")
            # labels=tk.Label(root,text="Bike is stolen", font=("Arial", 14))
            # labels.pack(pady=10)
            
        elif l_no == []:
            labels = tk.Label(root, text="Details not found", font=("Arial", 14))
            labels.pack(pady=10)
            print('details not found')

        else:
            # labels=tk.Label(root,text="Bike is not stolen", font=("Arial", 14))
            # labels.pack()
            print("Bike is not detected") 

        root.mainloop() 

def detect_stolen_bike():
    global file_path
    # Perform helmet detection on the selected image
    
    if file_path:
        img = cv2.imread(file_path)
        root = Tk()
        root.geometry('700x700')
        root.title(" Bike Detection")

        

        img = cv2.resize(img, (1280, 720))
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        numberplate_class_index = classes.index('numberplate')

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
                
        if len(indexes) > 0:
            for i in indexes.flatten():
                if class_ids[i] == numberplate_class_index:
                    x, y, w, h = boxes[i]
                    numberplate = img[y:y+h, x:x+w]
                    cv2.imshow('numberplate', numberplate)
                    cv2.imwrite('numberplate.jpg', numberplate)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Extract text from numberplate using EasyOCR
                    reader = easyocr.Reader(['en'])
                    result = reader.readtext('numberplate.jpg')
                    
                    
                    
                    if result:
                        reader = easyocr.Reader(['en'])
                        result = reader.readtext(numberplate, paragraph="False")
                        c=[]
                        for i in result:
                            c.append(i[-1:][0])

                        k = "".join(c)
                        l=[]
                        for i in k:
                            if i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                                l.append(i)
                            else:
                                continue

                        l_no = "".join(l)
                        labels=tk.Label(root,text=f"Numberplate text: {l_no}", font=("Arial", 14))
                        labels.pack(pady=10)
                        # print('Numberplate text:', l_no)
                        

                        # Check if the extracted number plate text matches the actual number plate text from the CSV file
                        for image_number, row_data in data_dict.items():
                            extracted_number = row_data['extracted_number']
                            
                            if l_no in extracted_number:
                                # labels = tk.Label(root, text=f"Match found for {l_no} in row {image_number}:", font=("Arial", 14))
                                # labels.pack(pady=10)
                                print(f"Match found for '{l_no}' in row {image_number}:")
                                img_num = image_number

                                # labels = tk.Label(root, text=f"Actual Number: {row_data['actual_number']}", font=("Arial", 14))
                                # labels.pack(pady=10)
                                act_num = row_data['actual_number']                            
                                print(f"Actual Number: {row_data['actual_number']}")

                                labels = tk.Label(root, text=f"Extracted Number: {row_data['extracted_number']}", font=("Arial", 14))
                                labels.pack(pady=10)
                                ext_num = row_data['extracted_number']
                                print(f"Extracted Number: {row_data['extracted_number']}")

                                labels = tk.Label(root, text=f"Owner Name: {row_data['owner_name']}", font=("Arial", 14))
                                labels.pack(pady=10)
                                own_name = row_data['owner_name']
                                print(f"Owner Name: {row_data['owner_name']}")

                                labels = tk.Label(root, text=f"Location: {row_data['location']}", font=("Arial", 14))
                                labels.pack(pady=10)
                                loc = row_data['location']
                                print(f"Location: {row_data['location']}\n")
                            

                    elif result==[]:
                        l_no = []
                        
                    else:
                        labels = tk.Label(root, text='Unable to extract numberplate text', font=("Arial", 14))
                        labels.pack(pady=10)
                        print('Unable to extract numberplate text')
                        
                    

                    break
        else:
            labels = tk.Label(root, text='Not able to detect Numberplate', font=("Arial", 14))
            labels.pack(pady=10)
            print('Not able to detect Numberplate')
            
        helmet_detected = False
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)
                if label == "helmet":
                    helmet_detected = True

        if helmet_detected:
            # labels = tk.Label(root, text="Helmet detected in the image.", font=("Arial", 14))
            # labels.pack(pady=10)
            print("Helmet detected in the image.")
        else:
            

            print(f"{own_name} owner of bike {act_num} found without helmet at {loc}. Please pay the fine amount of Rs:250/-.")
            print("No helmet detected in the image.")
            
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)
                if label == "helmet":
                    helmet_detected = True



        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

            


                    
        if l_no in stolen_bike_list:
            
            # account_sid = 'AC0b34ed9209501aea1ff0c25aa940e686'
            # auth_token = 'ad900b98924a107316d24aa5c2264ada'

            # twilio_number = '+12766246047'
            # my_phone_number = '+916303158858'

            # from twilio.rest import Client
            # # import keys

            # client = Client(account_sid, auth_token)

            # message = client.messages.create(
            #     body = f'{act_num} stolen bike found at {loc}',
            #     from_ = twilio_number,
            #     to = my_phone_number)
            
            print(f'{act_num} stolen bike found at {loc}')
            labels=tk.Label(root,text=f"{act_num} stolen bike found at {loc}", font=("Arial", 14))
            labels.pack(pady=10)

            print("Bike is stolen")
            labels=tk.Label(root,text="Bike is stolen", font=("Arial", 14))
            labels.pack(pady=10)
            
        elif l_no == []:
            labels = tk.Label(root, text="Details not found", font=("Arial", 14))
            labels.pack(pady=10)
            print('details not found')

        else:
            labels=tk.Label(root,text="Bike is not stolen", font=("Arial", 14))
            labels.pack()
            print("Bike is not stolen") 

        root.mainloop()

def capture_camera():
    root = Tk()
    root.geometry('700x700')
    root.title("Capture Camera Details")
    # Start the webcam
    cap = cv2.VideoCapture(0)

    

    while True:
        # Capture an image from the webcam
        _, img = cap.read()

        # Resize the image
        img = cv2.resize(img, (1280, 720))

        # Display the image
        cv2.imshow('img', img)

        # Wait for a key to be pressed
        key = cv2.waitKey(1) & 0xFF

        # If the 'c' key is pressed, capture the image and perform object detection
        if key == ord('c'):
            # Create a blob from the image and feed it to the network
            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)

            # Get the output layers' names and run the network
            output_layers_name = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_name)

            # Define empty lists to store bounding boxes, confidences, and class IDs
            boxes = []
            confidences = []
            class_ids = []

            # Loop over each output layer and each detection to extract the information
            for output in layerOutputs:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]
                    if confidence > 0.7:
                        center_x = int(detection[0] * img.shape[1])
                        center_y = int(detection[1] * img.shape[0])
                        w = int(detection[2] * img.shape[1])
                        h = int(detection[3] * img.shape[0])
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to remove overlapping boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            helmet_detected = False        
            
            numberplate_cropped = None
            
            extracted = False

            stolen = False
            
            # If objects were detected, draw the bounding boxes and display a message
            if len(indexes) > 0:
                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(boxes), 3))
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img, label + " " + confidence, (x, y+400), font, 2, color, 2)
                    
                    
                    
                    # If a number plate is detected, crop the image and display it
                    if label == "numberplate":
                        plate_img = img[y:y+h, x:x+w]
                        cv2.imshow('Number Plate', plate_img)
                        
                        
                    
                        # Apply OCR using EasyOCR
                        import easyocr
                        reader = easyocr.Reader(['en'])
                        c=[]

                        results = reader.readtext(plate_img, paragraph="False")
                        
                        for i in results:
                            c.append(i[-1:][0])                    
                        
                        k = "".join(c)
                        l=[]

                        if len(results) > 0:
                            numberplate_text = results[0][1]
                            for i in k:
                                if i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                                    l.append(i)
                                else:
                                    continue

                            l_no = "".join(l)
                            print('Numberplate text:', l_no)
                            labels = tk.Label(root, text='Numberplate text: ' + l_no, font=('Arial', 14))
                            labels.pack(pady=10)
                            # labels=tk.Label(root,text="Bike is stolen", font=("Arial", 14))
                            # labels.pack(pady=10)
                        
                            

                            extracted = True
                        else:
                            print("No text detected in the number plate.")
                            label = tk.Label(root, text="No text detected in the number plate.", font=('Arial', 14))
                            label.pack(pady=10)


                        if l_no in live_data_dict:
                            owner_name = live_data_dict[l_no]['Owner_name']
                            location = live_data_dict[l_no]['Location']

                            stolen = True

                        else:
                            print(f"Bike {l_no} is not found in the Stolen Bike List")



                    
                # Show the image with the bounding boxes
                cv2.imshow('img', img)
                cv2.waitKey(0)
                
                # for helemt detection
                if label == "helmet":
                    helmet_detected = True

                print("Object detection complete")
                # label = tk.Label(root, text="Object detection complete", font=('Arial', 14))
                # label.pack(pady=10)

            # If no objects were detected, display a message
            else:
                print("No objects detected")
                label = tk.Label(root, text="No objects detected", font=('Arial', 14))
                label.pack(pady=10)


            # # Show the image with the bounding boxes
            # cv2.imshow('img', img)

        # If the 'q' key is pressed, quit the program
        elif key == ord('q'):
            break

            


            
    if helmet_detected:
        print("Helmet detected in the image.")
        label = tk.Label(root, text="Helmet detected in the image.", font=('Arial', 14))
        label.pack(pady=10)
    elif extracted:
        # account_sid = 'ACe1a4be35b96170ef962c16b79ebfa65f'
        # auth_token = '0330c9ccac5e8f95a739bd51503ea44c'

        # twilio_number = '+12182766347'
        # my_phone_number = '+919096327390'

        # from twilio.rest import Client
        # # import keys

        # client = Client(account_sid, auth_token)

        # message = client.messages.create(
        #     body = f"bike {numberplate_text} found without helmet. Please pay the fine amount of Rs:250/-.",
        #     from_ = twilio_number,
        #     to = '+919096327390')
        print(message.body)
        print(f"bike {numberplate_text} found without helmet. Please pay the fine amount of Rs:250/-.")
        print("No helmet detected in the image.")
        label = tk.Label(root, text=f"bike {numberplate_text} found without helmet. Please pay the fine amount of Rs:250/-.\nNo helmet detected in the image.", font=('Arial', 14))
        label.pack(pady=10)

    else:
        print("No helmet detected in the image and no extraction")
        label = tk.Label(root, text="No helmet detected in the image and no extraction", font=('Arial', 14))
        label.pack(pady=10)


    if stolen:
        print("Bike Stolen")
        # account_sid = 'AC0b34ed9209501aea1ff0c25aa940e686'
        # auth_token = 'ad900b98924a107316d24aa5c2264ada'

        # twilio_number = '+12766246047'
        # my_phone_number = '+916303158858'

        # from twilio.rest import Client
        # # import keys

        # client = Client(account_sid, auth_token)

        # message = client.messages.create(
        #     body = f"Stolen Bike {l_no} is found at {location}",
        #     from_ = twilio_number,
        #     to = '+916303158858')
        print(message.body)

        print(f" Bike {l_no} is found at {location}, owned by {owner_name}")
        label = tk.Label(root, text=f" Bike {l_no} is found at {location}, owned by {owner_name}", font=('Arial', 14))
        label.pack(pady=10)

    else:

        print(f"Bike {l_no} is not found in the  Bike List")
        label = tk.Label(root, text=f"Bike {l_no} is not found in the  Bike List", font=('Arial', 14))
        label.pack(pady=10)






    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()




# Create the Tkinter GUI
root = tk.Tk()
root.title("Helmet Detection & Number Plate Recognition")

# Set the size of the window
root.geometry("800x600")

# Set the background color of the window
root.configure(bg="grey")





# Create a label for the heading
heading_label = tk.Label(root, text="Helmet Detection & Number Plate Recognition", font=("Arial", 24), fg="#333", bg="#F5F5F5")
heading_label.pack(pady=10)

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Create the button to select an image
select_button = tk.Button(button_frame, text="Select Image", command=select_image, font=("Arial", 16), bg="#333", fg="white", bd=0, padx=20, pady=10)
select_button.pack(side=tk.LEFT, padx=10)

# Create the button to capture video from the camera
camera_button = tk.Button(button_frame, text="Capture Camera", command=capture_camera, font=("Arial", 16), bg="#333", fg="white", bd=0, padx=20, pady=10)
camera_button.pack(side=tk.LEFT, padx=10)

# Create a frame for the image
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

# Create a label for the image
image_label = tk.Label(image_frame)
image_label.pack()

# Create the button to perform helmet detection
helmet_button = tk.Button(root, text="Helmet Detection", command=detect_helmet, font=("Arial", 16), bg="#00BFFF", fg="white", bd=0, padx=20, pady=10)
helmet_button.pack(pady=10)

# Create the button to perform number plate detection
plate_button = tk.Button(root, text="Number Plate", command=detect_stolen_bike, font=("Arial", 16), bg="#DC143C", fg="white", bd=0, padx=20, pady=10)
plate_button.pack(pady=10)


# Start the GUI
root.mainloop()


