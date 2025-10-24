"""This is the code that the rpi used and ran. It combines the pt file (AI parameters) with the google sheets API""" 

import gspread
from google.oauth2.service_account import Credentials

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


scopes = [
    "https://www.googleapis.com/auth/spreadsheets"
]
creds = Credentials.from_service_account_file("credentials.json", scopes=scopes) 
client = gspread.authorize(creds)

sheet_id = "19M_3bFODY65vMkwxtDRW74TyBPKGga4uw2Y_4RVhtjk"
sheet = client.open_by_key(sheet_id)
#Don't touch the above


num_classes = 6
model_path = r"C:\Users\wangh4\Documents\Andrew's File\Imperial Hackathon\garbage_classifier_v2.pth"
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

class GarbageClassifier(nn.Module):
    def _init_(self, num_classes):
        super()._init_()
        self.network = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

# Loading Model to model dict
device = torch.device("cpu")
model = GarbageClassifier(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
    
# Function to generalise to 3 rubbish bins 
def generalisation(rubbish_class):
    # 3 main classes are General, Paper, Plastic
    if (rubbish_class == "cardboard") or (rubbish_class == "paper"):
        return 1
    elif rubbish_class == "plastic":
        return 2
    else:
        return 3

# Inference function
def infer(photo_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]) # Function to transform Image

    # Load image
    img = Image.open(photo_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = outputs.argmax(dim=1).item()

    pred_class = classes[pred_idx]
    return pred_class



# defining sheets
Mainsheet = sheet.sheet1
Secondarysheet = sheet.worksheet("Secondary")

def to_int(value, default=0):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


# user input
def readandwrite_mainsheet(rubbish_class):
    gen = generalisation(rubbish_class)
    # Read (Main sheet)
    General_Trash = to_int(Mainsheet.cell(2, 2).value)   # row 2, col 2
    Paper_Trash   = to_int(Mainsheet.cell(2, 3).value)   # row 3, col 2
    Plastic_Trash = to_int(Mainsheet.cell(2, 4).value)   # row 4, col 2

    # write (Main sheet)
    if gen == 1:
        Paper_Trash += 1
        Mainsheet.update_cell(2, 3, str(Paper_Trash))
    elif gen == 2:
        Plastic_Trash += 1
        Mainsheet.update_cell(2, 4, str(Plastic_Trash))
    else:
        General_Trash += 1
        Mainsheet.update_cell(2, 2, str(General_Trash))

    category_list = Mainsheet.row_values(1)
    value_list = Mainsheet.row_values(2)
    print("Main sheets")
    print(category_list)
    print(value_list)



def readandwrite_secondarysheet(rubbish_class):
    # Read (Secondary sheet)
    Cardboard_Trash_2 = to_int(Secondarysheet.cell(2, 2).value)  # row 2
    Glass_Trash_2     = to_int(Secondarysheet.cell(2, 3).value)  # row 3
    Metal_Trash_2     = to_int(Secondarysheet.cell(2, 4).value)  # row 4
    Paper_Trash_2     = to_int(Secondarysheet.cell(2, 5).value)  # row 5
    Plastic_Trash_2   = to_int(Secondarysheet.cell(2, 6).value)  # row 6
    General_Trash_2   = to_int(Secondarysheet.cell(2, 7).value)  # row 7

    # write (Secondary sheet)
    if rubbish_class == "cardboard":
        Cardboard_Trash_2 += 1
        Secondarysheet.update_cell(2, 2, str(Cardboard_Trash_2))
    elif rubbish_class == "glass":
        Glass_Trash_2 += 1
        Secondarysheet.update_cell(2, 3, str(Glass_Trash_2))
    elif rubbish_class == "metal":
        Metal_Trash_2 += 1
        Secondarysheet.update_cell(2, 4, str(Metal_Trash_2))
    elif rubbish_class == "paper":
        Paper_Trash_2 += 1
        Secondarysheet.update_cell(2, 5, str(Paper_Trash_2))
    elif rubbish_class == "plastic":
        Plastic_Trash_2 += 1
        Secondarysheet.update_cell(2, 6, str(Plastic_Trash_2))
    else:
        General_Trash_2 += 1
        Secondarysheet.update_cell(2, 7, str(General_Trash_2))

    category_list2 = Secondarysheet.row_values(1)
    value_list2 = Secondarysheet.row_values(2)
    print("Secondary sheets")
    print(category_list2)
    print(value_list2)




# Usage
if _name_ == "_main_":
    test_img = r"C:\Users\wangh4\Documents\Andrew's File\Imperial Hackathon\food trash2.jpg"
    prediction = infer(test_img)
    readandwrite_mainsheet(prediction)
    readandwrite_secondarysheet(prediction)
    print(f'The model has predicted {prediction}')
