counter = [0] * 50
import os
from torchvision.datasets import ImageFolder


folder = "Stratified_sample/"

dl_images = ImageFolder("train.resized")

for i, (x, y) in enumerate(dl_images):
    
    
    if counter[y] < 50:
        fold = folder + "/" + dl_images.classes[y]
        try:
            os.mkdir(fold)
        except FileExistsError:
#             print(f"Class {y} already exists")
            pass
        x.save(f"{fold}/img{counter[y]}.jpg")
        counter[y] += 1
    
    if i % 1000 == 0:
        print(y)
        
    
            
            