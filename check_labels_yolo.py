import os
import cv2
import shutil
from dataclasses import dataclass

@dataclass
class Target:
    class_id: int
    box: tuple
    keypoints: list

class check_labels:

    def __init__(self, path):
        self.path=path
        self.count=0
        self.image_list=[]
        self.count=0
        self.windows_name="check_labels_yolo"
        self.label_list=[]


    def get_images(self,exts=[".jpg", ".png", ".jpeg"]):
        for fname in os.listdir(self.path):
            name=os.path.join(self.path,fname)
            if os.path.isfile(name):
                ext=os.path.splitext(fname)[-1].lower()
                if ext in exts:
                    base_name = os.path.splitext(fname)[0]
                    label_path = os.path.join(self.path, base_name + ".txt")
                    if os.path.isfile(label_path):
                        self.image_list.append(name)
                        self.label_list.append(label_path)
                        self.count+=1
                    else:
                        print(f"warn : {name} not found {label_path} in this {self.path}")
        self.image_list=sorted(self.image_list)
        self.label_list=sorted(self.label_list)
        print(f"Successfully found {self.count} correct files")

    def move_to_error(self,img_path,lab_path,target_dir):
        os.makedirs(target_dir,exist_ok=True)
        shutil.move(img_path, os.path.join(target_dir, os.path.basename(img_path)))
        shutil.move(lab_path, os.path.join(target_dir, os.path.basename(lab_path)))

    def read_lab(self,lab_path):
        with open(lab_path,'r') as labs:
            content = labs.read().strip().split()
            numbers = list(map(float, content))
        def is_integer(num):
            return int(num)==num
        targets=[]
        current=[]
        for num in numbers:
            if current and is_integer(num):
                class_id = int(current[0])
                box = tuple(current[1:5])
                keypoints = current[5:]
                targets.append(Target(class_id, box, keypoints))
                current = [num]
            else:
                current.append(num)

        if current:
            class_id = int(current[0])
            box = tuple(current[1:5])
            keypoints = current[5:]
            targets.append(Target(class_id, box, keypoints))

        return targets

    def draw_yolo(self,img,lab):
        targets=self.read_lab(lab)
        img_h,img_w=img.shape[0:2]
        for target in targets:
            class_id = target.class_id
            x, y, w, h = target.box
            keypoints = target.keypoints

            x1 = int((x - w / 2) * img_w)
            y1 = int((y - h / 2) * img_h)
            x2 = int((x + w / 2) * img_w)
            y2 = int((y + h / 2) * img_h)

            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(img,str(class_id),(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),4)
            for i in range(0, len(keypoints), 2):
                kx, ky = keypoints[i:i + 2]
                px = int(kx * img_w)
                py = int(ky * img_h)
                cv2.circle(img, (px, py), 3, (255, 0, 255), -1)
        return img

    def main_window(self):
        i=0
        while 0<=i<len(self.image_list):
            img_path=self.image_list[i]
            base_name = os.path.splitext(img_path)[0]
            lab = os.path.join(self.path, base_name + ".txt")
            bad_dir = os.path.join(os.path.dirname(img_path), "bad")
            img=cv2.imread(img_path)
            result=self.draw_yolo(img,lab)
            cv2.namedWindow(self.windows_name,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.windows_name,960,540)
            cv2.imshow(self.windows_name,result)
            key=cv2.waitKey(0)
            if key==ord('a'):
                i-=1
                continue
            if key==ord('d'):
                i+=1
                continue
            if key==ord('s'):
                self.move_to_error(img_path,lab,bad_dir)
                self.image_list.pop(i)
                self.label_list.pop(i)
                if i >= len(self.image_list):
                    i = len(self.image_list) - 1
                continue
            if key == 27:
                break


if __name__ == "__main__":
    folder_path = r"D:\RoboMaster-Unity\RoboMaster-Unity\Screenshots"
    checker = check_labels(folder_path)
    checker.get_images()
    checker.main_window()




