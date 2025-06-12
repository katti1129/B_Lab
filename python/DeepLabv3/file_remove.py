import os

dataset_dir = "C:/Users/cpsla/PycharmProjects/segmentation/dataset/final_segmentation/test"

count_A=0
count_B=0
# 不要なファイルを削除
for file in os.listdir(dataset_dir):
    if file.startswith("._"):
        count_A+=1
        os.remove(os.path.join(dataset_dir, file))
    else:
        count_B+=1

print(count_A,count_B)
