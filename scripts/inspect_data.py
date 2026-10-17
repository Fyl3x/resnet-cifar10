import torch 
import torchvision




class Dataloader():
    def load(self):
        self.thedata = torchvision.datasets.CIFAR10(root="data",train=True,download=True)
    def printer(self):
        num_of_images = len(self.thedata.data)
        image_type = self.thedata.data.dtype
        data_type = type(self.thedata)
        B,H,W,C = self.thedata.data.shape
        min_max_pixel = (self.thedata.data.min(), self.thedata.data.max())
        tragets = self.thedata.targets[:10]
        classes = self.thedata.classes
        print(f"number of images in the dataset is {num_of_images}")
        print(f"image type is : {image_type}")
        print(f"data type is : {data_type}")
        print(f"Imaghe shape is : {H,W,C}")
        print(f"first 10 classes of the targets in the DATA is : {tragets}")
        print(f"the classes in which targets are is {classes}")
        print(f"min and max are : {min_max_pixel}")







if __name__ == "__main__":
    dataload = Dataloader()
    dataload.load()
    dataload.printer()