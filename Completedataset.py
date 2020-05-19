
import cv2
import numpy as np
from PIL import Image
import skimage
import math

font = [cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_SCRIPT_COMPLEX] 
        # ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SCRIPT_COMPLEX

alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
             'q','r','s','t','u','v','w','x','y','z']

text = []

class ImageHandler:
    def __init__(self, img_height, img_width, fillcolor, imgid, storepath):
        self.img_height = img_height
        self.img_width = img_width
        self.num_words = 0
        self.startpos = []
        self.imgid = imgid
        self.filepath = storepath
        self.image = np.zeros((self.img_height,self.img_width,3), np.uint8)
        self.image.fill(fillcolor)
        self.wordlist = []
        self.inputwordlist = []
        self.alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
             'q','r','s','t','u','v','w','x','y','z']
        self.wordputpointdict = {}
        self.color = ()
        self.fillcolor = fillcolor
    
    def set_word_list(self, wordlist):
        self.inputwordlist = wordlist
    
    def generate_word_data(self,num_words, startpos,font,fontsize,linethickness, color, randomize = False):
        self.num_words = num_words
        self.startposition = startpos
        self.font = font
        self.fontsize = fontsize
        self.linethickness = linethickness
        self.textcolor = color
        self.wordboxlist = []
        self.wordbndboxdict = {}
        self.letterbndbox = {}
        self.color = color
        startPos = [startpos[0], startpos[1]]
        word = ""
        xmin = startPos[0]
        ymin = startPos[1]
        for i in range(self.num_words):
            if randomize:
                wordsize = np.random.randint(1,12)
                word = "".join(list(np.random.choice(alphabets,wordsize)))
            else:
                if(not len(self.inputwordlist)):
                    print("Wordlist not set...")
                    return 0
                else:
                    word = np.random.choice(self.inputwordlist)
            wordSize = cv2.getTextSize(word,self.font,self.fontsize,self.linethickness)
            wordlength = wordSize[0][0]
            wordheight = int(wordSize[0][1])
            margin = np.random.randint(5,20)
            xmax = xmin + wordlength
            ymax = ymin + wordheight + math.ceil(wordheight/3.0)
            word_x_cord = xmin
            word_y_cord = ymax - math.ceil(wordheight/3.0)
            
            if ymax >= self.img_height:
                break       
            
            elif xmax >= self.img_width:
                xmin = startpos[0]
                ymin += wordheight + margin
            
            else:
                self.wordlist.append(word)
                self.wordputpointdict[word] = (word_x_cord,word_y_cord)
                self.wordbndboxdict[word] = (xmin - 2 ,ymin - 2 ,xmax + 2 ,ymax + 2)
                xmin = xmax + margin
                
    def generate_letter_bndbox(self,drawbndbox = False):
        self.wordletterBoxdict = {}
        self.letterCrops = {}
        self.wordCrops = {}
        i = 0
        
        for word in list(self.wordlist):
            if word not in self.wordletterBoxdict.keys():
                self.wordletterBoxdict[word] = {}
            cropxmin = self.wordbndboxdict[word][0]
            cropymin = self.wordbndboxdict[word][1]
            cropxmax = self.wordbndboxdict[word][2]
            cropymax = self.wordbndboxdict[word][3]
            startposX = self.wordbndboxdict[word][0] + 2
            startposY = self.wordbndboxdict[word][1] + 2
            self.wordCrops[word] = self.image[cropymin:cropymax,cropxmin:cropxmax]
            
            for letter in word:
                labelSize = cv2.getTextSize(letter,self.font,self.fontsize,self.linethickness)
                lxmin = startposX
                lymin = startposY
                lxmax = lxmin+labelSize[0][0] - self.linethickness
                lymax = lymin+int(labelSize[0][1])
                word_height = lymax - lymin
                word_width = lxmax - lxmin
                lymax = lymax + math.ceil(word_height/3.0)
                flxmin = max((lxmin-self.linethickness),0)
                flymin = max((lymin - 5),0)
                flxmax = min(lxmax+self.linethickness,self.img_width)
                flymax = min(lymax,self.img_height)
                nimgcv = self.image[flymin:flymax,flxmin:flxmax]
                imgwl = flxmax - flxmin
                imghl = flymax - flymin
                
                self.wordletterBoxdict[word][letter] = (flxmin,flymin,flxmax,flymax)
                self.letterCrops[letter] = [nimgcv,(0,0,imgwl,imghl)]
                if drawbndbox:
                    cv2.rectangle(self.image,(flxmin,flymin),(flxmax,flymax),(0,0,255),1)
                
                startposX = lxmax
                startposY = lymin
                
        return self.wordletterBoxdict
    
    def put_words(self,randomize = False,drawbndbox = False):
        if randomize:
            selectionPercentage = np.random.randint(50,100)
            words_to_put = np.random.choice(self.wordlist,int(len(self.wordlist)*selectionPercentage/100.0))
            self.wordlist = words_to_put
        for word in self.wordlist:
            wordx = self.wordputpointdict[word][0]
            wordy = self.wordputpointdict[word][1]
            cv2.putText(self.image,word,(wordx,wordy),self.font,
                        self.fontsize,self.color,self.linethickness)
            if drawbndbox:
                box = self.wordbndboxdict[word]
                cv2.rectangle(self.image,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)
        
        
    def show_image(self):
        cv2.imshow(str(self.imgid),self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def show_words(self,num):
        selection = np.random.choice(list(self.wordCrops.keys()),num)
        for word in selection:
            cv2.imshow(word,self.wordCrops[word])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def show_letter(self,num):
        selection = np.random.choice(list(self.letterCrops.keys()),num)
        for letter in selection:
            cv2.imshow(letter,self.letterCrops[letter][0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
class ImageWordsDataset:
    
    def __init__(self,num_images):
        self.imagesobject = []
        self.labels = []
        self.num_images = num_images
        
    
    def generate_img_data(self, letterdata = False):
        for imgid in range(self.num_images):
            imgh = np.random.randint(300,600)
            imgw = np.random.randint(300,600)
            filcolor = np.random.randint(200,255)
            num_words = np.random.randint(50,200)
            startPosition = [np.random.randint(30,50), np.random.randint(30,50)]
            fontchoice = np.random.choice(font)
            fontsize = np.random.choice([1,1.5,2,2.5])
            linethickness= np.random.randint(1,3)
            self.imagesobject.append(ImageHandler(imgh, imgw,filcolor, imgid, None))
            self.imagesobject[imgid].generate_word_data(num_words, startPosition, fontchoice , fontsize, linethickness, (0,0,0),randomize = True)
            self.imagesobject[imgid].put_words(drawbndbox = False)
            if letterdata:
                self.imagesobject[imgid].generate_letter_bndbox(drawbndbox = False)
                
        return self.imagesobject
    
    def generatelabels(self,fileobject,path,classname = "text"):
        self.file = fileobject
        self.path = path
        self.classname = classname
        for imgobject in self.imagesobject:
            
            for word in imgobject.wordlist:
                xmin = imgobject.wordbndboxdict[word][0]
                ymin = imgobject.wordbndboxdict[word][1]
                xmax = imgobject.wordbndboxdict[word][2]
                ymax = imgobject.wordbndboxdict[word][3]
                datastring = path + str(imgobject.imgid) + ".jpg" + ',' + str(imgobject.img_width) +','+str(imgobject.img_height)+',' + self.classname + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) +'\n'
                
                print(datastring)
    
    def write_to_file(self):
        for imgobject in self.imagesobject:
            cv2.imwrite(self.path + str(imgobject.imgid) +".jpg",imgobject.image )
    
    def add_noise(self,percent_fraction):
        for imgobject in self.imagesobject:
            if np.random.random() > (1 - percent_fraction):
                imgobject.image = skimage.util.random_noise(imgobject.image, mode="gaussian")
                imgobject.image = np.array(imgobject.fillcolor*imgobject.image,dtype='uint8')
                imgobject.image = cv2.blur(imgobject.image,(5,5))
        
class letterBndBoxDataset:
    def __init__(self,imagesObjects):
        self.imagesObjects = imagesObjects
        
    def generate_labels(self,fileobject,path,classname = "text"):
        self.classname = classname
        self.path = path
        for imgobject in self.imagesObjects:
            for word in imgobject.wordlist:
                for letter in imgobject.wordletterBoxdict[word].keys():
                    xmin = imgobject.wordletterBoxdict[word][letter][0]
                    ymin = imgobject.wordletterBoxdict[word][letter][1]
                    xmax = imgobject.wordletterBoxdict[word][letter][2]
                    ymax = imgobject.wordletterBoxdict[word][letter][3]
                    datastring = self.path + str(imgobject.imgid) + ".jpg" + ',' + str(imgobject.img_width) +','+str(imgobject.img_height)+',' + self.classname + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) +'\n'
                    print(datastring)
    
    def write_to_file(self):
        for imgobject in self.imagesObjects:
            subid = 0
            for word in imgobject.wordCrops.keys():
                cv2.imwrite(self.path + str(imgobject.imgid) + '_' + str(subid) + ".jpg",imgobject.wordCrops[word])
                subid += 1
                
# img = ImageHandler(300,600,255,0,None)
# img.generate_word_data(100,[40,40],font[6],1.5,1,(0,0,0),True)
# print(img.wordlist)
# img.put_words(drawbndbox = False,randomize = True)
# print(img.generate_letter_bndbox(drawbndbox= True))
# img.show_image()
datafile = open("images\\traindata.csv","w")
TrainDataset = ImageWordsDataset(10)



#imgobjects = TrainDataset.generate_img_data(letterdata = True)

#TrainDataset.add_noise(0.8)

#TrainDataset.generatelabels(datafile,"images\\train\\")

#TrainDataset.write_to_file()

# LetterTrainDataset = letterBndBoxDataset(imgobjects)

# LetterTrainDataset.generate_labels(None, "images\\train\\")

# LetterTrainDataset.write_to_file()

# imgobjects[1].show_words(2)
    
