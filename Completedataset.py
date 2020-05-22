
import cv2
import numpy as np
from PIL import Image
import skimage
import math

font = [cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_SCRIPT_COMPLEX] 
        # ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SCRIPT_COMPLEX

alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
             'q','r','s','t','u','v','w','x','y','z']


class ImageHandler:
    def __init__(self, img_height, img_width, fillcolor, imgid, storepath):
        self.img_height = img_height
        self.img_width = img_width
        self.num_words = 0
        self.startpos = []
        self.imgid = imgid
        self.filepath = storepath
        self.image = np.zeros((self.img_height,self.img_width,3), np.uint8)
        self.fillcolor = fillcolor
        self.image.fill(self.fillcolor)
        self.wordlist = []
        self.inputwordlist = []
        self.alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
             'q','r','s','t','u','v','w','x','y','z']
        self.wordputpointdict = {}
        self.color = ()
        
    
    def set_word_list(self, wordlist):
        self.inputwordlist = wordlist
    
    def get_random_word(self,length):
        word = "".join(list(np.random.choice(alphabets,length)))
        return(word)
    
        
    def generate_word_data(self,num_words, startpos,font,fontsize,linethickness, color, randomize = False,skip_percentage = 0):
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
        self.skip_percentage = skip_percentage
        startPos = [startpos[0], startpos[1]]
        word = ""
        xmin = startPos[0]
        ymin = startPos[1]
        index = 0
        for i in range(self.num_words):
            if randomize:
                wordlength = np.random.randint(1,12)
                word = self.get_random_word(wordlength)
                
            else:
                if(not len(self.inputwordlist)):
                    print("Wordlist not set...")
                    return 0
                else:
                    word = np.random.choice(self.inputwordlist)
            wordSize = cv2.getTextSize(word,self.font,self.fontsize,self.linethickness)
            wordlength = wordSize[0][0]
            wordheight = int(wordSize[0][1])
            margin = np.random.randint(10,25)
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
                if np.random.random() > 1 - self.skip_percentage:
                    self.wordlist.append(word)
                    self.wordputpointdict[index] = (word_x_cord,word_y_cord,word)
                    self.wordbndboxdict[index] = (xmin - 2 ,ymin - 2 ,xmax + 2 ,ymax + 2,word)
                    index += 1
                xmin = xmax + margin
                
    def generate_letter_bndbox(self,drawbndbox = False):
        self.wordletterBoxdict = {}
        self.letterCrops = {}
        self.wordCrops = {}
        index = 0
        for word in list(self.wordlist):
            if index not in self.wordletterBoxdict.keys():
                self.wordletterBoxdict[index] = [0]*len(word)
            cropxmin = self.wordbndboxdict[index][0]
            cropymin = self.wordbndboxdict[index][1]
            cropxmax = self.wordbndboxdict[index][2]
            cropymax = self.wordbndboxdict[index][3]
            startposX = self.wordbndboxdict[index][0] + 2
            startposY = self.wordbndboxdict[index][1] + 2
            self.wordCrops[index] = self.image[cropymin:cropymax,cropxmin:cropxmax]
            lxmin = 0 # startposX
            lymin = 0 # startposY
            counter = 0
            for letter in word:
                labelSize = cv2.getTextSize(letter,self.font,self.fontsize,self.linethickness)
                lxmax = lxmin+labelSize[0][0] - self.linethickness
                lymax = lymin+int(labelSize[0][1])
                word_height = lymax - lymin
                word_width = lxmax - lxmin
                lymax = lymax + math.ceil(word_height/3.0)
                flxmin = max((lxmin-self.linethickness),0)
                flymin = max((lymin - 5),0)
                flxmax = min(lxmax+self.linethickness,self.img_width)
                flymax = min(lymax,self.img_height)
                #nimgcv = self.image[flymin:flymax,flxmin:flxmax]
                imgwl = flxmax - flxmin
                imghl = flymax - flymin
                
                self.wordletterBoxdict[index][counter] = (flxmin,flymin,flxmax,flymax,letter)
                #self.letterCrops[letter] = [nimgcv,(0,0,imgwl,imghl)]
                if drawbndbox:
                    cv2.rectangle(self.wordCrops[index],(flxmin,flymin),(flxmax,flymax),(0,0,255),1)
                
                lxmin = lxmax
                lymin = lymin
                # startposX = lxmax
                # startposY = lymin
                counter += 1
            index += 1
        return self.wordletterBoxdict
    
    def put_words(self,drawbndbox = False):
        index = 0
        for word in self.wordlist:
            wordx = self.wordputpointdict[index][0]
            wordy = self.wordputpointdict[index][1]
            cv2.putText(self.image,word,(wordx,wordy),self.font,
                        self.fontsize,self.color,self.linethickness)
            if drawbndbox:
                box = self.wordbndboxdict[index]
                cv2.rectangle(self.image,(box[0],box[1]),(box[2],box[3]),(0,0,255),1)
            index += 1
        
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
        
    
    def generate_img_data(self, letterdata = False,skip_percentage = 0.5,noise_add = 0.5,word_drawbndbox = False,letter_drawbndbox = False):
        self.skip_word_percentage = skip_percentage
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
            self.imagesobject[imgid].generate_word_data(num_words, startPosition, fontchoice , fontsize, linethickness, (0,0,0),randomize = True,skip_percentage = self.skip_word_percentage)
            self.imagesobject[imgid].put_words(word_drawbndbox)
            if letterdata:
                self.imagesobject[imgid].generate_letter_bndbox(letter_drawbndbox)
            if np.random.random() >= (1 - noise_add):
                self.imagesobject[imgid].image = self.add_noise(self.imagesobject[imgid].image, self.imagesobject[imgid].fillcolor)
            if(imgid%10 == 0):
                print("Done Generating ",imgid," images..")
        return self.imagesobject
    
    def generatelabels(self,filename,path,imagespath,classname = "text"):
        self.filename = filename
        self.path = path
        self.classname = classname
        self.labelfile = open(self.path + self.filename,'w')
        datastring = 'filename'+','+ 'width'+','+ 'height'+','+ 'class'+','+ 'xmin'+','+ 'ymin'+','+ 'xmax'+','+ 'ymax'+'\n'
        self.labelfile.write(datastring)
        self.folderpath = imagespath
        for imgobject in self.imagesobject:
            index = 0
            for word in imgobject.wordlist:
                xmin = imgobject.wordbndboxdict[index][0]
                ymin = imgobject.wordbndboxdict[index][1]
                xmax = imgobject.wordbndboxdict[index][2]
                ymax = imgobject.wordbndboxdict[index][3]
                datastring = self.folderpath + str(imgobject.imgid) + ".jpg" + ',' + str(imgobject.img_width) +','+str(imgobject.img_height)+',' + self.classname + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) +'\n'
                self.labelfile.write(datastring)
                index += 1
        print("Done Writing Train Lables to file..")
        self.labelfile.close()
            
    def write_images_to_folder(self):
        i = 0
        for imgobject in self.imagesobject:
            cv2.imwrite(self.folderpath + str(imgobject.imgid) +".jpg",imgobject.image )
            if i%10 == 0:
                print("Done writing ",i," images to folder..")
            i += 1
    
    def add_noise(self,image,fillcolor):
        image = skimage.util.random_noise(image, mode="gaussian")
        image = np.array(fillcolor*image,dtype='uint8')
        image = cv2.blur(image,(3,3))
        return image
        
class letterBndBoxDataset:
    def __init__(self,imagesObjects):
        self.imagesObjects = imagesObjects
        
    def generate_labels(self,filename,path,imagespath,classname = "text"):
        self.filename = filename
        self.folderpath = imagespath
        self.classname = classname
        self.labelpath = path
        self.labelfile = open(self.labelpath + self.filename,'w')
        datastring = 'filename'+','+ 'width'+','+ 'height'+','+ 'class'+','+ 'xmin'+','+ 'ymin'+','+ 'xmax'+','+ 'ymax'+'\n'
        self.labelfile.write(datastring)
        
        for imgobject in self.imagesObjects:
            subid = 0
            index = 0
            for word in imgobject.wordlist:
                for proplist in imgobject.wordletterBoxdict[index]:
                    xmin = proplist[0]
                    ymin = proplist[1]
                    xmax = proplist[2]
                    ymax = proplist[3]
                    letter = proplist[4]
                    datastring = self.folderpath + str(imgobject.imgid) +  '_' + str(subid) + ".jpg" + ',' + str(imgobject.img_width) +','+str(imgobject.img_height)+',' + letter + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) +'\n'
                    self.labelfile.write(datastring)
                    #print(datastring)
                subid += 1
                index += 1
        print("Done Writing Test Lables to file..")
        self.labelfile.close()
    
    def write_to_file(self):
        for imgobject in self.imagesObjects:
            subid = 0
            for word in imgobject.wordCrops.keys():
                cv2.imwrite(self.folderpath + str(imgobject.imgid) + '_' + str(subid) + ".jpg",imgobject.wordCrops[word])
                subid += 1