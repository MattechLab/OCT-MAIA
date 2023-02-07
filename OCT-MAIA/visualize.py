from buildDataset import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay,f1_score
from matplotlib import image
from matplotlib import pyplot as plt
from PIL import Image


def visualizePredictions(paths, mode,clf,rootpath = None,normmode = 'Z-score',onlyCtrl = True,onlyAMD = False):
    def normalize(img,mode = 'Z-score'):
        if mode == 'Z-score':
            img = dataset.z_score(img)
        elif mode == 'EQ-hist': 
            img = exposure.equalize_hist(img)
        elif mode == 'CLAHE':
            #TO Implement normalization by zone
            # Adaptive Equalization
            img = exposure.equalize_adapthist(img/max((img.max(),-img.min())), clip_limit=0.03)            
        return img
    
    def getxmax(rootpath):
        themax = 0
        allpaths =  glob.glob(os.path.join(rootpath + '\*.pickle'))
        for path in allpaths:
            PPocts = pd.read_pickle(path)
            for PPoct in PPocts:
                xnew = PPoct.image()
                if(xnew.shape[0] > themax):
                    themax = xnew.shape[0]
        return themax
    
    if mode == 'thickness':
        #xmax is the number of layers
        xmax = 5 
    elif mode =='raw':
        xmax = getxmax(rootpath) 

    for path in paths:
        try:
            group = path.split("\\").pop().split("_")[0]
            if(onlyCtrl and group =='amd' or onlyAMD and group =='control'):
                continue
            print(f'GROUP: {group}')
            PPocts = pd.read_pickle(path)
            img = Image.fromarray(PPocts[0].full_label())
            plt.imshow(img)
            plt.axis('off')
            for PPoct in PPocts:
                if ((PPoct.full_label() != img)).any():
                    plt.show()
                    print("HERE")
                    img = Image.fromarray(PPoct.full_label())
                    plt.imshow(img)
                start = PPoct.scan_data().oct_scan.start
                xstart = start[0]
                ystart = start[1]
                end = PPoct.scan_data().oct_scan.end
                xend = end[0]
                yend = end[1]
                if mode == 'thickness':
                    xnew = np.transpose(PPoct.thicknesses())
                elif mode == 'raw':
                    xnew = np.transpose(normalize(PPoct.image(),mode = normmode))
                    xnew = np.pad(xnew, ((0, 0), (xmax - xnew.shape[1], 0)), 'constant')
                ynew = PPoct.label()
                predictions = clf.predict(xnew)
                N = len(ynew)

                #to draw predictions

                seqX = [xstart + (xend - xstart)*(i/N) for i in range(N+1)] 
                seqY = [ystart + (yend - ystart)*(i/N) for i in range(N+1)] 

                # Draw predictions
                for i in range(N):
                    if predictions[i] == 1:
                        plt.plot([seqY[i],seqY[i+1]],[seqX[i],seqX[i+1]], color="red", linewidth=0.5)
                    else:
                        plt.plot([seqY[i],seqY[i+1]],[seqX[i],seqX[i+1]], color="blue", linewidth=0.5)
        except:
            print(path)
        plt.show()