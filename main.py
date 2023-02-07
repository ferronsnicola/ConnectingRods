import cv2
import utils
import time

imgs = []
imgs_names = ["TESI00", "TESI01", "TESI12", "TESI21", "TESI31", "TESI33", "TESI44", "TESI47", "TESI48", "TESI49", "TESI50", "TESI51", "TESI90", "TESI92", "TESI98"]

for i in range(len(imgs_names)): # ciclo per caricare tutte le immagini
    path = "imgs/" + imgs_names[i]+ ".BMP"
    imgs.append(cv2.imread(path, cv2.COLOR_GRAY2BGR))

micros = time.time() # utile per testare l'efficienza

bielle = []
for i in range(len(imgs)): # analisi di tutte le immagini caricate
    t, seg = cv2.threshold(imgs[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #seg = utils.otsu_segmentation(imgs[i])
    contours, hierarchy, seg = utils.outliers_removal(seg)
    imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR) # per un output piu espressivo
    bielle.append(utils.contours_analysis(contours, hierarchy, seg.shape, seg.dtype, imgs[i]))
    #cv2.imshow(imgs_names[i], imgs[i]) # immagini con dettagli relativi alle informazioni trovate, commentato per questioni di performance

for i in range(len(bielle)): # ciclo per stampare a console le informazioni relative ad ogni biella
    print "\n#####   " + imgs_names[i] + "   ######"
    for j in range(len(bielle[i])):
        print bielle[i][j]

print time.time() - micros

cv2.waitKey(0)
