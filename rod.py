import cv2
import numpy as np
import utils
from hole import Hole


class Rod:
    def __init__(self, img, contours, father):
        self.father = father # immagine originale su cui ogni biella disegna le proprie informazioni
        self.img = img # immagine segmentata contenente la singola biella, serve sia per il calcolo del baricentro che per la larghezza al baricentro
        self.contours = contours # lista di contorni di cui il primo esterno e i seguenti i buchi
        x_list,y_list = np.where(self.img == 255)
        self.points = np.stack((y_list,x_list), axis=-1) # lista di punti necessaria per il calcolo del baricentro
        self.barycenter = utils.compute_barycenter(self.points)
        self.holes = self.find_holes_by_contour()
        self.holes = self.find_holes()
        self.test = self.check() # serve per capire se l'oggetto creato e' effettivamente una biella, non serve se si vuole discriminare tramite l'area
        if self.test[0]:
            self.orientation = self.major_axis_orientation()
            self.mer, self.length, self.width = self.compute_MER()
            self.barycenter_width, self.bw1, self.bw2 = self.compute_barycenter_width()
            self.graph()


    def check(self): # questa funzione controlla che l'oggetto Rod sia una biella analizzando i buchi, e' piu robusto ad eventuali estensioni rispetto alla discriminazione tramite l'area
        test = True, "ok"
        if len(self.holes) == 0: # se non ha buchi e' una vite
            test = False, "vite"
        elif len(self.holes) == 1: # se ha un buco e...
            if self.img[self.barycenter[1], self.barycenter[0]] == 0: # ... il baricentro dell'oggetto cade in un buco, allora e' una rondella
                test = False, "rondella"
        return test


    ######################
    ### INFO SUI BUCHI ###
    ######################
    def find_holes_by_contour(self): # se si usa questa funzione i calcoli sul buco vengono fatti OBBLIGATORIAMENTE tramite MinEnclosingCircle (stima per eccesso)
        result = []
        for i in range(1, len(self.contours)):
            result.append(Hole(None, self.contours[i]))
        return result

    def find_holes(self): # da usare SOLO se non va bene il calcolo dei buchi tramite MEC, ma si preferisce calcolarlo tramite l'area (stima per difetto)
        result = []
        for i in range(1, len(self.contours)): # creo un img vuota, gli disegno il buco pieno, creo l'oggetto buco, lo aggiungo alla lista
            img = np.zeros(self.img.shape, self.img.dtype)
            contour = self.contours[i]
            cv2.drawContours(img, self.contours, i, 255, cv2.FILLED)
            cv2.drawContours(img, self.contours, i, 0, 1)
            a,b = np.where(img == 255)
            points = np.stack((b,a), axis=-1)
            result.append(Hole(points, contour))
        return result

    ######################



    ######################################
    ### INFO ORIENTAZIONE E DIMENSIONI ###
    ######################################

    def major_axis_orientation(self):
        y, x = np.nonzero(self.img)
        x = x - np.mean(x)
        y = y - np.mean(y)
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        m = -y_v1 / float(x_v1)
        rad = np.arctan(m)
        return rad

    def compute_MER(self):
        M = np.tan(self.orientation)
        m = np.tan(self.orientation + np.pi/2)
        ic, jc = self.barycenter
        h, w = self.img.shape

        # computing lines (a,b,c)
        Mi = Mj = mi = mj = -1
        dM = dm = np.inf
        for i in range(w):
            for j in range(h):
                if j == 0 or j == h - 1 or i == 0 or i == w - 1:
                    cur_m = np.inf if j-jc==0 else (i - ic) / float(j - jc)
                    cur_dm = np.abs(cur_m - m)
                    cur_dM = np.abs(cur_m - M)
                    if cur_dM < dM:
                        Mi, Mj, dM = i, j, cur_dM
                    if cur_dm < dm:
                        mi, mj, dm = i, j, cur_dm

        # major axis: a_M*j + b_M*i + c_M = 0
        a_M = 1 / float(Mi - ic)
        b_M = 1 / float(Mj - jc)
        c_M = jc / float(Mi - ic) - ic / float(Mj - jc)

        # minor axis: a_m * j + b_m * i + c_m = 0
        a_m = 1 / float(mi - ic)
        b_m = 1 / float(mj - jc)
        c_m = jc / float(mi - ic) - ic / float(mj - jc)

        # computing MER vertices
        dMAmin = dMImin = np.inf
        dMAmax = dMImax = -np.inf
        normMA = np.sqrt(np.square(a_M) + np.square(b_M))
        normMI = np.sqrt(np.square(a_m) + np.square(b_m))
        i1 = j1 = i2 = j2 = i3 = j3 = i4 = j4 = -1
        cnt = self.contours[0]
        for p in range(len(cnt)):
            i, j = cnt[p][0]
            dMA_cur = (a_M * j + b_M * i + c_M) / float(normMA)
            dMI_cur = (a_m * j + b_m * i + c_m) / float(normMI)

            if dMA_cur < dMAmin:
                dMAmin = dMA_cur
                i1, j1 = i, j
            if dMA_cur > dMAmax:
                dMAmax = dMA_cur
                i2, j2 = i, j
            if dMI_cur < dMImin:
                dMImin = dMI_cur
                i3, j3 = i, j
            if dMI_cur > dMImax:
                dMImax = dMI_cur
                i4, j4 = i, j

        cl1 = -(a_M * j1 + b_M * i1)
        cl2 = -(a_M * j2 + b_M * i2)
        cw1 = -(a_m * j3 + b_m * i3)
        cw2 = -(a_m * j4 + b_m * i4)

        den = float(a_M * b_m - b_M * a_m)
        iv1 = int(round((a_m * cl1 - a_M * cw1) / den))
        jv1 = int(round((b_M * cw1 - b_m * cl1) / den))
        iv2 = int(round((a_m * cl1 - a_M * cw2) / den))
        jv2 = int(round((b_M * cw2 - b_m * cl1) / den))
        iv3 = int(round((a_m * cl2 - a_M * cw1) / den))
        jv3 = int(round((b_M * cw1 - b_m * cl2) / den))
        iv4 = int(round((a_m * cl2 - a_M * cw2) / den))
        jv4 = int(round((b_M * cw2 - b_m * cl2) / den))

        mer = [(iv1, jv1), (iv2, jv2), (iv3, jv3), (iv4, jv4)]
        width = np.sqrt(np.square(iv1 - iv2) + np.square(jv1 - jv2))
        length = np.sqrt(np.square(iv1 - iv3) + np.square(jv1 - jv3))
        if width > length:
            width, length = length, width
        return mer, length, width

    ##################################################################





    ###############################
    ### LARGHEZZA AL BARICENTRO ###
    ###############################

    def compute_barycenter_width(self): # interseco il contorno di una biella con una retta passante per il baricentro e perpendicolare all'orientazione del blob
        minor_axis_orientation = self.orientation + np.pi / 2
        cnt = self.contours[0] # contorno esterno della biella
        utils.draw_line(self.img, self.barycenter, minor_axis_orientation) # disegno la linea passante per il baricentro parallela al minor_axis
        pts = []
        for i in range(len(cnt)): # controllo i punti che hanno cambiato colore dopo aver disegnato la retta (ovvero le intersezioni)
            col, row = cnt[i][0]
            if self.img[row, col] != 255:
                pts.append((row,col))

        # a questo punto ho tutti i punti di intersezione (che possono essere piu di 2 dal momento che abbiamo usato necessariamente linee 4-connesse)
        # le linee 8-connesse potrebbero anche non intersecare nulla (a livello dei pixel)
        # dobbiamo vedere quale coppia di punti approssima meglio il coefficiente angolare della retta
        best_delta = np.inf
        for i in range(len(pts)):
            for j in range(len(pts)):
                if i!=j:
                    dx = float(pts[i][1] - pts[j][1])
                    m = np.inf if dx == 0 else -(pts[i][0] - pts[j][0]) / dx
                    angle = np.arctan(m)
                    if angle < 0:
                        angle += np.pi
                    delta = np.absolute(minor_axis_orientation-angle)

                    if delta < best_delta:
                        best_delta = delta
                        p1 = pts[i]
                        p2 = pts[j]
        # ho trovato la coppia di punti che meglio approssima il segmento desiderato
        distance = utils.distance(p1,p2)

        return distance, (p1[1], p1[0]), (p2[1],p2[0]) # i due punti servono per poter graficare il risultato

    ##########################################################################################################




    ##################
    ##### OUTPUT #####
    ##################

    def graph(self): # serve per mostrare alcuni dettagli utili nell'immagine
        cv2.line(self.father, self.bw1, self.bw2, (0,0,255), 1, lineType=cv2.LINE_AA)
        utils.draw_rect(self.father, self.mer, lineType=cv2.LINE_AA)

        utils.draw_line(self.father, self.barycenter, self.orientation, color=(0,0,255), length=1000, linetype=cv2.LINE_AA)
        for i in range(len(self.holes)):
            cv2.circle(self.father, self.holes[i].center, int(self.holes[i].diameter/2), (0,255,0), 1, lineType=cv2.LINE_AA)
        #cv2.imshow("biella"+str(self.barycenter), self.img) # commentato per misurare le performance

    def __str__(self):
        if self.test[0]:
            h = ""
            for i in range(len(self.holes)):
                h += "[" + str(self.holes[i]) + "]"
            result = "type = " + str(len(self.holes)) +"\n"
            result += "barycenter = " + str(self.barycenter) + "\n"
            result += "orientation = " + str(round(self.orientation, 2)) + "rad | " + str(round(self.orientation * 180 / np.pi, 2)) + "deg\n"
            result += "length = " + str(round(self.length, 2)) + "\nwidth = " + str(round(self.width, 2)) + "\nbarycenter width = " + str(round(self.barycenter_width, 2)) + "\n"
            result += "holes: " + h + "\n"
            return result
        else:
            return self.test[1]