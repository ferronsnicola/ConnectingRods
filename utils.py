import cv2
import numpy as np
import math
import rod

ANGLE_THRESH = 70 # soglia usata per detection dei punti di contatto
MAX_OUTLIER_SIZE = 10 # consideriamo outlier tutti i blob con perimetro inferiore a 10px
MAX_ROD_AREA = 6000 # soglia usata per detection di contatti tra blob
MIN_ROD_AREA = 1300 # soglia usata per detection di distrattori
N = 256*255 # dimensione delle immagini
CONTOUR_STEP = 5


########################################################
############## FUNZIONI PER SEGMENTAZIONE ##############
########################################################

def hist(img):
    result = [0]*256
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            result[img[i,j]] += 1
    return result

def otsu(img):
    h = hist(img)
    q1_0 = float(h[0])/N
    u1_0 = 0
    q2_0 = 1-q1_0
    u2_0 = 0
    for i in range(1, 256):
        temp = (i*float(h[i])/N)/q2_0
        u2_0 += temp

    u = 0
    for i in range(0, 256):
        temp = float(i*h[i])/N
        u += temp

    max_var = q1_0*(1-q1_0)*(u1_0 - u2_0)**2
    thresh = 0
    for t in range(0, len(h)-1):
        q1 = q1_0 + (float(h[t+1])/N)
        u1 = (q1_0*u1_0 + (t+1)*float(h[t+1])/N) / q1 if q1 > 0 else 0
        u2 = (u - q1*u1) / (1-q1) if q1<1 else 0

        cur_var = q1*(1-q1)*((u1 - u2)**2)
        if cur_var > max_var:
            max_var = cur_var
            thresh = t
        q1_0, u1_0, u2_0 = q1, u1, u2
    return thresh

def otsu_segmentation(img):
    t = otsu(img)
    _, result = cv2.threshold(img, t, 255, cv2.THRESH_BINARY_INV)
    return result

##################################################################







############################################################
############## FUNZIONI PER RIMOZIONE OUTLIER ##############
############################################################

def outliers_removal(seg): # algoritmo piu evoluto a livello logico e solitamente piu efficiente: filtra solo quando serve
    while True:  # CICLO CHE PULISCE GLI OUTLIER e le impurita' ai bordi dei buchi
        test = False
        _, contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if len(contours[i]) < MAX_OUTLIER_SIZE:
                test = True
                break
        if test:
            seg = cv2.medianBlur(seg, 3)
        else:
            break
    return contours, hierarchy[0], seg # openv ritorna per hierarchy una struttura inutilmente(?) incapsulata.

def outliers_removal_naive(seg):
     # alternativa piu compatta e comunque efficiente, solo meno precisa a livello logico
    for i in range(4):
        seg = cv2.medianBlur(seg, 3)
    r, contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy[0], seg

def iron_dust_removal (seg): # non utilizzata perche solitamente apre i buchi con contorni sottili, ma comunque implementata e testata
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    _, contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy[0], seg

#########################################################################################





#################################################################
############## FUNZIONI PER ANALISI DELLE IMMAGINI ##############
#################################################################

def contours_analysis(contours, hierarchy, shape, dtype, img): # itera su tutti i contorni, ricostruendo la lista di blob
    bielle = []
    for i in range(len(contours)):
        if hierarchy[i][3] == -1:  # se non ha un padre e' una biella e quindi vado a cercare i figli (buchi)
            blob_img, blob_contours = blob_analysis(contours, hierarchy, i, shape, dtype)
            x_list, _ = np.where(blob_img == 255) # lista delle ascisse di tutti i punti con livello 255 (foreground)
            area = len(x_list) # uso solo a perche sono
            if area > MAX_ROD_AREA: # gestione contatti
                print "trovato un contatto"
                imgs_and_cts = line_separator(blob_img) # ottengo immagini e contorni separati delle bielle che erano in contatto
                for j in range(len(imgs_and_cts)):
                    bielle.append(rod.Rod(imgs_and_cts[j][0], imgs_and_cts[j][1], img))
            elif area < MIN_ROD_AREA: # questa versione riduce le operazioni da fare, ma non distingue il tipo di distrattore, si puo commentare per usare l'altra versione
                print "trovato un distrattore"
            else: # caso base
                bielle.append(rod.Rod(blob_img, blob_contours, img))
    return bielle

def blob_analysis(contours, hierarchy, index, shape, dtype): # a partire da contorni, gerarchia e indice del contorno in esame, ritorna l'img del blob con i suoi contorni figli (buchi)
    children_indexes = []
    blob_contours = []
    blob_contours.append(contours[index]) # la lista di contorni inizia sempre con il contorno esterno della biella

    # recupero i figli
    for i in range(len(hierarchy)):
        if hierarchy[i][3] == index:
            children_indexes.append(i)

    blob_img = np.zeros(shape, dtype)
    cv2.drawContours(blob_img, contours, index, 255, cv2.FILLED) # disegno il blob pieno

    # per ogni figlio (buco) disegno il buco in nero
    for i in range(len(children_indexes)):
        cv2.drawContours(blob_img, contours, children_indexes[i], 0, cv2.FILLED)
        cv2.drawContours(blob_img, contours, children_indexes[i], 255, 1) # senza questo comando il buco risulterebbe privato del contorno (che e' parte della biella),
                                                                          # risulterebbe ancora piu grave nel caso di buchi con contorni molto sottili!
        blob_contours.append(contours[children_indexes[i]])

    return blob_img, blob_contours


def line_separator(img):
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # necessaria perche' ho bisogno dei soli contorni del blob da separare
    hierarchy = hierarchy[0]

    angles = [] # salvo tutti gli angoli di punti consecutivi a step di 5
    for i in range(len(contours[0])):
        third_index = i + CONTOUR_STEP if i + CONTOUR_STEP < len(contours[0]) else i + CONTOUR_STEP - len(contours[0])
        angles.append(three_points_angle(contours[0][i][0], contours[0][i - CONTOUR_STEP][0], contours[0][third_index][0]))

    min_indexes = [] # salvo gli indici degli angoli minimi
    for i in range(len(angles)): # a fine ciclo avro, per ogni contatto, i due punti estremi del contatto
        if angles[i] < ANGLE_THRESH:
            last_index = i + CONTOUR_STEP if i + CONTOUR_STEP < len(angles) else i + CONTOUR_STEP - len(angles) # l'array non e' circolare "in avanti", quindi per prendere il pixel consecutivo e' necessario questo comando
            first_index = i-CONTOUR_STEP # usando il segno "-" invece, python accede automaticamente agli ultimi valori dell'array
            minrange = np.min(angles[first_index:last_index]) # trovo il valore minimo nel range analizzato
            if minrange == angles[i]:
                min_indexes.append(i)

    # collego gli angoli vicini per separare i blob
    # per come e' scritto questo ciclo, ogni linea viene tracciata due volte, si e' scelto di non gestire l'ottimizzazione per non peggiorare ulteriormente la leggibilita' del codice
    for i in range(len(min_indexes)):
        min_dist = np.inf
        nearest_index = 0 # indice relativo al punto piu vicino a quello di indice i dell'array
        for j in range(len(min_indexes)):
            if i != j:
                dist = distance(contours[0][min_indexes[i]][0], contours[0][min_indexes[j]][0])
                if dist < min_dist:
                    min_dist = dist
                    nearest_index = j
        p1 = (contours[0][min_indexes[i]][0][0], contours[0][min_indexes[i]][0][1])
        p2 = (contours[0][min_indexes[nearest_index]][0][0], contours[0][min_indexes[nearest_index]][0][1])
        cv2.line(img, p1, p2, 0, thickness=1, lineType=4)
    # alla fine di questo ciclo avro' l'immagine con i blob separati, rimangono da gestire eventuali buchi con contorni sottili che si sono "aperti"

    imgs_cts = [] # questo e' risultato della funzione, ovvero una lista di [immagine, list(contorni)], che servono per creare l'oggetto biella
    _, s_contours, s_hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    s_hierarchy = s_hierarchy[0]
    for i in range(len(s_hierarchy)):
        if s_hierarchy[i][3] == -1:  # mi interessano solo i nuovi contorni esterni (ovvero le bielle separate, ma con eventuali buchi aperti)
            blob_img, _ = blob_analysis(s_contours, s_hierarchy, i, img.shape, img.dtype) # per ogni biella trovata creo la sua immagine
            for j in range(len(hierarchy)):
                if hierarchy[j][3] != -1: # mi interessano solo i VECCHI contorni figli (ovvero i buchi originali)
                    # dovro' capire se questi contorni appartengono alla biella in analisi e, nel caso, "ricalcarli" per tappare eventuali buchi
                    hp1, hp2 = (contours[j][0][0][1], contours[j][0][0][0]), (contours[j][len(contours[j])/2][0][1], contours[j][len(contours[j])/2][0][0])
                    # hp1 e hp2 sono due punti opposti del contorno di un buco, se almeno uno dei due risulta bianco (foreground) nell'immagine della biella,
                    # allora significa che il buco appartiene alla biella. Se ne considerano due per evitare di prendere proprio il pixel che ha causato l'apertura del buco nella fase di separazione
                    if int(blob_img[hp1]) + int(blob_img[hp2]) > 0:
                        cv2.drawContours(blob_img, contours, j, 255, 1, lineType=4)
            _, cnt, _ = cv2.findContours(blob_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # ho bisogno del contorno appena fixato
            imgs_cts.append((blob_img, cnt))
    return imgs_cts


def three_points_angle(vertex, p1, p2): # angolo individuato da tre punti
    v1 = vertex - p1
    v2 = vertex - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if cos_angle >= 1.0: # necessario perche' altrimenti spara un warning a runtime dovuto all'approssimazione del valore +/-1 in +/- 1.0000000001
        cos_angle = 1.0
    elif cos_angle <= -1.0:
        cos_angle = -1.0
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

##########################################################################




############################################################
#################### UTILS PER BIELLA ######################
############################################################

def compute_barycenter(points):
    x=0
    y=0
    for i in range(len(points)):
        x += points[i][0]
        y += points[i][1]
    xc = int(round(x/float(len(points))))
    yc = int(round(y/float(len(points))))
    return xc,yc

def draw_line(img, p, angle, length=1000, color=128, linetype=4):
    p1 = (int(round(p[0] - length * np.cos(angle))), int(round(p[1] + length * np.sin(angle))))  # due punti che serviranno per tracciare la retta
    p2 = (int(round(p[0] + length * np.cos(angle))), int(round(p[1] - length * np.sin(angle))))
    cv2.line(img, p1, p2, color, 1, linetype)


def draw_rect(img, rect, color=(255, 127, 0), lineType=8):
    cv2.line(img, rect[0], rect[1], color, 1, lineType)
    cv2.line(img, rect[0], rect[2], color, 1, lineType)
    cv2.line(img, rect[2], rect[3], color, 1, lineType)
    cv2.line(img, rect[3], rect[1], color, 1, lineType)


def distance (p0, p1):
    return math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
