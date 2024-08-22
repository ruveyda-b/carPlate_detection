# bu algoritma verisetimdeki 30 arabanın 12 sinin plakasini dogru bir sekilde tespit ediyor
# daha sonra gelistirecegim


# resimlere on isleme adimlarini uygulayacagiz

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# resimleri tek tek girerek im proc islemleri uygulayacagim

# Resimlerin bulunduğu dizini alıyoruz
image_dir = "images"
image_paths = os.listdir(image_dir)

image_path = os.path.join(image_dir, image_paths[0]) # ilk resmi aliyoruz

image = cv2.imread(image_path) # resmi yukluyoruz
image = cv2.resize(image,(500,500)) # resimlerde bazi uygulamaları sabit parametrelerle yapabilmek icin hepsini ayni sekilde boyutlandiriyoruz

# plt.imshow() resmi matplotlib kullanarak gösterir
# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) # BGR ı RGB donusturme yaptik (bu metot boyle calisiyor)
# plt.show() # plt.show() resmi ekrana getirir

# plakalar siyah beyaz objeler oldugundan renkli formatta calismaya ihtiyacimiz yok

img_bgr = image # renkli resmimiz
img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # resmimizi gri formata donusturduk

plt.imshow(img_gray, cmap = "gray") # plt.imshow() komutuna resmi ve hangi renk uzayinda goruntuleyecegimizi belirtmeliyiz,
                                      # color map kisaltilması cmap = "gray"
plt.show()

# kenar tespiti yapacagiz (ihtiyacımız olan kenar ve koseler)

# median blurlama kullanacagiz cunku meadian blurring kenarlara cok dokunmadan blurlama yapiyor 
# kernel size pozitif tam sayi olmalidir!
# plaka tespitinde isimize yarayacak

median_blurred_img = cv2.medianBlur(img_gray,5) # kernel size 5x5 verdik, kernel alanının altındaki tüm piksellerin medyanını alır ve merkezi eleman bu medyan değeriyle değiştirilir
median_blurred_img = cv2.medianBlur(median_blurred_img,5) # bu islemi 2 kere yaptik ki daha etkili blurlama yapalim

# yaptiklarimiza bakalim
# plt.imshow(median_blurred_img, cmap = "gray")  
# plt.show()

# resimdeki bir cok detayin kayboldugunu gorduk ve kenarlar istedigimiz gibi kalmis

# Medyan, bir görüntüdeki piksel yoğunluklarının orta değeridir.
# Medyanı kullanmak, kenar tespiti için uygun eşik değerlerini otomatik olarak ayarlamakta faydalıdır,
# çünkü medyan, görüntüdeki yoğunluk dağılımına duyarlıdır ve aşırı uç değerlerden (çok parlak veya çok karanlık alanlar) etkilenmez.
median = np.median(median_blurred_img) # piksellerin yogunluk merkezini buluyoruz

# bu çarpanlar (0.67 ve 1.33), medyan değerine göre düşük ve yüksek eşik değerlerini ayarlamak için kullanılır
# Bu oranlar, genellikle iyi bir başlangıç noktasıdır ve pek çok görüntüde etkili sonuçlar verir. 
# Ancak, her zaman belirli bir görüntüye uyarlanabilirler; bazı durumlarda bu çarpanlar değiştirilerek daha iyi sonuçlar elde edilebilir.
low = 0.67 * median
high = 1.33 * median

edges = cv2.Canny(median_blurred_img, low, high) # kenar tespiti icin kullaniriz
                                                    # low ve high Eşik Değerleri
                                                    # low değeri: Kenar olarak kabul edilebilecek en düşük piksel yoğunluğu.
                                                    # high değeri: Kenar olarak kesinlikle kabul edilen ve güçlü bir kenar olarak kabul edilebilecek piksel yoğunluğu.
# Canny algoritması, önce yüksek eşik değerini geçen pikselleri güçlü kenarlar olarak kabul eder ve ardından düşük eşik
# değerini geçen pikselleri, yüksek eşik değerine bağlı olarak kenar olarak kabul eder.

# buldugumuz kenarlara bakalim

# plt.imshow(edges, cmap = "gray")
# plt.show()

# plakanin cevresi cok ince gorunuyor yani kenar piksel yogunlugu az biz bunu arttirmak icin
# dilation yani genisletme islemi yapacagiz ve daha kalin kenarlarimiz olacak

# 3x3 luk bir kernel olusturduk ve pozitif 8 bitlik int deger kullaniyoruz --> [[1,1,1],
#                                                                               [1,1,1],
#                                                                               [1,1,1]]
# bu sekilde genisletme islemi yapilir 
# iterations bu islemin ne kadar tekrarlanmasini istedigimiz degerdir
edges = cv2.dilate(edges, np.ones((3,3), np.uint8),iterations = 1) 
# plt.imshow(edges, cmap = "gray")
# plt.show()


# KONTUR TESPITI

# cv2.RETR_TREE --> hiyerarsik yapi, en dis kontur parent, ictekiler child
# cv2.CHAIN_APPROX_SIMPLE --> tek tek pikselleri almak yerine, dikdortgensi bir yapida tam koselerin konumlarini almak icin
cnt, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cnt konturların bulunduğu bir liste veya dizidir

cnt = sorted(cnt, key = cv2.contourArea, reverse = True) # sorted(): Python'da bir listeyi belirli bir kritere göre sıralamak için kullanılan bir fonksiyondur
                                                         # key = cv2.contourArea: Sıralama kriteri olarak her konturun alanını kullanır.
                                                         # reverse = True: Sıralamanın büyükten küçüğe (azalan sırada) yapılmasını sağlar
H,W = 500,500 # yukseklik ve genislik 500x500 olacak
plate = None # henuz bir plaka tespit etmedik

for c in cnt:
    # Bu fonksiyon, verilen konturun (c) etrafına çizilebilecek minimum alanlı dikdörtgeni bulur.
    # Bu dikdörtgen, konturu tamamen kapsar ve döndürülebilir (yani, eksenlere paralel olmak zorunda değildir).
    rectangle = cv2.minAreaRect(c)
    (center_x,center_y),(width,height),r = rectangle # center_x, center_y: Dikdörtgenin merkez koordinatları (x,y) yazilabilir
                                                     # width, height: Dikdörtgenin genişliği ve yüksekliği
                                                     # r: Dikdörtgenin yatay eksene göre dönme açısı
    # 2.Adim h/w orani en az 2 olacak
    if (width>height and width> height*2) or (height>width and height>width*2):
        box = cv2.boxPoints(rectangle) # dikdortgenin 4 noktasini dondurdu boxPoints() fonksiyonu
        box = np.int64(box) # bu noktalari int degerde almaliyiz
        # kontur koordinatlarini tam ve pozitif sayi almaya dikkat etmeliyiz

        """
        eger koordinat noktalarini kendimiz bulmak isteseydik, merkez noktasini bildigimiz icin kendimiz koseleri bulabilirdik
        mesela sag alt koseyi bulalim

        sag_alt_x = center_x + w/2
        sag_alt_y = center_y + h/2 # islemiyle rahatca hesaplanabilirdi
        
        # ama zaten boxPoints() metodu bize bu degerleri dondurdugunden bu islemlere gerek yok
        """

        # max ve min noktalar alindi 
        # bu dört köşenin oluşturduğu dikdörtgenin sınırlarını belirlemiş olduk
        # bu islemleri bounding box olusturmak icin yaptik
        minx = np.min(box[:,0])
        miny = np.min(box[:,1])
        maxx = np.max(box[:,0])
        maxy = np.max(box[:,1])

        # 3. Adim 100<piksel_degeri<200

        # plaka olabilecek noktalar
        pos_plate = img_gray[miny:maxy, minx:maxx].copy()
        pos_median = np.median(pos_plate)

        check1 = pos_median > 60 and pos_median < 250 # yogunluk kontrolu (plakanın bulundugu bolgenin yogunlugu)
        check2 = height < 80 and width < 200 # yükseklik ve genislik kontrolu 
        check3 = width < 80 and height < 200 # yükseklik ve genislik kontrolu 

        print(f" pos_median :{pos_median} genislik: {width} yukseklik:{height}")

        check = False
        if(check1 and (check2 or check3)): # bu kosulları saglayan bbox plaka olur

            cv2.drawContours(image,[box],0,(0,255,0),2) # orijinal resimin üzerine yesil renk ile cizdiriyorum
            plate = [int(i) for i in [minx,miny,width,height]]    #x,y,w,h
            plt.title("plaka tespit edildi!")
            check = True
        else:
            #plaka değidir
            cv2.drawContours(image,[box],0,(0,0,255),2)
            plt.title("plaka tespit edilemedi!")
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        if check:
            break






                                                    

